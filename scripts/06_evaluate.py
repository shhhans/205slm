"""
SQL 评估脚本
──────────────
在 Spider dev 集上测量微调模型的 Valid SQL Rate。

两层验证：
  Layer 1 - Syntax ：sqlglot 语法解析
  Layer 2 - Schema  ：表名 + 字段名对照 Spider tables.json 验证
                      子查询递归独立验证；相关子查询引用外层别名时宽松跳过

用法：
  # 完整评估（推理 + 验证）
  python scripts/06_evaluate.py --adapter output/qwen-sql-lora

  # 只跑前 100 条快速验证
  python scripts/06_evaluate.py --adapter output/qwen-sql-lora --limit 100

  # 跳过推理，直接对已有结果文件重新验证
  python scripts/06_evaluate.py --pred-file output/eval_results.jsonl

依赖：
  pip install sqlglot peft transformers tqdm
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import sqlglot
import torch
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "你是MySQL SQL生成器。根据提供的视图Schema和查询任务，"
    "生成正确的MySQL查询语句。只输出SQL，不要解释。"
)

# ── Schema 构建 ───────────────────────────────────────────────────────────────

def build_spider_schemas(tables_path: Path):
    """
    返回：
      schema_text: {db_id: "表 t1: c1(TYPE), ..."}      用于 prompt
      schema_cols: {db_id: {TABLE_UPPER: {COL_UPPER}}}   用于验证
    """
    with open(tables_path, encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_text, schema_cols = {}, {}

    for db in tables_data:
        db_id = db["db_id"]
        tables = db["table_names_original"]
        columns = db["column_names_original"]
        col_types = db.get("column_types", [])

        tbl_cols_list = defaultdict(list)
        tbl_cols_set = defaultdict(set)

        for idx, (tbl_idx, col_name) in enumerate(columns):
            if tbl_idx < 0:
                continue
            col_type = col_types[idx].upper() if idx < len(col_types) else "TEXT"
            tbl_cols_list[tbl_idx].append(f"{col_name}({col_type})")
            tbl_cols_set[tables[tbl_idx].upper()].add(col_name.upper())

        lines = [
            f"表 {tbl}: {', '.join(tbl_cols_list[i])}"
            for i, tbl in enumerate(tables)
        ]
        schema_text[db_id] = "\n".join(lines)
        schema_cols[db_id] = dict(tbl_cols_set)

    return schema_text, schema_cols


# ── SQL 提取 ──────────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    """从模型输出中提取纯 SQL，兼容 markdown 代码块和裸文本"""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


# ── 两层验证 ──────────────────────────────────────────────────────────────────

def validate_syntax(sql: str) -> tuple[bool, str | None]:
    """Layer 1：sqlglot 语法解析"""
    try:
        sqlglot.parse_one(sql, dialect="mysql")
        return True, None
    except sqlglot.errors.ParseError as e:
        return False, str(e)


def validate_schema(sql: str, db_cols: dict) -> tuple[bool, str | None]:
    """
    Layer 2：表名 + 字段名验证
    db_cols: {TABLE_NAME_UPPER: {COL_NAME_UPPER}}

    验证规则：
      - * / 空字段名           → 跳过
      - 子查询别名的字段        → 跳过（虚拟列，子查询本身单独验证）
      - 别名在当前 scope 找不到 → 跳过（相关子查询引用外层别名）
      - scope 分析异常          → 宽松通过，不误判
    """
    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except sqlglot.errors.ParseError:
        return False, "ParseError"

    try:
        for scope in traverse_scope(ast):
            # sources: {alias_upper → Table | Subquery | CTE}
            sources = {k.upper(): v for k, v in scope.sources.items()}

            for col in scope.columns:
                col_name = col.name.upper()

                # 跳过通配符和空名
                if not col_name or col_name == "*":
                    continue

                table_ref = col.table.upper() if col.table else None

                if table_ref:
                    source = sources.get(table_ref)
                    if source is None:
                        # 当前 scope 找不到 → 相关子查询引用外层，跳过
                        continue
                    if isinstance(source, exp.Table):
                        real_table = source.name.upper()
                        if real_table not in db_cols:
                            return False, f"表 '{source.name}' 不在 schema 中"
                        if col_name not in db_cols[real_table]:
                            return False, f"字段 '{col.name}' 不在表 '{source.name}' 中"
                    # else: Subquery / CTE → 虚拟列，跳过
                else:
                    # 无限定符：在 scope 内所有真实表中查找
                    real_tables = [
                        s.name.upper()
                        for s in sources.values()
                        if isinstance(s, exp.Table)
                    ]
                    if real_tables and not any(
                        col_name in db_cols.get(t, set()) for t in real_tables
                    ):
                        return False, f"字段 '{col.name}' 在当前 scope 的所有表中均不存在"

    except Exception:
        # scope 分析失败时宽松处理
        return True, None

    return True, None


# ── 模型推理 ──────────────────────────────────────────────────────────────────

def load_model(base_model: str, adapter_path: str, device: str):
    if adapter_path:
        # 微调模型：base + LoRA adapter
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # 基础模型：直接加载，不挂 LoRA
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_sql(model, tokenizer, schema: str, question: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[视图Schema]\n{schema}\n\n[查询任务]\n{question}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy decoding，评估用确定性输出
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 结果报告 ──────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    total = len(results)
    syntax_pass = sum(1 for r in results if r["syntax_valid"])
    schema_pass = sum(1 for r in results if r["schema_valid"])
    syn_errors  = [r for r in results if not r["syntax_valid"]]
    sch_errors  = [r for r in results if r["syntax_valid"] and not r["schema_valid"]]

    print("\n── 评估结果 ────────────────────────────────────────")
    print(f"  总样本数        {total}")
    print(f"  Syntax Valid    {syntax_pass}/{total}  ({syntax_pass/total*100:.1f}%)")
    print(f"  Schema Valid    {schema_pass}/{total}  ({schema_pass/total*100:.1f}%)")
    print(f"\n  语法错误样本    {len(syn_errors)}")
    print(f"  字段错误样本    {len(sch_errors)}")

    if syn_errors:
        print("\n── 语法错误示例（前3条）──────────────────────────")
        for r in syn_errors[:3]:
            print(f"  Q: {r['question'][:60]}")
            print(f"  P: {r['pred_sql'][:100]}")
            print(f"  E: {r['syntax_error']}\n")

    if sch_errors:
        print("── 字段错误示例（前3条）──────────────────────────")
        for r in sch_errors[:3]:
            print(f"  Q: {r['question'][:60]}")
            print(f"  P: {r['pred_sql'][:100]}")
            print(f"  E: {r['schema_error']}\n")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--adapter",    default="",
                        help="LoRA adapter 路径；留空则直接评估基础模型")
    parser.add_argument("--spider-dir", default="data/external/spider")
    parser.add_argument("--limit",      type=int, default=0,
                        help="评估样本数上限，0 = 全量 (~1034 条)")
    parser.add_argument("--output",     default="output/eval_results.jsonl")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--pred-file",  default="",
                        help="已有推理结果文件路径；设置此项则跳过推理，仅重新验证")
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir)
    print("加载 Spider schema...")
    schema_text, schema_cols = build_spider_schemas(spider_dir / "tables.json")

    # ── 模式 A：跳过推理，直接验证已有结果 ──────────────────────────────────
    if args.pred_file:
        print(f"从 {args.pred_file} 加载已有预测结果，重新验证...")
        with open(args.pred_file, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        results = []
        for r in tqdm(records, desc="验证中"):
            db_cols = schema_cols.get(r["db_id"], {})
            syn_ok, syn_err = validate_syntax(r["pred_sql"])
            sch_ok, sch_err = (False, "syntax failed") if not syn_ok else validate_schema(r["pred_sql"], db_cols)
            results.append({**r, "syntax_valid": syn_ok, "schema_valid": sch_ok,
                             "syntax_error": syn_err, "schema_error": sch_err})

        print_report(results)
        return

    # ── 模式 B：推理 + 验证 ───────────────────────────────────────────────────
    with open(spider_dir / "dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)
    if args.limit > 0:
        dev_data = dev_data[:args.limit]
    print(f"评估样本数：{len(dev_data)}")

    mode = "基础模型" if not args.adapter else f"微调模型（{args.adapter}）"
    print(f"加载模型：{mode}...")
    model, tokenizer = load_model(args.base_model, args.adapter, args.device)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in tqdm(dev_data, desc="推理+验证"):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            schema   = schema_text.get(db_id, f"数据库: {db_id}")
            db_cols  = schema_cols.get(db_id, {})

            raw_output = generate_sql(model, tokenizer, schema, question)
            pred_sql   = extract_sql(raw_output)

            syn_ok, syn_err = validate_syntax(pred_sql)
            sch_ok, sch_err = (False, "syntax failed") if not syn_ok else validate_schema(pred_sql, db_cols)

            record = {
                "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "pred_sql": pred_sql,
                "syntax_valid": syn_ok, "schema_valid": sch_ok,
                "syntax_error": syn_err, "schema_error": sch_err,
            }
            results.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n详细结果已保存至：{args.output}")
    print_report(results)


if __name__ == "__main__":
    main()
