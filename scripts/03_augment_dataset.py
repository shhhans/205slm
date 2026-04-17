"""
数据增强脚本
────────────
对业务种子数据（business_seed.jsonl）进行以下几类增强：

1. 同义词替换（鲁棒性）
   相同查询意图用不同自然语言表达，期望生成相同 SQL

2. 混合 Schema 粒度（字段无注释 vs 有注释混用）
   部分样本去掉字段类型注释，模拟"仅字段名"的输入风格

3. 时间表达多样化
   "今年" / "本年度" / "2024年" 等不同时间表达 → 同一 SQL 逻辑

4. 干扰视图注入（Schema Linking 鲁棒性）
   在正确 Schema 中随机混入 1~2 个无关视图片段，
   验证模型不会错误使用干扰视图的字段

输出：data/augmented/augmented.jsonl
"""

import json
import random
import re
import os
from pathlib import Path
from copy import deepcopy

random.seed(42)

SYSTEM_PROMPT = (
    "你是MySQL SQL生成器。根据提供的视图Schema和查询任务，"
    "生成正确的MySQL查询语句。只输出SQL，不要解释。"
)

# ── 1. 同义词替换表 ───────────────────────────────────────────────────────────

SYNONYM_GROUPS = [
    # 时间表达
    (r"查询2024年", ["查询2024年度", "统计2024全年", "列出2024年的"]),
    (r"近30天", ["最近30天", "过去30天", "近一个月"]),
    (r"近12个月", ["最近12个月", "过去一年", "近一年"]),
    (r"第一季度（1月至3月）", ["Q1（一月到三月）", "上半年第一季度", "1月到3月"]),

    # 动词/表达方式
    (r"^查询", ["列出", "找出", "获取", "返回"]),
    (r"^统计", ["汇总", "计算", "求"]),
    (r"返回前(\d+)条", [r"只取前\1条", r"限制\1条结果", r"取前\1名"]),

    # 金额表达
    (r"超过(\d+)万元", [r"大于\1万元", r"高于\1万", r"\1万以上"]),
    (r"超过(\d+)00万", [r"大于\1百万", r"\1百万以上"]),

    # 字段别名表达
    (r"合同金额", ["合同价款", "合同总额", "签约金额"]),
    (r"收款总额", ["实收金额", "到账金额", "收到的款项总计"]),
    (r"开票金额", ["发票金额", "票面金额"]),
    (r"付款总额", ["实付金额", "已付金额", "支付金额合计"]),
    (r"负责人", ["经办人", "责任人", "业务负责人"]),
    (r"分包公司", ["分包商", "承包商", "施工单位"]),
    (r"甲方公司", ["甲方", "发包方", "建设单位"]),
]


def apply_synonyms(task: str, n_variants: int = 2) -> list[str]:
    """对 task 文本随机应用同义词替换，生成 n_variants 个变体"""
    variants = []
    for _ in range(n_variants):
        new_task = task
        # 随机选 1~3 个替换规则
        rules = random.sample(SYNONYM_GROUPS, k=min(3, len(SYNONYM_GROUPS)))
        for pattern, replacements in rules:
            if re.search(pattern, new_task):
                replacement = random.choice(replacements)
                # 处理反向引用（\1）
                try:
                    new_task = re.sub(pattern, replacement, new_task, count=1)
                except re.error:
                    pass
        if new_task != task:
            variants.append(new_task)
    return variants


# ── 2. Schema 粒度变换 ────────────────────────────────────────────────────────

def strip_type_annotations(schema_text: str) -> str:
    """去掉字段后面的类型注释，模拟"仅字段名"风格"""
    # 去掉括号中的类型信息，如 field(INT) → field
    return re.sub(r'\((?:INT|BIGINT|VARCHAR|DECIMAL|TIMESTAMP|TINYINT|CHAR)[^)]*\)', '', schema_text)


def partial_strip_annotations(schema_text: str, ratio: float = 0.5) -> str:
    """随机去掉约 ratio 比例的字段类型注释"""
    def maybe_strip(m):
        if random.random() < ratio:
            return m.group(1)  # 只保留字段名
        return m.group(0)      # 保留完整

    return re.sub(r'(\w+)\([A-Z,×\d\s\.]+\)', maybe_strip, schema_text)


# ── 3. 干扰视图注入 ───────────────────────────────────────────────────────────

# 用于干扰的视图片段（不在正确查询范围内的视图）
DISTRACTOR_SCHEMAS = [
    "视图 v_user（系统用户）:\n  user_id(INT), user_name(VARCHAR), email(VARCHAR), last_login_time(TIMESTAMP)",
    "视图 v_subcon_company（分包公司）:\n  subcon_company_id(INT), company_name(VARCHAR), tel(VARCHAR), bank_account(VARCHAR)",
    "视图 v_party_company（甲方公司）:\n  party_company_id(INT), company_name(VARCHAR), address(VARCHAR), tax_number(VARCHAR)",
    "视图 v_refund（退款）:\n  refund_id(INT), refund_type(VARCHAR), refund_money_yuan(DECIMAL), refund_time(TIMESTAMP)",
    "视图 v_salesman_summary（业务员汇总）:\n  user_id(INT), salesman_name(VARCHAR), project_count(BIGINT), receive_total_yuan(DECIMAL)",
]


def inject_distractor(schema_text: str, n: int = 1) -> str:
    """在现有 Schema 中随机插入 n 个干扰视图"""
    distractors = random.sample(DISTRACTOR_SCHEMAS, k=min(n, len(DISTRACTOR_SCHEMAS)))
    # 随机插入位置（开头或结尾）
    if random.random() < 0.5:
        return "\n\n".join(distractors) + "\n\n" + schema_text
    else:
        return schema_text + "\n\n" + "\n\n".join(distractors)


# ── 4. 时间表达标准化变体 ─────────────────────────────────────────────────────

TIME_VARIANTS = [
    # (task_pattern, sql_pattern, [(new_task, new_sql), ...])
    (
        r"2024年第一季度（1月至3月）",
        r"invoice_time >= '2024-01-01' AND invoice_time < '2024-04-01'",
        [
            ("2024年Q1", "invoice_time >= '2024-01-01' AND invoice_time < '2024-04-01'"),
            ("2024年一季度", "invoice_time >= '2024-01-01' AND invoice_time < '2024-04-01'"),
        ]
    ),
    (
        r"近30天内",
        r"last_login_time >= DATE_SUB\(NOW\(\), INTERVAL 30 DAY\)",
        [
            ("过去30天内", "last_login_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"),
            ("最近一个月内", "last_login_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"),
        ]
    ),
]


# ── 主增强流程 ────────────────────────────────────────────────────────────────

def augment(samples: list[dict]) -> list[dict]:
    augmented = []

    for sample in samples:
        task = ""
        schema_text = ""

        # 从 input 中解析出 schema 和 task
        raw_input = sample.get("input", "")
        schema_match = re.search(r'\[视图Schema\]\n(.*?)\n\n\[查询任务\]', raw_input, re.DOTALL)
        task_match = re.search(r'\[查询任务\]\n(.*)', raw_input, re.DOTALL)

        if not schema_match or not task_match:
            continue

        schema_text = schema_match.group(1)
        task = task_match.group(1).strip()
        sql = sample.get("output", "")

        # ── 增强 A：同义词替换（每条生成 2 个变体）──────────────────────────
        for variant_task in apply_synonyms(task, n_variants=2):
            augmented.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"[视图Schema]\n{schema_text}\n\n[查询任务]\n{variant_task}",
                "output": sql,
            })

        # ── 增强 B：Schema 粒度变换（去掉类型注释）──────────────────────────
        stripped_schema = strip_type_annotations(schema_text)
        if stripped_schema != schema_text:
            augmented.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"[视图Schema]\n{stripped_schema}\n\n[查询任务]\n{task}",
                "output": sql,
            })

        # ── 增强 C：部分去注释（50% 字段保留类型）──────────────────────────
        partial_schema = partial_strip_annotations(schema_text, ratio=0.5)
        if partial_schema != schema_text:
            augmented.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"[视图Schema]\n{partial_schema}\n\n[查询任务]\n{task}",
                "output": sql,
            })

        # ── 增强 D：干扰视图注入（约 30% 的样本）────────────────────────────
        if random.random() < 0.30:
            n_distractors = random.choice([1, 2])
            distracted_schema = inject_distractor(schema_text, n=n_distractors)
            augmented.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"[视图Schema]\n{distracted_schema}\n\n[查询任务]\n{task}",
                "output": sql,
            })

    return augmented


def main():
    seed_path = Path("data/seed/business_seed.jsonl")
    out_path = Path("data/augmented/augmented.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(seed_path, encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    augmented = augment(seeds)
    random.shuffle(augmented)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in augmented:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"原始种子：{len(seeds)} 条")
    print(f"增强后：  {len(augmented)} 条 → {out_path}")


if __name__ == "__main__":
    main()
