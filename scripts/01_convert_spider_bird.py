"""
Spider / BIRD 数据集转换脚本
────────────────────────────
将下载的 Spider 或 BIRD 数据集转换为本项目的训练格式（LLaMA-Factory alpaca JSONL）。

使用前请先下载数据集：
  Spider: https://yale-lily.github.io/spider  → 解压到 data/external/spider/
  BIRD:   https://bird-bench.github.io/       → 解压到 data/external/bird/

运行：
  python scripts/01_convert_spider_bird.py --dataset spider
  python scripts/01_convert_spider_bird.py --dataset bird
  python scripts/01_convert_spider_bird.py --dataset both

输出：
  data/external/spider_mysql.jsonl
  data/external/bird_mysql.jsonl
"""

import json
import os
import re
import argparse
from pathlib import Path

SYSTEM_PROMPT = (
    "你是MySQL SQL生成器。根据提供的视图Schema和查询任务，"
    "生成正确的MySQL查询语句。只输出SQL，不要解释。"
)

# ── SQLite → MySQL 语法转换 ────────────────────────────────────────────────────

# 需要过滤掉的 SQLite 专属函数和语法（无法可靠转换的）
SQLITE_ONLY_PATTERNS = [
    r'\bSTRFTIME\s*\(',
    r'\bDATETIME\s*\(\s*["\']now',
    r'\bJULIANDAY\s*\(',
]

def is_sqlite_only(sql: str) -> bool:
    """检测是否包含无法转换的 SQLite 专属语法"""
    sql_upper = sql.upper()
    for pat in SQLITE_ONLY_PATTERNS:
        if re.search(pat, sql_upper):
            return True
    return False


def sqlite_to_mysql(sql: str) -> str:
    """将 SQLite SQL 语法转换为 MySQL 兼容语法"""
    # 1. 双引号标识符 → 反引号
    sql = re.sub(r'"([^"]+)"', lambda m: f"`{m.group(1)}`", sql)

    # 2. STRFTIME 日期函数转换（可靠的子集）
    #    strftime('%Y', col) → YEAR(col)
    sql = re.sub(
        r"STRFTIME\s*\(\s*'%Y'\s*,\s*([^)]+)\)",
        lambda m: f"YEAR({m.group(1).strip()})",
        sql, flags=re.IGNORECASE
    )
    #    strftime('%m', col) → MONTH(col)
    sql = re.sub(
        r"STRFTIME\s*\(\s*'%m'\s*,\s*([^)]+)\)",
        lambda m: f"MONTH({m.group(1).strip()})",
        sql, flags=re.IGNORECASE
    )
    #    strftime('%Y-%m', col) → DATE_FORMAT(col, '%Y-%m')
    sql = re.sub(
        r"STRFTIME\s*\(\s*'%Y-%m'\s*,\s*([^)]+)\)",
        lambda m: f"DATE_FORMAT({m.group(1).strip()}, '%Y-%m')",
        sql, flags=re.IGNORECASE
    )

    # 3. IFNULL → IFNULL（MySQL 支持，无需改动）

    # 4. TYPEOF() → 无 MySQL 等价，标记为需要跳过（在调用方处理）

    # 5. INTEGER PRIMARY KEY AUTOINCREMENT → 不涉及 SELECT，忽略

    # 6. 清理多余空白
    sql = re.sub(r'\s+', ' ', sql).strip()

    return sql


def schema_to_text(db_schema: dict) -> str:
    """
    将 Spider/BIRD 的 schema dict 转换为本项目的 Schema 文本格式。
    db_schema 格式（Spider tables.json 解析后）：
      {
        "table_names_original": [...],
        "column_names_original": [[table_idx, col_name], ...],
        "column_types": [...]
      }
    """
    tables = db_schema.get("table_names_original", [])
    columns = db_schema.get("column_names_original", [])
    col_types = db_schema.get("column_types", [])

    # 按表组织字段
    table_cols: dict[int, list] = {i: [] for i in range(len(tables))}
    for idx, (tbl_idx, col_name) in enumerate(columns):
        if tbl_idx < 0:  # Spider 中 -1 表示全局 *
            continue
        col_type = col_types[idx] if idx < len(col_types) else "TEXT"
        table_cols[tbl_idx].append(f"{col_name}({col_type.upper()})")

    lines = []
    for tbl_idx, tbl_name in enumerate(tables):
        cols_str = ", ".join(table_cols[tbl_idx])
        lines.append(f"表 {tbl_name}: {cols_str}")

    return "\n".join(lines)


# ── Spider 转换 ────────────────────────────────────────────────────────────────

def load_spider_schemas(spider_dir: Path) -> dict:
    """加载 Spider tables.json，返回 {db_id: schema_text} 映射"""
    tables_file = spider_dir / "tables.json"
    if not tables_file.exists():
        raise FileNotFoundError(f"未找到 {tables_file}，请确认 Spider 数据集已解压到正确目录")

    with open(tables_file, encoding="utf-8") as f:
        tables_data = json.load(f)

    schemas = {}
    for db in tables_data:
        schemas[db["db_id"]] = schema_to_text(db)
    return schemas


def convert_spider(spider_dir: Path, out_path: Path) -> int:
    """转换 Spider 训练集，返回有效样本数"""
    schemas = load_spider_schemas(spider_dir)
    samples = []

    for split in ["train_spider.json", "train_others.json"]:
        data_file = spider_dir / split
        if not data_file.exists():
            continue

        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            db_id = item.get("db_id", "")
            question = item.get("question", "").strip()
            sql = item.get("query", "").strip()

            if not question or not sql or db_id not in schemas:
                continue
            if is_sqlite_only(sql):
                continue

            mysql_sql = sqlite_to_mysql(sql)
            schema_text = schemas[db_id]

            samples.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"[视图Schema]\n{schema_text}\n\n[查询任务]\n{question}",
                "output": mysql_sql,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


# ── BIRD 转换 ─────────────────────────────────────────────────────────────────

def convert_bird(bird_dir: Path, out_path: Path) -> int:
    """转换 BIRD 训练集，返回有效样本数"""
    # BIRD 结构：train/train.json + train/train_tables.json
    train_file = bird_dir / "train" / "train.json"
    tables_file = bird_dir / "train" / "train_tables.json"

    if not train_file.exists():
        raise FileNotFoundError(f"未找到 {train_file}，请确认 BIRD 数据集已解压到正确目录")

    # 加载 Schema
    schemas = {}
    if tables_file.exists():
        with open(tables_file, encoding="utf-8") as f:
            tables_data = json.load(f)
        for db in tables_data:
            schemas[db["db_id"]] = schema_to_text(db)

    with open(train_file, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        db_id = item.get("db_id", "")
        question = item.get("question", "").strip()
        sql = item.get("SQL", "").strip()

        if not question or not sql:
            continue
        if is_sqlite_only(sql):
            continue

        mysql_sql = sqlite_to_mysql(sql)
        schema_text = schemas.get(db_id, f"数据库: {db_id}")

        # BIRD 包含 evidence 字段（额外业务知识），一并注入
        evidence = item.get("evidence", "").strip()
        task_text = question
        if evidence:
            task_text = f"{question}\n（提示：{evidence}）"

        samples.append({
            "instruction": SYSTEM_PROMPT,
            "input": f"[视图Schema]\n{schema_text}\n\n[查询任务]\n{task_text}",
            "output": mysql_sql,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spider/BIRD → MySQL JSONL 转换器")
    parser.add_argument("--dataset", choices=["spider", "bird", "both"], default="both")
    parser.add_argument("--spider-dir", default="data/external/spider")
    parser.add_argument("--bird-dir", default="data/external/bird")
    args = parser.parse_args()

    if args.dataset in ("spider", "both"):
        spider_dir = Path(args.spider_dir)
        out_path = Path("data/external/spider_mysql.jsonl")
        try:
            n = convert_spider(spider_dir, out_path)
            print(f"[Spider] 转换完成：{n} 条有效样本 → {out_path}")
        except FileNotFoundError as e:
            print(f"[Spider] 跳过：{e}")

    if args.dataset in ("bird", "both"):
        bird_dir = Path(args.bird_dir)
        out_path = Path("data/external/bird_mysql.jsonl")
        try:
            n = convert_bird(bird_dir, out_path)
            print(f"[BIRD]   转换完成：{n} 条有效样本 → {out_path}")
        except FileNotFoundError as e:
            print(f"[BIRD]   跳过：{e}")


if __name__ == "__main__":
    main()
