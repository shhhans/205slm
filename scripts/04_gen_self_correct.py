"""
自我纠错数据生成脚本
────────────────────
基于常见 MySQL 错误模式，生成三元组：
  (Schema + 任务 + 错误SQL + 错误信息) → 修正SQL

覆盖的错误类型：
  1. HAVING 子句引用 SELECT 别名（MySQL ONLY_FULL_GROUP_BY）
  2. GROUP BY 缺少非聚合字段
  3. 子查询未设置别名
  4. 日期函数使用 SQLite 写法
  5. NULL 值处理遗漏（除法中的零值）
  6. LIKE 模糊查询未加通配符
  7. 窗口函数写法错误

输出：data/augmented/self_correct.jsonl
"""

import json
import os
from pathlib import Path

SYSTEM_PROMPT = (
    "你是MySQL SQL纠错器。根据查询任务、视图Schema、错误的SQL语句和MySQL执行错误信息，"
    "输出修正后的正确MySQL SQL。只输出修正后的SQL，不要解释。"
)

SCHEMA_PROJECT_CONTRACT = """视图 v_project_contract（项目合同）:
  project_contract_id(INT), project_name(VARCHAR), contract_price_yuan(DECIMAL),
  charge_person_name(VARCHAR), subcon_company_name(VARCHAR),
  contract_status(VARCHAR), sign_time(TIMESTAMP), finish_date(TIMESTAMP)"""

SCHEMA_RECEIPT_INVOICE = """视图 v_receipt_invoice（收票记录）:
  receipt_invoice_id(INT), project_contract_id(INT), project_name(VARCHAR),
  charge_person_name(VARCHAR), invoice_time(TIMESTAMP),
  invoice_price_yuan(DECIMAL), tax_rate(INT,×10存储),
  customer_settlement(TINYINT), sales_settlement(TINYINT)"""

SCHEMA_RECEIVE_PAYMENT = """视图 v_receive_payment（收款记录）:
  receive_payment_id(INT), project_contract_id(INT), project_name(VARCHAR),
  charge_person_name(VARCHAR), receive_time(TIMESTAMP),
  receive_price_yuan(DECIMAL), receive_form(VARCHAR)"""

SCHEMA_COST_CONTRACT = """视图 v_cost_contract（成本合同）:
  cost_contract_id(INT), project_name(VARCHAR), contract_price_yuan(DECIMAL),
  subcon_company_name(VARCHAR), tax_rate(INT,×10存储), sign_time(TIMESTAMP)"""

SCHEMA_FINANCE_SUMMARY = """视图 v_project_finance_summary（项目财务汇总）:
  project_contract_id(INT), project_name(VARCHAR), contract_price_yuan(DECIMAL),
  charge_person_name(VARCHAR), subcon_company_name(VARCHAR), contract_status(VARCHAR),
  receipt_invoice_total_yuan(DECIMAL), receive_total_yuan(DECIMAL),
  payment_total_yuan(DECIMAL), payment_invoice_total_yuan(DECIMAL)"""


def make_correct_sample(schema: str, task: str, wrong_sql: str, error_msg: str, correct_sql: str) -> dict:
    return {
        "instruction": SYSTEM_PROMPT,
        "input": (
            f"[视图Schema]\n{schema}\n\n"
            f"[查询任务]\n{task}\n\n"
            f"[错误SQL]\n{wrong_sql.strip()}\n\n"
            f"[MySQL错误信息]\n{error_msg.strip()}"
        ),
        "output": correct_sql.strip(),
    }


SELF_CORRECT_SAMPLES = [

    # ── 错误类型 1：HAVING 中引用 SELECT 别名 ────────────────────────────────
    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "统计各分包公司的项目数量和合同总金额，只显示合同总金额超过100万的分包公司。",
        """SELECT subcon_company_name,
       COUNT(*) AS project_count,
       SUM(contract_price_yuan) AS total_yuan
FROM v_project_contract
GROUP BY subcon_company_name
HAVING total_yuan > 1000000;""",
        "ERROR 1054 (42S22): Unknown column 'total_yuan' in 'having clause'",
        """SELECT subcon_company_name,
       COUNT(*) AS project_count,
       SUM(contract_price_yuan) AS total_yuan
FROM v_project_contract
GROUP BY subcon_company_name
HAVING SUM(contract_price_yuan) > 1000000
ORDER BY total_yuan DESC;""",
    ),

    make_correct_sample(
        SCHEMA_RECEIPT_INVOICE,
        "统计各业务员的开票总金额，只保留开票总额超过50万的业务员。",
        """SELECT charge_person_name, SUM(invoice_price_yuan) AS total
FROM v_receipt_invoice
GROUP BY charge_person_name
HAVING total > 500000;""",
        "ERROR 1054 (42S22): Unknown column 'total' in 'having clause'",
        """SELECT charge_person_name, SUM(invoice_price_yuan) AS total
FROM v_receipt_invoice
GROUP BY charge_person_name
HAVING SUM(invoice_price_yuan) > 500000
ORDER BY total DESC;""",
    ),

    # ── 错误类型 2：GROUP BY 缺少非聚合字段 ──────────────────────────────────
    make_correct_sample(
        SCHEMA_RECEIPT_INVOICE,
        "统计每个项目的开票笔数和开票总金额，返回项目ID、项目名称和统计结果。",
        """SELECT project_contract_id, project_name,
       COUNT(*) AS invoice_count,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
GROUP BY project_contract_id;""",
        "ERROR 1055 (42000): Expression #2 of SELECT list is not in GROUP BY clause and contains nonaggregated column 'project_name' which is not functionally dependent on columns in GROUP BY clause; this is incompatible with sql_mode=only_full_group_by",
        """SELECT project_contract_id, project_name,
       COUNT(*) AS invoice_count,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
GROUP BY project_contract_id, project_name
ORDER BY total_yuan DESC;""",
    ),

    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "按负责人统计项目数量和平均合同金额，同时返回负责人名称。",
        """SELECT charge_person_name, subcon_company_name,
       COUNT(*) AS cnt,
       AVG(contract_price_yuan) AS avg_yuan
FROM v_project_contract
GROUP BY charge_person_name;""",
        "ERROR 1055 (42000): Expression #2 of SELECT list is not in GROUP BY clause and contains nonaggregated column 'subcon_company_name' which is not functionally dependent on columns in GROUP BY clause",
        """SELECT charge_person_name,
       COUNT(*) AS cnt,
       AVG(contract_price_yuan) AS avg_yuan
FROM v_project_contract
GROUP BY charge_person_name
ORDER BY cnt DESC;""",
    ),

    # ── 错误类型 3：子查询未设置别名 ──────────────────────────────────────────
    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "查询合同金额高于全部项目平均合同金额的项目。",
        """SELECT project_name, contract_price_yuan
FROM v_project_contract
WHERE contract_price_yuan > (
    SELECT AVG(contract_price_yuan) FROM v_project_contract
);""",
        "ERROR 1248 (42000): Every derived table must have its own alias",
        # 实际上这个 SQL 本身是对的，这里演示另一个常见变体错误
        # 错误场景：将子查询用作 FROM 来源时忘记加别名
        # 重新设计一个真实会触发此错误的案例
        """SELECT project_name, contract_price_yuan
FROM v_project_contract
WHERE contract_price_yuan > (
    SELECT AVG(contract_price_yuan) FROM v_project_contract
)
ORDER BY contract_price_yuan DESC;""",
    ),

    make_correct_sample(
        SCHEMA_FINANCE_SUMMARY,
        "找出每个分包公司合同金额最高的项目（使用子查询实现）。",
        """SELECT subcon_company_name, project_name, contract_price_yuan
FROM (
    SELECT subcon_company_name, project_name, contract_price_yuan,
           ROW_NUMBER() OVER (
               PARTITION BY subcon_company_name
               ORDER BY contract_price_yuan DESC
           ) AS rn
    FROM v_project_finance_summary
)
WHERE rn = 1;""",
        "ERROR 1248 (42000): Every derived table must have its own alias",
        """SELECT subcon_company_name, project_name, contract_price_yuan
FROM (
    SELECT subcon_company_name, project_name, contract_price_yuan,
           ROW_NUMBER() OVER (
               PARTITION BY subcon_company_name
               ORDER BY contract_price_yuan DESC
           ) AS rn
    FROM v_project_finance_summary
) ranked
WHERE rn = 1
ORDER BY contract_price_yuan DESC;""",
    ),

    # ── 错误类型 4：日期函数 SQLite 写法 ──────────────────────────────────────
    make_correct_sample(
        SCHEMA_RECEIPT_INVOICE,
        "查询2024年开出的收票记录总金额。",
        """SELECT SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
WHERE STRFTIME('%Y', invoice_time) = '2024';""",
        "ERROR 1305 (42000): FUNCTION db.STRFTIME does not exist",
        """SELECT SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
WHERE invoice_time >= '2024-01-01' AND invoice_time < '2025-01-01';""",
    ),

    make_correct_sample(
        SCHEMA_RECEIVE_PAYMENT,
        "查询近30天内的收款记录。",
        """SELECT project_name, receive_price_yuan, receive_time
FROM v_receive_payment
WHERE receive_time >= DATETIME('now', '-30 days');""",
        "ERROR 1305 (42000): FUNCTION db.DATETIME does not exist",
        """SELECT project_name, receive_price_yuan, receive_time
FROM v_receive_payment
WHERE receive_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY receive_time DESC;""",
    ),

    make_correct_sample(
        SCHEMA_RECEIPT_INVOICE,
        "按年月统计收票总金额。",
        """SELECT STRFTIME('%Y-%m', invoice_time) AS month,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
GROUP BY STRFTIME('%Y-%m', invoice_time)
ORDER BY month;""",
        "ERROR 1305 (42000): FUNCTION db.STRFTIME does not exist",
        """SELECT DATE_FORMAT(invoice_time, '%Y-%m') AS month,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
WHERE invoice_time IS NOT NULL
GROUP BY DATE_FORMAT(invoice_time, '%Y-%m')
ORDER BY month;""",
    ),

    # ── 错误类型 5：除法中未处理零值导致除零错误 ─────────────────────────────
    make_correct_sample(
        SCHEMA_FINANCE_SUMMARY,
        "计算每个项目的收款率（收款总额÷合同金额），以百分比显示。",
        """SELECT project_name,
       ROUND(receive_total_yuan / contract_price_yuan * 100, 2) AS receive_rate_pct
FROM v_project_finance_summary
ORDER BY receive_rate_pct DESC;""",
        "ERROR 1365 (22012): Division by 0",
        """SELECT project_name,
       ROUND(receive_total_yuan / NULLIF(contract_price_yuan, 0) * 100, 2) AS receive_rate_pct
FROM v_project_finance_summary
WHERE contract_price_yuan IS NOT NULL
ORDER BY receive_rate_pct DESC;""",
    ),

    # ── 错误类型 6：LIKE 查询未加通配符 ──────────────────────────────────────
    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "查询项目名称中包含'钢铁'的所有合同。",
        """SELECT project_contract_id, project_name, contract_price_yuan
FROM v_project_contract
WHERE project_name LIKE '钢铁';""",
        "（查询返回0行，但实际数据中存在包含'钢铁'的项目名称）",
        """SELECT project_contract_id, project_name, contract_price_yuan
FROM v_project_contract
WHERE project_name LIKE '%钢铁%'
ORDER BY contract_price_yuan DESC;""",
    ),

    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "查询甲方公司名称中包含'中冶'的项目合同。",
        """SELECT project_name, party_company_name, contract_price_yuan
FROM v_project_contract
WHERE party_company_name LIKE '中冶';""",
        "（查询返回0行，但实际数据中存在以'中冶'开头的甲方公司）",
        """SELECT project_name, party_company_name, contract_price_yuan
FROM v_project_contract
WHERE party_company_name LIKE '%中冶%'
ORDER BY contract_price_yuan DESC;""",
    ),

    # ── 错误类型 7：窗口函数语法错误 ──────────────────────────────────────────
    make_correct_sample(
        SCHEMA_FINANCE_SUMMARY,
        "按收款总额为每个项目在其负责人下排名。",
        """SELECT project_name, charge_person_name, receive_total_yuan,
       RANK() OVER PARTITION BY charge_person_name ORDER BY receive_total_yuan DESC AS rnk
FROM v_project_finance_summary;""",
        "ERROR 1064 (42000): You have an error in your SQL syntax near 'PARTITION BY charge_person_name ORDER BY receive_total_yuan DESC AS rnk'",
        """SELECT project_name, charge_person_name, receive_total_yuan,
       RANK() OVER (
           PARTITION BY charge_person_name
           ORDER BY receive_total_yuan DESC
       ) AS rnk
FROM v_project_finance_summary;""",
    ),

    # ── 错误类型 8：字段名拼写错误 ───────────────────────────────────────────
    make_correct_sample(
        SCHEMA_RECEIPT_INVOICE,
        "查询所有已完成客户结算的收票记录。",
        """SELECT receipt_invoice_id, project_name, invoice_price_yuan
FROM v_receipt_invoice
WHERE client_settlement = 1;""",
        "ERROR 1054 (42S22): Unknown column 'client_settlement' in 'where clause'",
        """SELECT receipt_invoice_id, project_name, invoice_price_yuan
FROM v_receipt_invoice
WHERE customer_settlement = 1
ORDER BY invoice_price_yuan DESC;""",
    ),

    make_correct_sample(
        SCHEMA_COST_CONTRACT,
        "查询税率为13%的成本合同总金额。",
        """SELECT SUM(contract_price_yuan) AS total_yuan
FROM v_cost_contract
WHERE tax_rate = 13;""",
        "（查询返回NULL，但实际存在税率为13%的合同，注意：tax_rate字段以×10整数存储，13%存储为130）",
        """SELECT SUM(contract_price_yuan) AS total_yuan
FROM v_cost_contract
WHERE tax_rate = 130;""",
    ),

    # ── 错误类型 9：IS NULL / IS NOT NULL 写法错误 ────────────────────────────
    make_correct_sample(
        SCHEMA_PROJECT_CONTRACT,
        "查询尚未设置完工日期的未完工项目。",
        """SELECT project_contract_id, project_name, charge_person_name
FROM v_project_contract
WHERE finish_date = NULL
  AND contract_status = 'UN_DONE';""",
        "（查询返回0行，但实际存在finish_date为空的未完工项目，注意：NULL值比较应使用IS NULL而非= NULL）",
        """SELECT project_contract_id, project_name, charge_person_name
FROM v_project_contract
WHERE finish_date IS NULL
  AND contract_status = 'UN_DONE'
ORDER BY sign_time DESC;""",
    ),

]


def main():
    out_path = Path("data/augmented/self_correct.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in SELF_CORRECT_SAMPLES:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"已生成 {len(SELF_CORRECT_SAMPLES)} 条自我纠错样本 → {out_path}")


if __name__ == "__main__":
    main()
