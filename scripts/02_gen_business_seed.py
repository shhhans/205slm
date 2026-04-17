"""
业务种子数据生成脚本
基于真实 14 个视图，手工构造覆盖全部视图和主要查询类型的高质量样本。
输出: data/seed/business_seed.jsonl (LLaMA-Factory alpaca 格式)
"""

import json
import os

SYSTEM_PROMPT = (
    "你是MySQL SQL生成器。根据提供的视图Schema和查询任务，"
    "生成正确的MySQL查询语句。只输出SQL，不要解释。"
)

# ── Schema 片段定义（每条样本只注入相关视图，节省 token）─────────────────────

SCHEMAS = {
    "v_project_contract": """视图 v_project_contract（项目合同）:
  project_contract_id(INT), contract_no(VARCHAR), project_name(VARCHAR),
  contract_price_yuan(DECIMAL), tax_rate(INT,×10存储,如130=13%),
  charge_person_id(INT), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  party_company_id(INT), party_company_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  sign_time(TIMESTAMP), contract_start_time(TIMESTAMP), contract_end_time(TIMESTAMP),
  project_start_time(TIMESTAMP), project_end_time(TIMESTAMP),
  contract_status(VARCHAR), settlement_type(VARCHAR),
  finish_date(TIMESTAMP), is_permission(TINYINT),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_customer": """视图 v_customer（客户）:
  customer_id(INT), customer_name(VARCHAR), phone_number(VARCHAR),
  email(VARCHAR), customer_scale(VARCHAR), region(VARCHAR),
  customer_category(VARCHAR), owner_name(VARCHAR), owner_id(INT),
  big_customer(TINYINT), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_party_company": """视图 v_party_company（甲方公司）:
  party_company_id(INT), company_name(VARCHAR), address(VARCHAR),
  tax_number(VARCHAR), tel(VARCHAR), bank_account(VARCHAR),
  bank_name(VARCHAR), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_cost_contract": """视图 v_cost_contract（成本合同）:
  cost_contract_id(INT), contract_no(VARCHAR), cost_contract_no(VARCHAR),
  project_contract_id(INT), project_name(VARCHAR),
  contract_price_yuan(DECIMAL), tax_rate(INT,×10存储),
  charge_person_id(INT), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  sign_time(TIMESTAMP), contract_status(VARCHAR),
  status(TINYINT), finish_date(TIMESTAMP),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_payment": """视图 v_payment（付款记录）:
  payment_id(INT), project_contract_id(INT), cost_contract_id(INT),
  project_name(VARCHAR), charge_person_id(INT), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  party_company_id(INT), party_company_name(VARCHAR),
  payment_time(TIMESTAMP), payment_price_yuan(DECIMAL),
  payment_form(VARCHAR), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_payment_invoice": """视图 v_payment_invoice（付款发票）:
  payment_invoice_id(INT), project_contract_id(INT), cost_contract_id(INT),
  project_name(VARCHAR), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  invoice_time(TIMESTAMP), invoice_price_yuan(DECIMAL),
  tax_rate(INT,×10存储), tax_yuan(DECIMAL),
  invoice_type(VARCHAR), invoice_class(VARCHAR),
  is_deduction(TINYINT), deduction_status(INT),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_receipt_invoice": """视图 v_receipt_invoice（收票记录）:
  receipt_invoice_id(INT), project_contract_id(INT),
  project_name(VARCHAR), charge_person_id(INT), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  party_company_id(INT), party_company_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  invoice_time(TIMESTAMP), invoice_price_yuan(DECIMAL),
  tax_rate(INT,×10存储), tax_yuan(DECIMAL),
  invoice_type(VARCHAR), invoice_class(VARCHAR),
  customer_settlement(TINYINT), sales_settlement(TINYINT),
  company_settlement(TINYINT), is_deduction(TINYINT),
  pay_price_yuan(DECIMAL), settlement_date(TIMESTAMP),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_receive_payment": """视图 v_receive_payment（收款记录）:
  receive_payment_id(INT), project_contract_id(INT), receipt_invoice_id(INT),
  project_name(VARCHAR), charge_person_id(INT), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  party_company_id(INT), party_company_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  receive_time(TIMESTAMP), receive_price_yuan(DECIMAL),
  receive_form(VARCHAR), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_project_finance_summary": """视图 v_project_finance_summary（项目财务汇总，已聚合，每个项目一行）:
  project_contract_id(INT), contract_no(VARCHAR), project_name(VARCHAR),
  contract_price_yuan(DECIMAL), charge_person_name(VARCHAR),
  party_company_name(VARCHAR), subcon_company_name(VARCHAR),
  contract_status(VARCHAR),
  receipt_invoice_count(BIGINT), receipt_invoice_total_yuan(DECIMAL),
  payment_invoice_count(BIGINT), payment_invoice_total_yuan(DECIMAL),
  receive_count(BIGINT), receive_total_yuan(DECIMAL),
  payment_count(BIGINT), payment_total_yuan(DECIMAL),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_salesman_summary": """视图 v_salesman_summary（业务员业绩汇总，已聚合，每人一行）:
  user_id(INT), salesman_name(VARCHAR), employee_no(VARCHAR),
  project_count(BIGINT), project_total_yuan(DECIMAL),
  receipt_invoice_count(BIGINT), receipt_invoice_total_yuan(DECIMAL),
  receive_count(BIGINT), receive_total_yuan(DECIMAL),
  payment_count(BIGINT), payment_total_yuan(DECIMAL)""",

    "v_refund": """视图 v_refund（退款记录）:
  refund_id(INT), refund_type(VARCHAR), business_id(INT),
  refund_time(TIMESTAMP), refund_account(VARCHAR), refund_form(VARCHAR),
  refund_money_yuan(DECIMAL), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_refund_invoice": """视图 v_refund_invoice（退票记录）:
  refund_invoice_id(INT), project_contract_id(INT), cost_contract_id(INT),
  project_name(VARCHAR), charge_person_name(VARCHAR),
  customer_id(INT), customer_name(VARCHAR),
  party_company_id(INT), party_company_name(VARCHAR),
  subcon_company_id(INT), subcon_company_name(VARCHAR),
  refund_type(VARCHAR), invoice_time(TIMESTAMP),
  invoice_price_yuan(DECIMAL), refund_price_yuan(DECIMAL),
  refund_time(TIMESTAMP), tax_rate(INT,×10存储),
  settlement(TINYINT), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_subcon_company": """视图 v_subcon_company（分包公司）:
  subcon_company_id(INT), company_name(VARCHAR), address(VARCHAR),
  tax_number(VARCHAR), tel(VARCHAR), bank_account(VARCHAR),
  bank_name(VARCHAR), create_time(TIMESTAMP), update_time(TIMESTAMP)""",

    "v_user": """视图 v_user（系统用户/业务员）:
  user_id(INT), user_name(VARCHAR), phone_number(VARCHAR),
  email(VARCHAR), employee_no(VARCHAR), account(VARCHAR),
  settlement_type(VARCHAR), last_login_time(TIMESTAMP),
  create_time(TIMESTAMP), update_time(TIMESTAMP)""",
}


def make_sample(views: list, task: str, sql: str) -> dict:
    schema_text = "\n\n".join(SCHEMAS[v] for v in views)
    return {
        "instruction": SYSTEM_PROMPT,
        "input": f"[视图Schema]\n{schema_text}\n\n[查询任务]\n{task}",
        "output": sql.strip(),
    }


# ── 种子样本 ─────────────────────────────────────────────────────────────────

SEEDS = [

    # ── v_project_contract：单表查询 ──────────────────────────────────────────
    make_sample(
        ["v_project_contract"],
        "查询所有状态为未完工（contract_status = 'UN_DONE'）的项目合同，返回合同ID、项目名称、合同金额和负责人姓名。",
        """SELECT project_contract_id, project_name, contract_price_yuan, charge_person_name
FROM v_project_contract
WHERE contract_status = 'UN_DONE';"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询2023年签署的项目合同数量。",
        """SELECT COUNT(*) AS contract_count
FROM v_project_contract
WHERE sign_time >= '2023-01-01' AND sign_time < '2024-01-01';"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询合同金额超过500万元的项目，按合同金额从高到低排序，返回前10条。",
        """SELECT project_contract_id, project_name, contract_price_yuan, charge_person_name
FROM v_project_contract
WHERE contract_price_yuan > 5000000
ORDER BY contract_price_yuan DESC
LIMIT 10;"""
    ),

    make_sample(
        ["v_project_contract"],
        "统计每位负责人的项目合同数量和合同总金额，按总金额降序排列。",
        """SELECT charge_person_name,
       COUNT(*) AS contract_count,
       SUM(contract_price_yuan) AS total_yuan
FROM v_project_contract
GROUP BY charge_person_name
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询税率为9%的项目合同列表，返回项目名称、合同编号和合同金额。",
        """SELECT project_name, contract_no, contract_price_yuan
FROM v_project_contract
WHERE tax_rate = 90;"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询结算方式为固定总价的合同总数和总金额。",
        """SELECT COUNT(*) AS contract_count, SUM(contract_price_yuan) AS total_yuan
FROM v_project_contract
WHERE settlement_type = '固定总价';"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询已完工（contract_status = 'DONE'）但完工时间在2024年之前的项目列表。",
        """SELECT project_contract_id, project_name, finish_date, charge_person_name
FROM v_project_contract
WHERE contract_status = 'DONE'
  AND finish_date < '2024-01-01';"""
    ),

    make_sample(
        ["v_project_contract"],
        "统计各分包公司承接的项目数量和合同总金额，只显示合同总金额超过100万的分包公司。",
        """SELECT subcon_company_name,
       COUNT(*) AS project_count,
       SUM(contract_price_yuan) AS total_yuan
FROM v_project_contract
GROUP BY subcon_company_name
HAVING SUM(contract_price_yuan) > 1000000
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询李春来负责的所有未完工项目，按签约时间降序排列。",
        """SELECT project_contract_id, project_name, contract_price_yuan, sign_time
FROM v_project_contract
WHERE charge_person_name = '李春来'
  AND contract_status = 'UN_DONE'
ORDER BY sign_time DESC;"""
    ),

    # ── v_receipt_invoice：收票 ───────────────────────────────────────────────
    make_sample(
        ["v_receipt_invoice"],
        "查询2024年开出的收票总金额。",
        """SELECT SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
WHERE invoice_time >= '2024-01-01' AND invoice_time < '2025-01-01';"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "统计每个项目的收票数量和收票总金额，按收票总金额降序排列前20名。",
        """SELECT project_contract_id, project_name,
       COUNT(*) AS invoice_count,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
GROUP BY project_contract_id, project_name
ORDER BY total_yuan DESC
LIMIT 20;"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "查询已完成客户结算（customer_settlement = 1）的收票记录，返回项目名称、开票金额和结算日期。",
        """SELECT project_name, invoice_price_yuan, settlement_date
FROM v_receipt_invoice
WHERE customer_settlement = 1;"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "查询开票金额超过100万元且税率为9%的收票记录。",
        """SELECT receipt_invoice_id, project_name, invoice_price_yuan, invoice_time
FROM v_receipt_invoice
WHERE invoice_price_yuan > 1000000
  AND tax_rate = 90;"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "统计各业务员的开票笔数和开票总金额。",
        """SELECT charge_person_name,
       COUNT(*) AS invoice_count,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
GROUP BY charge_person_name
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "查询2024年第一季度（1月至3月）每个月的开票金额汇总。",
        """SELECT DATE_FORMAT(invoice_time, '%Y-%m') AS month,
       SUM(invoice_price_yuan) AS total_yuan
FROM v_receipt_invoice
WHERE invoice_time >= '2024-01-01' AND invoice_time < '2024-04-01'
GROUP BY DATE_FORMAT(invoice_time, '%Y-%m')
ORDER BY month;"""
    ),

    # ── v_receive_payment：收款 ───────────────────────────────────────────────
    make_sample(
        ["v_receive_payment"],
        "查询2024年的总收款金额。",
        """SELECT SUM(receive_price_yuan) AS total_yuan
FROM v_receive_payment
WHERE receive_time >= '2024-01-01' AND receive_time < '2025-01-01';"""
    ),

    make_sample(
        ["v_receive_payment"],
        "按收款方式统计收款笔数和总金额。",
        """SELECT receive_form,
       COUNT(*) AS payment_count,
       SUM(receive_price_yuan) AS total_yuan
FROM v_receive_payment
GROUP BY receive_form
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_receive_payment"],
        "查询单笔收款金额超过200万元的收款记录，包含项目名称、收款金额和收款时间。",
        """SELECT project_name, receive_price_yuan, receive_time, charge_person_name
FROM v_receive_payment
WHERE receive_price_yuan > 2000000
ORDER BY receive_price_yuan DESC;"""
    ),

    make_sample(
        ["v_receive_payment"],
        "统计每个分包公司下属项目的收款总额，只返回收款总额超过500万的分包公司。",
        """SELECT subcon_company_name, SUM(receive_price_yuan) AS total_yuan
FROM v_receive_payment
GROUP BY subcon_company_name
HAVING SUM(receive_price_yuan) > 5000000
ORDER BY total_yuan DESC;"""
    ),

    # ── v_cost_contract：成本合同 ─────────────────────────────────────────────
    make_sample(
        ["v_cost_contract"],
        "查询誉童分包公司的所有成本合同，返回合同编号、项目名称和合同金额。",
        """SELECT cost_contract_no, project_name, contract_price_yuan
FROM v_cost_contract
WHERE subcon_company_name = '誉童'
ORDER BY contract_price_yuan DESC;"""
    ),

    make_sample(
        ["v_cost_contract"],
        "统计各分包公司的成本合同数量和合同总金额。",
        """SELECT subcon_company_name,
       COUNT(*) AS contract_count,
       SUM(contract_price_yuan) AS total_yuan
FROM v_cost_contract
GROUP BY subcon_company_name
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_cost_contract"],
        "查询税率为13%的成本合同数量和总金额。",
        """SELECT COUNT(*) AS contract_count, SUM(contract_price_yuan) AS total_yuan
FROM v_cost_contract
WHERE tax_rate = 130;"""
    ),

    make_sample(
        ["v_cost_contract"],
        "查询2023年签署的成本合同，按签约时间排序。",
        """SELECT cost_contract_id, cost_contract_no, project_name,
       contract_price_yuan, subcon_company_name, sign_time
FROM v_cost_contract
WHERE sign_time >= '2023-01-01' AND sign_time < '2024-01-01'
ORDER BY sign_time;"""
    ),

    # ── v_payment：付款 ───────────────────────────────────────────────────────
    make_sample(
        ["v_payment"],
        "查询2024年向誉童公司的付款记录，包含项目名称、付款金额和付款时间。",
        """SELECT project_name, payment_price_yuan, payment_time
FROM v_payment
WHERE subcon_company_name = '誉童'
  AND payment_time >= '2024-01-01' AND payment_time < '2025-01-01'
ORDER BY payment_time;"""
    ),

    make_sample(
        ["v_payment"],
        "统计各分包公司收到的付款总额，按总额降序排列。",
        """SELECT subcon_company_name,
       COUNT(*) AS payment_count,
       SUM(payment_price_yuan) AS total_yuan
FROM v_payment
GROUP BY subcon_company_name
ORDER BY total_yuan DESC;"""
    ),

    make_sample(
        ["v_payment"],
        "查询付款金额超过50万元的付款记录。",
        """SELECT payment_id, project_name, payment_price_yuan,
       payment_time, subcon_company_name
FROM v_payment
WHERE payment_price_yuan > 500000
ORDER BY payment_price_yuan DESC;"""
    ),

    # ── v_payment_invoice：付款发票 ───────────────────────────────────────────
    make_sample(
        ["v_payment_invoice"],
        "查询2024年收到的付款发票总金额。",
        """SELECT SUM(invoice_price_yuan) AS total_yuan
FROM v_payment_invoice
WHERE invoice_time >= '2024-01-01' AND invoice_time < '2025-01-01';"""
    ),

    make_sample(
        ["v_payment_invoice"],
        "查询已完成抵扣（is_deduction = 1）的付款发票列表，返回项目名称、发票金额和开票时间。",
        """SELECT project_name, invoice_price_yuan, invoice_time
FROM v_payment_invoice
WHERE is_deduction = 1
ORDER BY invoice_time DESC;"""
    ),

    # ── v_project_finance_summary：项目财务汇总（预聚合视图）────────────────────
    make_sample(
        ["v_project_finance_summary"],
        "查询收票总金额超过付款总金额的项目，返回项目名称、收票总额、付款总额及差额。",
        """SELECT project_name,
       receipt_invoice_total_yuan,
       payment_invoice_total_yuan,
       (receipt_invoice_total_yuan - payment_invoice_total_yuan) AS diff_yuan
FROM v_project_finance_summary
WHERE receipt_invoice_total_yuan > payment_invoice_total_yuan
ORDER BY diff_yuan DESC;"""
    ),

    make_sample(
        ["v_project_finance_summary"],
        "查询已开票但尚未收款（receive_total_yuan = 0 且 receipt_invoice_count > 0）的项目列表。",
        """SELECT project_contract_id, project_name,
       receipt_invoice_total_yuan, receive_total_yuan,
       charge_person_name
FROM v_project_finance_summary
WHERE receipt_invoice_count > 0
  AND (receive_total_yuan = 0 OR receive_total_yuan IS NULL)
ORDER BY receipt_invoice_total_yuan DESC;"""
    ),

    make_sample(
        ["v_project_finance_summary"],
        "查询未完工项目中，合同金额排名前5的项目及其财务情况。",
        """SELECT project_name, contract_price_yuan,
       receipt_invoice_total_yuan, receive_total_yuan,
       payment_total_yuan
FROM v_project_finance_summary
WHERE contract_status = 'UN_DONE'
ORDER BY contract_price_yuan DESC
LIMIT 5;"""
    ),

    make_sample(
        ["v_project_finance_summary"],
        "按负责人统计其所有项目的合同总金额、收票总额和收款总额。",
        """SELECT charge_person_name,
       COUNT(*) AS project_count,
       SUM(contract_price_yuan) AS total_contract_yuan,
       SUM(receipt_invoice_total_yuan) AS total_receipt_yuan,
       SUM(receive_total_yuan) AS total_receive_yuan
FROM v_project_finance_summary
GROUP BY charge_person_name
ORDER BY total_contract_yuan DESC;"""
    ),

    # ── v_salesman_summary：业务员汇总（预聚合视图）──────────────────────────────
    make_sample(
        ["v_salesman_summary"],
        "查询项目数量最多的前5名业务员。",
        """SELECT salesman_name, project_count, project_total_yuan
FROM v_salesman_summary
ORDER BY project_count DESC
LIMIT 5;"""
    ),

    make_sample(
        ["v_salesman_summary"],
        "查询所有业务员的收款总额，按收款总额降序排列。",
        """SELECT salesman_name, receive_total_yuan, receive_count
FROM v_salesman_summary
ORDER BY receive_total_yuan DESC;"""
    ),

    make_sample(
        ["v_salesman_summary"],
        "查询项目合同总金额超过1000万元的业务员名单。",
        """SELECT salesman_name, project_count, project_total_yuan
FROM v_salesman_summary
WHERE project_total_yuan > 10000000
ORDER BY project_total_yuan DESC;"""
    ),

    # ── v_customer：客户 ──────────────────────────────────────────────────────
    make_sample(
        ["v_customer"],
        "查询所有大客户（big_customer = 1）的名单，包含客户名称、负责人和联系方式。",
        """SELECT customer_name, owner_name, phone_number, email
FROM v_customer
WHERE big_customer = 1;"""
    ),

    make_sample(
        ["v_customer"],
        "统计各地区的客户数量，按客户数量降序排列。",
        """SELECT region, COUNT(*) AS customer_count
FROM v_customer
WHERE region IS NOT NULL
GROUP BY region
ORDER BY customer_count DESC;"""
    ),

    # ── v_subcon_company：分包公司 ────────────────────────────────────────────
    make_sample(
        ["v_subcon_company"],
        "查询所有分包公司的名称和联系电话。",
        """SELECT company_name, tel
FROM v_subcon_company
ORDER BY company_name;"""
    ),

    # ── v_party_company：甲方公司 ─────────────────────────────────────────────
    make_sample(
        ["v_party_company"],
        "查询所有甲方公司列表，返回公司名称、地址和税号。",
        """SELECT company_name, address, tax_number
FROM v_party_company
ORDER BY company_name;"""
    ),

    # ── v_user：用户 ──────────────────────────────────────────────────────────
    make_sample(
        ["v_user"],
        "查询最近30天内有登录记录的用户列表。",
        """SELECT user_name, employee_no, last_login_time
FROM v_user
WHERE last_login_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY last_login_time DESC;"""
    ),

    # ── v_refund / v_refund_invoice：退款退票 ─────────────────────────────────
    make_sample(
        ["v_refund"],
        "查询2023年所有退款记录，包含退款金额、退款方式和退款时间。",
        """SELECT refund_money_yuan, refund_form, refund_time
FROM v_refund
WHERE refund_time >= '2023-01-01' AND refund_time < '2024-01-01'
ORDER BY refund_time;"""
    ),

    make_sample(
        ["v_refund_invoice"],
        "查询已结算（settlement = 1）的退票记录，返回项目名称、退票金额和退票时间。",
        """SELECT project_name, refund_price_yuan, refund_time
FROM v_refund_invoice
WHERE settlement = 1
ORDER BY refund_time DESC;"""
    ),

    make_sample(
        ["v_refund_invoice"],
        "统计各项目的退票总金额，只显示退票总额超过5万元的项目。",
        """SELECT project_name, SUM(refund_price_yuan) AS total_refund
FROM v_refund_invoice
GROUP BY project_contract_id, project_name
HAVING SUM(refund_price_yuan) > 50000
ORDER BY total_refund DESC;"""
    ),

    # ── 多视图 JOIN：跨视图关联查询 ───────────────────────────────────────────
    make_sample(
        ["v_project_contract", "v_customer"],
        "查询大客户（big_customer = 1）关联的项目合同列表，返回客户名称、项目名称和合同金额。",
        """SELECT c.customer_name, p.project_name, p.contract_price_yuan
FROM v_project_contract p
JOIN v_customer c ON p.customer_id = c.customer_id
WHERE c.big_customer = 1
ORDER BY p.contract_price_yuan DESC;"""
    ),

    make_sample(
        ["v_project_contract", "v_party_company"],
        "查询中冶集团相关甲方公司承接的项目合同，返回甲方公司名称、项目名称和合同金额。",
        """SELECT pc.company_name, p.project_name, p.contract_price_yuan
FROM v_project_contract p
JOIN v_party_company pc ON p.party_company_id = pc.party_company_id
WHERE pc.company_name LIKE '%中冶%'
ORDER BY p.contract_price_yuan DESC;"""
    ),

    make_sample(
        ["v_receipt_invoice", "v_receive_payment"],
        "查询已开票但尚未收款的收票记录（即在收票表中存在、但在收款表中没有对应记录的发票）。",
        """SELECT ri.receipt_invoice_id, ri.project_name,
       ri.invoice_price_yuan, ri.invoice_time
FROM v_receipt_invoice ri
LEFT JOIN v_receive_payment rp ON ri.receipt_invoice_id = rp.receipt_invoice_id
WHERE rp.receive_payment_id IS NULL
ORDER BY ri.invoice_time;"""
    ),

    make_sample(
        ["v_cost_contract", "v_payment"],
        "查询成本合同金额与实际付款金额的差值，返回合同编号、项目名称、合同金额、已付金额和未付差额。",
        """SELECT cc.cost_contract_no, cc.project_name,
       cc.contract_price_yuan AS contract_yuan,
       COALESCE(SUM(p.payment_price_yuan), 0) AS paid_yuan,
       cc.contract_price_yuan - COALESCE(SUM(p.payment_price_yuan), 0) AS unpaid_yuan
FROM v_cost_contract cc
LEFT JOIN v_payment p ON cc.cost_contract_id = p.cost_contract_id
GROUP BY cc.cost_contract_id, cc.cost_contract_no,
         cc.project_name, cc.contract_price_yuan
ORDER BY unpaid_yuan DESC;"""
    ),

    make_sample(
        ["v_project_contract", "v_subcon_company"],
        "查询誉童分包公司承接的所有项目合同，包含项目名称、合同金额和签约时间。",
        """SELECT p.project_name, p.contract_price_yuan, p.sign_time
FROM v_project_contract p
JOIN v_subcon_company sc ON p.subcon_company_id = sc.subcon_company_id
WHERE sc.company_name = '誉童'
ORDER BY p.sign_time DESC;"""
    ),

    make_sample(
        ["v_project_finance_summary", "v_user"],
        "查询每位业务员负责的项目中，收款率（收款总额/合同总金额）最高的前5名，收款率以百分比表示。",
        """SELECT pf.charge_person_name,
       SUM(pf.contract_price_yuan) AS total_contract,
       SUM(pf.receive_total_yuan) AS total_receive,
       ROUND(SUM(pf.receive_total_yuan) / NULLIF(SUM(pf.contract_price_yuan), 0) * 100, 2) AS receive_rate_pct
FROM v_project_finance_summary pf
GROUP BY pf.charge_person_name
ORDER BY receive_rate_pct DESC
LIMIT 5;"""
    ),

    # ── 复杂查询：子查询、CTE、窗口函数 ──────────────────────────────────────
    make_sample(
        ["v_project_contract"],
        "查询合同金额高于所有项目平均合同金额的项目列表。",
        """SELECT project_contract_id, project_name, contract_price_yuan, charge_person_name
FROM v_project_contract
WHERE contract_price_yuan > (
    SELECT AVG(contract_price_yuan) FROM v_project_contract
)
ORDER BY contract_price_yuan DESC;"""
    ),

    make_sample(
        ["v_receipt_invoice"],
        "按年份统计收票总金额，同时计算每年相比上一年的增长额。",
        """SELECT
    YEAR(invoice_time) AS year,
    SUM(invoice_price_yuan) AS total_yuan,
    SUM(invoice_price_yuan) - LAG(SUM(invoice_price_yuan))
        OVER (ORDER BY YEAR(invoice_time)) AS yoy_change
FROM v_receipt_invoice
WHERE invoice_time IS NOT NULL
GROUP BY YEAR(invoice_time)
ORDER BY year;"""
    ),

    make_sample(
        ["v_project_finance_summary"],
        "找出每个分包公司中合同金额最高的项目。",
        """SELECT subcon_company_name, project_name, contract_price_yuan
FROM (
    SELECT subcon_company_name, project_name, contract_price_yuan,
           ROW_NUMBER() OVER (
               PARTITION BY subcon_company_name
               ORDER BY contract_price_yuan DESC
           ) AS rn
    FROM v_project_finance_summary
) t
WHERE rn = 1
ORDER BY contract_price_yuan DESC;"""
    ),

    make_sample(
        ["v_receive_payment"],
        "查询近12个月（从当前月份往前推12个月）每个月的收款总额。",
        """SELECT DATE_FORMAT(receive_time, '%Y-%m') AS month,
       SUM(receive_price_yuan) AS total_yuan
FROM v_receive_payment
WHERE receive_time >= DATE_SUB(DATE_FORMAT(NOW(), '%Y-%m-01'), INTERVAL 11 MONTH)
GROUP BY DATE_FORMAT(receive_time, '%Y-%m')
ORDER BY month;"""
    ),

    make_sample(
        ["v_project_contract"],
        "查询同一甲方公司下有超过3个项目合同的甲方公司名称及其项目数量。",
        """SELECT party_company_name, COUNT(*) AS project_count
FROM v_project_contract
WHERE party_company_name IS NOT NULL
GROUP BY party_company_name
HAVING COUNT(*) > 3
ORDER BY project_count DESC;"""
    ),

    make_sample(
        ["v_receipt_invoice", "v_receive_payment"],
        "统计每个项目的开票总额和收款总额，返回两者均大于0的项目及其收款完成率。",
        """SELECT ri.project_contract_id,
       ri.project_name,
       SUM(ri.invoice_price_yuan) AS total_invoiced,
       COALESCE(rp.total_received, 0) AS total_received,
       ROUND(COALESCE(rp.total_received, 0) /
             NULLIF(SUM(ri.invoice_price_yuan), 0) * 100, 2) AS receive_rate_pct
FROM v_receipt_invoice ri
LEFT JOIN (
    SELECT project_contract_id, SUM(receive_price_yuan) AS total_received
    FROM v_receive_payment
    GROUP BY project_contract_id
) rp ON ri.project_contract_id = rp.project_contract_id
GROUP BY ri.project_contract_id, ri.project_name, rp.total_received
HAVING SUM(ri.invoice_price_yuan) > 0
ORDER BY receive_rate_pct DESC;"""
    ),

]


def main():
    os.makedirs("data/seed", exist_ok=True)
    out_path = "data/seed/business_seed.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in SEEDS:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"已生成 {len(SEEDS)} 条业务种子样本 → {out_path}")


if __name__ == "__main__":
    main()
