# Phase 1 运行指南

数据准备 → LoRA 微调 → 评估

---

## 环境说明

| 环境 | 用途 |
|------|------|
| 本地 Windows（Miniforge） | 数据生成脚本（01–05） |
| AutoDL（NVIDIA L20, CUDA 12.4） | 训练（LLaMA-Factory）、评估（06） |

本地 Python 路径：`D:/Environments/Miniforge/python.exe`

---

## 目录结构

```
205_SLM/
├── configs/
│   └── qwen_lora_sft.yaml       # LLaMA-Factory 训练配置
├── data/
│   ├── external/
│   │   └── spider/              # Spider 数据集（dev.json, tables.json, database/）
│   ├── seed/                    # 业务种子数据（02 输出）
│   ├── augmented/               # 增强数据（03 输出）
│   └── final/
│       ├── train.jsonl          # 最终训练集（05 输出）
│       └── dataset_info.json    # LLaMA-Factory 数据集注册文件
├── output/
│   ├── qwen-sql-lora/           # LoRA adapter 权重
│   ├── eval_finetuned.jsonl     # 微调模型评估结果
│   └── eval_baseline.jsonl      # 基础模型评估结果
├── scripts/
│   ├── 01_convert_spider_bird.py
│   ├── 02_gen_business_seed.py
│   ├── 03_augment_dataset.py
│   ├── 04_gen_self_correct.py
│   ├── 05_assemble_final.py
│   └── 06_evaluate.py
└── instructions/
    └── PHASE1.md                # 本文件
```

---

## Step 1 — 外部数据转换（本地）

将 Spider / BIRD 数据集从 SQLite 语法转换为 MySQL，输出 alpaca JSONL。

```bash
D:/Environments/Miniforge/python.exe scripts/01_convert_spider_bird.py
```

**输出：** `data/spider_mysql.jsonl`（约 7000 条）

---

## Step 2 — 业务种子数据生成（本地）

手工标注的 55 条业务查询，覆盖全部 14 个视图。

```bash
D:/Environments/Miniforge/python.exe scripts/02_gen_business_seed.py
```

**输出：** `data/seed/business_seed.jsonl`

---

## Step 3 — 数据增强（本地）

四类增强策略：同义词替换、Schema 注释脱敏、部分注释脱敏、干扰视图注入。

```bash
D:/Environments/Miniforge/python.exe scripts/03_augment_dataset.py
```

**输出：** `data/augmented/augmented.jsonl`（约 157 条）

---

## Step 4 — 自纠正样本生成（本地）

生成"错误 SQL + 报错信息 → 正确 SQL"格式的训练样本，覆盖常见 MySQL 错误类型。

```bash
D:/Environments/Miniforge/python.exe scripts/04_gen_self_correct.py
```

**输出：** `data/seed/self_correct.jsonl`（16 条）

---

## Step 5 — 组装最终训练集（本地）

合并所有来源，去重，按权重采样，生成 LLaMA-Factory 所需的数据文件。

```bash
D:/Environments/Miniforge/python.exe scripts/05_assemble_final.py
```

**输出：**
- `data/final/train.jsonl`
- `data/final/dataset_info.json`

采样权重：业务种子 ×3、自纠正 ×2、外部数据按配额采样。

---

## Step 6 — 同步到 AutoDL

```bash
# 本地
git add -A && git commit -m "update dataset" && git push

# AutoDL
cd ~/autodl-tmp/205slm
git pull
```

Spider SQLite 数据库（Layer 3 评估用）从本地 scp 传输，只需传一次：

```powershell
# 本地 PowerShell
scp -P <port> -r "D:\Download\spider_data\spider_data\database" root@<host>:~/autodl-tmp/205slm/data/external/spider/
```

---

## Step 7 — LoRA 微调（AutoDL）

使用 LLaMA-Factory 进行 SFT 训练。

```bash
cd ~/autodl-tmp/205slm
llamafactory-cli train configs/qwen_lora_sft.yaml
```

**关键配置**（`configs/qwen_lora_sft.yaml`）：

| 参数 | 值 |
|------|----|
| 模型 | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| 方法 | LoRA（rank=16, alpha=32） |
| 数据集 | `data/final/train.jsonl`（最多 10000 条） |
| Epochs | 5 |
| Batch size | 4 × grad_accum 4 = 等效 16 |
| Learning rate | 2e-4，cosine scheduler |
| 精度 | bf16 |

**输出：** `output/qwen-sql-lora/`

---

## Step 8 — 评估（AutoDL）

三层验证：语法（sqlglot）→ Schema（字段名对照）→ 执行准确率（SQLite）

### 8a. 评估微调模型

```bash
python scripts/06_evaluate.py \
    --adapter output/qwen-sql-lora \
    --spider-dir data/external/spider \
    --output output/eval_finetuned.jsonl
```

### 8b. 评估基础模型（对照组）

```bash
python scripts/06_evaluate.py \
    --base-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --spider-dir data/external/spider \
    --output output/eval_baseline.jsonl
```

### 8c. 对已有结果重新验证（跳过推理）

```bash
# 含 Layer 3
python scripts/06_evaluate.py --pred-file output/eval_finetuned.jsonl

# 跳过 Layer 3（无 SQLite 数据库时）
python scripts/06_evaluate.py --pred-file output/eval_finetuned.jsonl --skip-exec
```

### 评估报告字段说明

| 指标 | 含义 |
|------|------|
| Syntax Valid | SQL 语法合法率（sqlglot 解析通过） |
| Schema Valid | 表名 + 字段名均存在于 Spider schema 中 |
| Exec Accuracy | 执行结果集与金标准一致（顺序无关） |

---

## 依赖安装

### 本地（数据脚本）

```bash
D:/Environments/Miniforge/python.exe -m pip install sqlglot datasets
```

### AutoDL（训练 + 评估）

```bash
# 1. 安装 PyTorch（CUDA 12.4）
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 2. 安装 LLaMA-Factory 及评估依赖
pip install "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git"
pip install transformers==4.55.0 peft==0.18.1 trl==0.18.0
pip install bitsandbytes==0.44.1
pip install sqlglot tqdm
```

---

## 已知问题

| 问题 | 解决方法 |
|------|----------|
| `bf16` 报错 GPU not found | AutoDL 控制台确认 GPU 实例已开机 |
| CUDA driver version too old | `pip install torch==2.5.1 --index-url .../cu124` |
| `libcudart.so.13` not found | `pip install bitsandbytes==0.44.1` |
| `PreTrainedModel` import error | `pip install transformers==4.55.0 peft==0.18.1 trl==0.18.0` |
| `warmup_ratio` deprecated | 配置文件改用 `warmup_steps: 50` |
| git pull 报 diverged | `GIT_EDITOR=true git pull --no-rebase` |
