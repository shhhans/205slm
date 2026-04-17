"""
Unsloth + SFTTrainer 微调脚本
──────────────────────────────
适用模型：Qwen2.5-0.5B-Instruct / Qwen3-0.6B
数据格式：data/final/train.jsonl（alpaca 格式）

安装依赖：
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl datasets

运行：
  python scripts/train_unsloth.py
  python scripts/train_unsloth.py --model Qwen/Qwen3-0.6B --epochs 5
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ── 参数 ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="模型路径（HuggingFace ID 或本地路径）")
parser.add_argument("--data", default="data/final/train.jsonl")
parser.add_argument("--output", default="output/qwen-sql-unsloth")
parser.add_argument("--max-len", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lora-rank", type=int, default=16)
args = parser.parse_args()

# ── 1. 加载模型与 Tokenizer ───────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_len,
    dtype=None,        # 自动检测 bf16 / fp16
    load_in_4bit=True, # QLoRA：4bit 量化，显著降低显存占用
)

# ── 2. 挂载 LoRA ──────────────────────────────────────────────────────────────

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,
    lora_alpha=args.lora_rank * 2,
    lora_dropout=0.05,
    target_modules="all-linear",  # 对所有线性层应用 LoRA
    bias="none",
    use_gradient_checkpointing="unsloth",  # 节省显存
    random_state=42,
)

# ── 3. 构建 Prompt 模板 ───────────────────────────────────────────────────────

ALPACA_TEMPLATE = """\
### 指令：
{instruction}

### 输入：
{input}

### 回答：
{output}"""

EOS = tokenizer.eos_token


def format_sample(example: dict) -> dict:
    text = ALPACA_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    ) + EOS
    return {"text": text}


# ── 4. 加载数据集 ─────────────────────────────────────────────────────────────

data_path = Path(args.data)
raw = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
dataset = Dataset.from_list(raw).map(format_sample, remove_columns=["instruction", "input", "output"])

print(f"\n训练样本数：{len(dataset)}")
print(f"样本示例（前200字）：\n{dataset[0]['text'][:200]}\n")

# ── 5. 训练 ───────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=True,       # 若 GPU 不支持 bf16 改为 fp16=True, bf16=False
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        output_dir=args.output,
        report_to="none",
        seed=42,
    ),
)

trainer.train()

# ── 6. 保存 ──────────────────────────────────────────────────────────────────

model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
print(f"\n模型已保存至：{args.output}")

# 可选：合并 LoRA 权重到基础模型（推理时无需加载 LoRA 适配器）
# model.save_pretrained_merged(args.output + "-merged", tokenizer, save_method="merged_16bit")
