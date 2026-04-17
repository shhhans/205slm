"""
数据集合并脚本
──────────────
将所有数据源合并为最终训练集 data/final/train.jsonl。

数据源优先级与采样策略：
  1. 业务种子数据（business_seed）    权重最高，全量保留并重复3次
  2. 增强数据（augmented）            全量保留
  3. 自我纠错数据（self_correct）     全量保留并重复2次
  4. Spider 外部数据（可选）          随机采样最多 3000 条
  5. BIRD 外部数据（可选）            随机采样最多 2000 条

运行：
  python scripts/05_assemble_final.py
  python scripts/05_assemble_final.py --spider-quota 3000 --bird-quota 2000
"""

import json
import random
import argparse
from pathlib import Path

random.seed(42)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_or_all(samples: list, quota: int) -> list:
    if quota <= 0 or len(samples) <= quota:
        return samples
    return random.sample(samples, quota)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-quota", type=int, default=3000)
    parser.add_argument("--bird-quota", type=int, default=2000)
    parser.add_argument("--seed-repeat", type=int, default=3,
                        help="业务种子数据重复次数（提高学习权重）")
    parser.add_argument("--correct-repeat", type=int, default=2,
                        help="纠错数据重复次数")
    args = parser.parse_args()

    data_dir = Path("data")
    final_dir = data_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    stats = {}

    # 1. 业务种子数据（最高优先级，重复采样）
    seeds = load_jsonl(data_dir / "seed" / "business_seed.jsonl")
    seed_samples = seeds * args.seed_repeat
    all_samples.extend(seed_samples)
    stats["business_seed"] = f"{len(seeds)} 条 × {args.seed_repeat} = {len(seed_samples)} 条"

    # 2. 增强数据
    augmented = load_jsonl(data_dir / "augmented" / "augmented.jsonl")
    all_samples.extend(augmented)
    stats["augmented"] = f"{len(augmented)} 条"

    # 3. 自我纠错数据（重复采样）
    correct = load_jsonl(data_dir / "augmented" / "self_correct.jsonl")
    correct_samples = correct * args.correct_repeat
    all_samples.extend(correct_samples)
    stats["self_correct"] = f"{len(correct)} 条 × {args.correct_repeat} = {len(correct_samples)} 条"

    # 4. Spider 外部数据（可选）
    spider = load_jsonl(data_dir / "external" / "spider_mysql.jsonl")
    spider_sampled = sample_or_all(spider, args.spider_quota)
    all_samples.extend(spider_sampled)
    stats["spider"] = f"{len(spider_sampled)} 条（原始 {len(spider)} 条，配额 {args.spider_quota}）"

    # 5. BIRD 外部数据（可选）
    bird = load_jsonl(data_dir / "external" / "bird_mysql.jsonl")
    bird_sampled = sample_or_all(bird, args.bird_quota)
    all_samples.extend(bird_sampled)
    stats["bird"] = f"{len(bird_sampled)} 条（原始 {len(bird)} 条，配额 {args.bird_quota}）"

    # 打乱顺序
    random.shuffle(all_samples)

    # 去重（基于 input+output 的哈希）
    seen = set()
    deduped = []
    for s in all_samples:
        key = (s.get("input", ""), s.get("output", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # 写出最终训练集
    out_path = final_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in deduped:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 写出 LLaMA-Factory dataset_info.json
    dataset_info = {
        "sql_finetune": {
            "file_name": "train.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    info_path = final_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n── 数据集合并统计 ──────────────────────────────")
    for src, desc in stats.items():
        print(f"  {src:<20} {desc}")
    print(f"  {'去重前':<20} {len(all_samples)} 条")
    print(f"  {'最终训练集':<20} {len(deduped)} 条")
    print(f"\n输出文件：")
    print(f"  {out_path}")
    print(f"  {info_path}")


if __name__ == "__main__":
    main()
