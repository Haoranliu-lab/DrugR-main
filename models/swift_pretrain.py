
import os


cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
tmp_dir = os.environ.get("TMPDIR", os.environ.get("TMP", os.path.expanduser("/tmp")))
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", f"{cache_dir}/datasets")
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", f"{cache_dir}/transformers")
os.environ["TMPDIR"] = tmp_dir
os.environ["TMP"] = tmp_dir
os.environ["TEMP"] = tmp_dir
os.environ["ARROW_TEMPDIR"] = os.environ.get("ARROW_TEMPDIR", tmp_dir)
os.environ["PYARROW_TEMPDIR"] = os.environ.get("PYARROW_TEMPDIR", tmp_dir)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
print(f"缓存目录设置为: {os.environ['HF_DATASETS_CACHE']}")
print(f"临时文件目录设置为: {os.environ['TMPDIR']}")
print(f"PyArrow 临时目录设置为: {os.environ.get('ARROW_TEMPDIR', 'N/A')}")

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Literal
import torch
import tempfile


tempfile.tempdir = tmp_dir
print(f"Python tempfile 默认目录设置为: {tempfile.gettempdir()}")


try:
    import pyarrow as pa
    print(f"PyArrow 版本: {pa.__version__}")
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="使用 Swift 框架进行全参数预训练")


    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="模型本地路径或模型 ID")
    parser.add_argument("--model_type", type=str, default="qwen3",
                        help="模型类型")


    parser.add_argument("--qa_data_dir", type=str, default=None,
                        help="QA 数据目录")
    parser.add_argument("--ultrachat_dir", type=str, default=None,
                        help="UltraChat 数据目录")
    parser.add_argument("--molnet_ultra_path", type=str, default=None,
                        help="MoleculeNet Ultra 数据路径")
    parser.add_argument("--cpt_data_path", type=str, default=None,
                        help="CPT 预训练数据路径")
    parser.add_argument("--qa_limit", type=int, default=150000,
                        help="QA 数据限制")
    parser.add_argument("--ultrachat_limit", type=int, default=150000,
                        help="UltraChat 数据限制")
    parser.add_argument("--molnet_limit", type=int, default=150000,
                        help="MoleculeNet 数据限制")


    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup 步数")
    parser.add_argument("--max_steps", type=int, default=2600,
                        help="最大训练步数")
    parser.add_argument("--max_length", type=int, default=3072,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=600,
                        help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=10,
                        help="保存的检查点数量限制")


    parser.add_argument("--val_dataset", type=str, default=None,
                        help="验证集文件路径（可选）")
    parser.add_argument("--split_dataset_ratio", type=float, default=0.05,
                        help="从训练集中自动分割验证集的比例（默认 5%），如果设置了 val_dataset 则忽略此参数")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        help="评估策略: steps, epoch, no")
    parser.add_argument("--eval_steps", type=int, default=600,
                        help="评估步数间隔（当 eval_strategy=steps 时）")


    parser.add_argument("--use_swift", action="store_true", default=True,
                        help="使用 Swift 框架")
    parser.add_argument("--swift_type", type=str, default="full",
                        help="Swift 类型: lora, qlora, full 等")

    args = parser.parse_args()

    try:
        import swift
        from swift.llm import sft_main
        from swift.llm.template.base import Template
        print("Swift 框架已安装")
    except ImportError:
        print("❌ 错误: Swift 框架未安装")
        print("请运行: pip install ms-swift")
        sys.exit(1)


    original_truncate = Template._truncate

    def smart_truncate(self, input_ids: List[int], labels: Optional[List[int]],
                       loss_mask: Optional[List[float]],
                       truncation_strategy: Literal["left", "right"]):
        if self.max_length is None:
            return input_ids, labels, loss_mask

        if hasattr(self, "placeholder_tokens") and len(self.placeholder_tokens) > 0:
            placeholder_tokens = torch.tensor(self.placeholder_tokens)
            input_ids_tensor = torch.tensor(input_ids)
            protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
            n_protected = protected.sum().item()
        else:
            n_protected = 0

        if len(input_ids) <= self.max_length:
            if labels is not None:
                labels = labels[:len(input_ids)]
            if loss_mask is not None:
                loss_mask = loss_mask[:len(input_ids)]
            return input_ids, labels, loss_mask

        if labels is not None:
            assistant_start = None
            for i, label in enumerate(labels):
                if label != -100:
                    assistant_start = i
                    break

            if assistant_start is not None:
                prompt_ids = input_ids[:assistant_start]
                assistant_ids = input_ids[assistant_start:]
                assistant_labels = labels[assistant_start:]
                assistant_loss_mask = loss_mask[assistant_start:] if loss_mask is not None else None

                prompt_keep_ratio = 0.35
                domain_min_keep = {"chem": 512, "ultrachat": 256}
                domain_max_keep = {"chem": 1024, "ultrachat": 512}

                min_keep_p = domain_min_keep.get("ultrachat", 256)
                max_keep_p = domain_max_keep.get("ultrachat", 512)

                target_keep_p = int(self.max_length * prompt_keep_ratio)
                keep_p = max(min_keep_p, min(target_keep_p, max_keep_p))

                remain_for_a = self.max_length - keep_p
                if remain_for_a <= 0:
                    keep_p = min_keep_p
                    remain_for_a = max(0, self.max_length - keep_p)

                if len(assistant_ids) > remain_for_a:
                    assistant_ids = assistant_ids[-remain_for_a:]
                    assistant_labels = assistant_labels[-remain_for_a:] if assistant_labels is not None else None
                    assistant_loss_mask = assistant_loss_mask[-remain_for_a:] if assistant_loss_mask is not None else None

                keep_p = max(0, self.max_length - len(assistant_ids))
                if keep_p < min_keep_p and (len(assistant_ids) + min_keep_p) <= self.max_length:
                    keep_p = min_keep_p
                if len(prompt_ids) > keep_p:
                    prompt_ids = prompt_ids[-keep_p:]

                truncated_input_ids = prompt_ids + assistant_ids
                truncated_labels = [-100] * len(prompt_ids) + (assistant_labels if assistant_labels is not None else [])
                truncated_loss_mask = None
                if loss_mask is not None:
                    truncated_loss_mask = [0.0] * len(prompt_ids) + (assistant_loss_mask if assistant_loss_mask is not None else [])

                return truncated_input_ids, truncated_labels, truncated_loss_mask

        if truncation_strategy == "left":
            truncated = input_ids[-(self.max_length - n_protected):]
        else:
            truncated = input_ids[:self.max_length - n_protected]

        if labels is not None:
            labels = labels[:len(truncated)]
        if loss_mask is not None:
            loss_mask = loss_mask[:len(truncated)]

        return truncated, labels, loss_mask

    Template._truncate = smart_truncate
    print("已应用智能截断策略（与 general_dataset.py 一致）")
    print("策略: prompt 保留比例 35%, 智能分配 prompt 和 assistant 长度")

    print("准备数据集...")
    dataset_paths = []

    def collect_jsonl_files(path):
        """收集路径下的所有 JSONL 文件"""
        jsonl_files = []
        if path and os.path.isfile(path) and path.endswith(".jsonl"):
            jsonl_files.append(path)
        elif path and os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".jsonl"):
                        jsonl_files.append(os.path.join(root, file))
        return jsonl_files

    if args.qa_data_dir and os.path.exists(args.qa_data_dir):
        qa_files = collect_jsonl_files(args.qa_data_dir)
        dataset_paths.extend(qa_files)
        print(f"  QA 数据: 找到 {len(qa_files)} 个文件")

    if args.ultrachat_dir and os.path.exists(args.ultrachat_dir):
        ultrachat_files = collect_jsonl_files(args.ultrachat_dir)
        dataset_paths.extend(ultrachat_files)
        print(f"  UltraChat 数据: 找到 {len(ultrachat_files)} 个文件")

    if args.molnet_ultra_path and os.path.exists(args.molnet_ultra_path):
        dataset_paths.append(args.molnet_ultra_path)
        print(f"  MoleculeNet 数据: {args.molnet_ultra_path}")

    if args.cpt_data_path and os.path.exists(args.cpt_data_path):
        dataset_paths.append(args.cpt_data_path)
        print(f"  CPT 数据: {args.cpt_data_path}")

    print(f"总共找到 {len(dataset_paths)} 个数据集文件")

    print("开始使用 Swift 框架训练...")

    swift_args = {
        "model_type": args.model_type,
        "model": args.model_name_or_path,
        "train_type": args.swift_type,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "truncation_strategy": "left",
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "split_dataset_ratio": args.split_dataset_ratio,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "load_from_cache_file": False,
        "streaming": True,
    }

    if args.val_dataset and os.path.exists(args.val_dataset):
        print(f"使用指定的验证集: {args.val_dataset}")

    print("训练参数:")
    for key, value in swift_args.items():
        print(f"  {key}: {value}")

    try:
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ["swift_pretrain.py"]

        for key, value in swift_args.items():
            if value is not None:
                sys.argv.extend([f"--{key}", str(value)])

        if dataset_paths:
            for dataset_path in dataset_paths:
                sys.argv.extend(["--dataset", dataset_path])

        if args.val_dataset and os.path.exists(args.val_dataset):
            sys.argv.extend(["--val_dataset", args.val_dataset])
            print(f"验证集: {args.val_dataset}")
        else:
            print(f"验证集: 从训练集中自动分割 {args.split_dataset_ratio*100:.1f}%")

        sft_main()

        sys.argv = original_argv

    except TypeError as e:
        print(f"参数格式可能需要调整: {e}")
        print("尝试使用命令行参数方式...")

        import subprocess
        cmd = ["swift", "sft"]
        for k, v in swift_args.items():
            if v is not None:
                cmd.extend([f"--{k}", str(v)])
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        print("提示:")
        print("1. 请检查 Swift 框架版本: pip install ms-swift -U")
        print("2. 查看 Swift 文档: https://github.com/modelscope/swift")
        print("3. 查看示例代码: https://github.com/modelscope/swift/tree/main/examples")
        sys.exit(1)

if __name__ == "__main__":
    main()
