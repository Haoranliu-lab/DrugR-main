
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

import sys
import argparse
from typing import List, Optional, Literal
import torch
import tempfile


tempfile.tempdir = tmp_dir
print(f"Python tempfile 默认目录设置为: {tempfile.gettempdir()}")

def main():
    parser = argparse.ArgumentParser(description="使用 Swift 框架进行 SFT 微调")

    # 模型配置
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="模型本地路径或模型 ID")
    parser.add_argument("--model_type", type=str, default="qwen3",
                        help="模型类型")

    # Checkpoint 配置
    parser.add_argument("--pretrained_adapter_path", type=str, default=None,
                        help="预训练的 LoRA adapter 路径（只加载权重，不恢复训练状态）")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从哪个 checkpoint 继续训练（如果设置，会恢复训练状态，但不适用于 streaming 模式）")
    parser.add_argument("--reset_training_state", action="store_true", default=False,
                        help="重置训练状态（从 step 0 开始），即使从 checkpoint 恢复")
    parser.add_argument("--smiles_only_mode", action="store_true", default=False,
                        help="只训练 SMILES 部分（第一阶段），推理部分不计算损失")

    # 数据配置
    parser.add_argument("--chem_jsonl_path", type=str, default=None,
                        help="化学推理数据路径1")
    parser.add_argument("--chem_jsonl_path2", type=str, default=None,
                        help="化学推理数据路径2")

    # 训练配置
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率（SFT 通常使用较小的学习率）")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Warmup 步数")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="最大训练步数")
    parser.add_argument("--max_length", type=int, default=3072,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=400,
                        help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=10,
                        help="保存的检查点数量限制")

    # 验证集配置
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="验证集文件路径（可选）")
    parser.add_argument("--split_dataset_ratio", type=float, default=0.05,
                        help="从训练集中自动分割验证集的比例（默认 5%）")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        help="评估策略: steps, epoch, no")
    parser.add_argument("--eval_steps", type=int, default=400,
                        help="评估步数间隔")

    # Swift 框架配置
    parser.add_argument("--train_type", type=str, default="full",
                        help="训练类型: lora (LoRA微调), full (全参数训练), qlora 等")

    args = parser.parse_args()

    # 检查 Swift 是否安装
    try:
        import swift
        from swift.llm import sft_main
        from swift.llm.template.base import Template
        print("Swift 框架已安装")
    except ImportError:
        print("错误: Swift 框架未安装")
        print("请运行: pip install ms-swift")
        sys.exit(1)

    # 如果指定了 pretrained_adapter_path，优先使用它（只加载权重，不恢复训练状态）
    if args.pretrained_adapter_path:
        if not os.path.exists(args.pretrained_adapter_path):
            print(f"❌ 错误: 预训练 adapter 不存在: {args.pretrained_adapter_path}")
            sys.exit(1)
        print(f"将加载预训练 LoRA adapter: {args.pretrained_adapter_path}")
        print("   注意: 只加载模型权重，不恢复训练状态（从 step 0 开始）")
        if args.resume_from_checkpoint:
            print("注意: 同时指定了 pretrained_adapter_path 和 resume_from_checkpoint")
            print(" 将使用 pretrained_adapter_path（只加载权重），忽略 resume_from_checkpoint")
            args.resume_from_checkpoint = None

    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"错误: Checkpoint 不存在: {args.resume_from_checkpoint}")
            sys.exit(1)
        print(f"将从 checkpoint 继续训练: {args.resume_from_checkpoint}")
        if args.reset_training_state:
            print("   注意: 使用 --reset_training_state，将从 step 0 开始（只恢复模型权重）")

    # 实现智能截断策略（与 general_dataset.py 一致）
    original_truncate = Template._truncate

    import re
    _SMILES_LINE = re.compile(r"^\s*Optimized[_\s-]*SMILES\s*[:：]\s*(.*)$", re.I | re.M)

    smiles_only_mode = args.smiles_only_mode
    model_name_or_path = args.model_name_or_path

    def smart_truncate(self, input_ids: List[int], labels: Optional[List[int]],
                       loss_mask: Optional[List[float]],
                       truncation_strategy: Literal["left", "right"]):

        if self.max_length is None:
            return input_ids, labels, loss_mask

        placeholder_tokens = torch.tensor(self.placeholder_tokens)
        input_ids_tensor = torch.tensor(input_ids)
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
        n_protected = protected.sum().item()

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

                if smiles_only_mode:
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                        assistant_text = tokenizer.decode(assistant_ids, skip_special_tokens=False)
                        m = _SMILES_LINE.search(assistant_text)
                        if m:
                            reasoning_text = assistant_text[:m.start()].strip()
                            reasoning_ids = tokenizer(reasoning_text, add_special_tokens=False).get("input_ids", [])
                            reasoning_end_pos = len(reasoning_ids)
                            if reasoning_end_pos < len(assistant_labels):
                                assistant_labels = [-100] * reasoning_end_pos + assistant_labels[reasoning_end_pos:]
                    except Exception:
                        pass

                prompt_keep_ratio = 0.35
                domain_min_keep = {"chem": 512, "ultrachat": 256}
                domain_max_keep = {"chem": 1024, "ultrachat": 512}

                min_keep_p = domain_min_keep.get("chem", 512)
                max_keep_p = domain_max_keep.get("chem", 1024)

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
    if smiles_only_mode:
        print("已应用智能截断策略（SMILES-only 模式：只对 SMILES 部分计算损失）")
    else:
        print("已应用智能截断策略（完整输出模式：对推理+SMILES 计算损失）")

    print("准备数据集...")
    dataset_paths = []

    if args.chem_jsonl_path and os.path.exists(args.chem_jsonl_path):
        dataset_paths.append(args.chem_jsonl_path)
        print(f"  化学数据1: {args.chem_jsonl_path}")

    if args.chem_jsonl_path2 and os.path.exists(args.chem_jsonl_path2):
        dataset_paths.append(args.chem_jsonl_path2)
        print(f"  化学数据2: {args.chem_jsonl_path2}")

    if not dataset_paths:
        print("错误: 没有找到数据集文件")
        sys.exit(1)

    print(f"总共找到 {len(dataset_paths)} 个数据集文件")

    print("开始使用 Swift 框架进行 SFT 微调...")

    swift_args = {
        "model_type": args.model_type,
        "output_dir": args.output_dir,
        "train_type": args.train_type,
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
    }

    if args.pretrained_adapter_path:
        swift_args["model"] = args.model_name_or_path
        swift_args["resume_from_checkpoint"] = args.pretrained_adapter_path
        swift_args["streaming"] = False
        swift_args["resume_only_model"] = True

        import json
        import shutil
        trainer_state_path = os.path.join(args.pretrained_adapter_path, "trainer_state.json")
        backup_path = trainer_state_path + ".backup"

        if not os.path.exists(trainer_state_path):
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, trainer_state_path)
                print(f"从备份恢复 trainer_state.json: {backup_path}")
            else:
                trainer_state = {
                    "global_step": 0,
                    "epoch": 0.0,
                    "best_metric": None,
                    "best_model_checkpoint": None,
                    "log_history": [],
                }
                with open(trainer_state_path, "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=2, ensure_ascii=False)
                print("创建新的 trainer_state.json（global_step=0）")

        if os.path.exists(trainer_state_path):
            if not os.path.exists(backup_path):
                shutil.copy2(trainer_state_path, backup_path)
                print(f"已备份 trainer_state.json: {backup_path}")

            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)

            trainer_state["global_step"] = 0
            trainer_state["epoch"] = 0.0
            trainer_state["best_metric"] = None
            trainer_state["best_model_checkpoint"] = None
            trainer_state["log_history"] = []

            with open(trainer_state_path, "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2, ensure_ascii=False)
            print("已重置 trainer_state.json（global_step=0），强制从 step 0 开始训练")

        print(f"使用 pretrained_adapter_path 加载 adapter: {args.pretrained_adapter_path}")
        print("   使用 resume_only_model=True，只恢复模型权重，从 step 0 开始训练")
    elif args.resume_from_checkpoint:
        swift_args["model"] = args.model_name_or_path
        swift_args["resume_from_checkpoint"] = args.resume_from_checkpoint
        swift_args["streaming"] = False

        if args.reset_training_state:
            swift_args["resume_only_model"] = True

            import json
            import shutil
            trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
            backup_path = trainer_state_path + ".backup"

            if os.path.exists(trainer_state_path):
                if not os.path.exists(backup_path):
                    shutil.copy2(trainer_state_path, backup_path)
                    print(f"已备份 trainer_state.json: {backup_path}")

                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)

                trainer_state["global_step"] = 0
                trainer_state["epoch"] = 0.0
                trainer_state["best_metric"] = None
                trainer_state["best_model_checkpoint"] = None
                trainer_state["log_history"] = []

                with open(trainer_state_path, "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=2, ensure_ascii=False)
                print("已重置 trainer_state.json（global_step=0），强制从 step 0 开始训练")
            else:
                trainer_state = {
                    "global_step": 0,
                    "epoch": 0.0,
                    "best_metric": None,
                    "best_model_checkpoint": None,
                    "log_history": [],
                }
                with open(trainer_state_path, "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=2, ensure_ascii=False)
                print("创建新的 trainer_state.json（global_step=0）")

            print("使用 resume_from_checkpoint 并重置训练状态")
            print("   使用 resume_only_model=True，只恢复模型权重，从 step 0 开始")
        else:
            print("警告: 使用 resume_from_checkpoint，会恢复训练状态")
            print("   如果 checkpoint 的步数超过 max_steps，训练会立即结束")
            print("   建议使用 --pretrained_adapter_path 或 --reset_training_state")
    else:
        swift_args["model"] = args.model_name_or_path
        swift_args["streaming"] = True

    print("训练参数:")
    for key, value in swift_args.items():
        print(f"  {key}: {value}")

    try:
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ["swift_sft.py"]

        for key, value in swift_args.items():
            if value is not None:
                sys.argv.extend([f"--{key}", str(value)])

        for dataset_path in dataset_paths:
            sys.argv.extend(["--dataset", dataset_path])

        if args.val_dataset and os.path.exists(args.val_dataset):
            sys.argv.extend(["--val_dataset", args.val_dataset])
            print(f"验证集: {args.val_dataset}")
        else:
            print(f"验证集: 从训练集中自动分割 {args.split_dataset_ratio*100:.1f}%")

        sft_main()

        sys.argv = original_argv

    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
