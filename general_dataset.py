# -*- coding: utf-8 -*-
import os
import json
import re
from typing import List, Dict, Union, Iterable, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
MIN_PROMPT_KEEP = 256  # 至少保留的 prompt token 数，避免“无条件学习”


# ----------------------------
# 规范化输出（分域）
# ----------------------------
_SMILES_TAG_RE = re.compile(r'^\s*Optimized[_\s]*SMILES\s*:\s*(.*)$', flags=re.I)
_REASON_TAG_RE = re.compile(r'^\s*Reasoning\s*:\s*(.*)$', flags=re.I)



_REASON_LINE = re.compile(r'^\s*Reasoning\s*[:：]\s*(.*)$', re.I|re.M)
_SMILES_LINE = re.compile(r'^\s*Optimized[_\s-]*SMILES\s*[:：]\s*(.*)$', re.I|re.M)

def _extract_reasoning(text: str) -> str:
    if not isinstance(text, str): return ""
    m = _REASON_LINE.search(text)
    return (m.group(1).strip() if m else "").strip()

def _extract_smiles(text: str) -> str:
    if not isinstance(text, str): return ""
    m = _SMILES_LINE.search(text)
    return (m.group(1).strip() if m else "").strip()

import numpy as np



def _build_prompt_text(ex, tokenizer):
    system = ex.get("system", "")
    user   = ex.get("user", "")
    # 对于 Mistral 等模型，chat template 可能对格式要求严格，如果失败则回退到自定义格式
    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        try:
            return tokenizer.apply_chat_template(
                [{"role":"system","content":system},
                 {"role":"user","content":user}],
                add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            # 如果 chat template 失败（如格式不符合要求），回退到自定义格式
            pass
    return f"System: {system}\nUser: {user}\nAssistant: "





  
def _token_len_of(ex, tokenizer, smiles_only: bool = False):
    prompt = _build_prompt_text(ex, tokenizer)
    p_ids  = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_text = normalize_output_by_domain(
        ex.get("assistant_raw",""),
        ex.get("source","ultrachat"),
        smiles_only=False,  # 固定为 False，确保基于全文本
    )
    a_ids  = tokenizer(a_text, add_special_tokens=False)["input_ids"]
    return len(p_ids) + len(a_ids)
 

def compute_length_cap(dataset, tokenizer, percentile: int = 99, hard_max: int | None = None, *, smiles_only: bool = False) -> int:
    lengths = []
    for ex in dataset:
        lengths.append(_token_len_of(ex, tokenizer, smiles_only=False))  # 固定为 False
    cap = int(np.percentile(lengths, percentile))
    if hard_max is not None:
        cap = min(cap, int(hard_max))
    return max(16, cap)




def normalize_output_by_domain(raw_out: str, source: str, *, smiles_only: bool = False) -> str:
    """
    chem:
      - smiles_only=True  : 只监督 SMILES，返回 `Optimized_SMILES: ...\n<END>`
      - smiles_only=False : 两行模板（Reasoning + Optimized_SMILES）并补 <END>
    ultrachat: 原样返回
    """
    raw_out = (raw_out or "").strip()
    if source == "chem":
        if smiles_only:
            smi = _extract_smiles(raw_out)
            return f"Optimized_SMILES: {smi}\n<END>"
        else:
            lines = [ln.strip() for ln in raw_out.splitlines() if ln.strip()]
            reason_line, smiles_line = None, None
            for ln in lines:
                if _REASON_TAG_RE.match(ln):
                    reason_line = ln
                elif _SMILES_TAG_RE.match(ln):
                    smiles_line = ln
            if reason_line is None:
                reason_line = "Reasoning:"
            if smiles_line is None:
                smiles_line = "Optimized_SMILES:"
            out = f"{reason_line}\n{smiles_line}"
            if "<END>" not in out:
                out += "\n<END>"
            return out
    else:
        return raw_out



# ----------------------------
# 加载 reasoning（化学）数据（jsonl）
# ----------------------------
def load_chem_reasoning_jsonl(path: str, chem_stage: str = "reason_plus_smiles") -> List[Dict]:
    """
    chem_stage:
      - "smiles_only"       : 保持原格式，但在collate时只监督SMILES部分
      - "reason_plus_smiles": 保持两行 Reasoning+Optimized_SMILES（原格式）
    """
    data = []
    guid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            instr = (obj.get("instruction") or "").strip()
            inp   = (obj.get("input") or "").strip()
            outp  = (obj.get("output") or "").strip()

            # 无论哪种模式，都保持原始格式
            user_txt = inp
            asst_txt = outp

            data.append({
                "id": f"chem-{guid}",
                "source": "chem",
                "system": instr,
                "user": user_txt,
                "assistant_raw": asst_txt,
                "chem_stage": chem_stage,  # 标记训练阶段
            })
            guid += 1
    return data

# ----------------------------
# 加载 UltraChat（目录下若干 jsonl）
# 只取前 limit 条
# ----------------------------
def _iter_ultrachat_jsonl_files(dir_path: str) -> Iterable[str]:
    for fn in sorted(os.listdir(dir_path)):
        if fn.endswith(".jsonl"):
            yield os.path.join(dir_path, fn)

def load_ultrachat_first_n(dir_path: str, limit: int = 20000) -> List[Dict]:
    """
    读取 UltraChat，优先使用 messages 字段：
      messages: [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
    若无 messages，则退化使用 prompt（并将 assistant 置空）。
    仅取前 limit 条（跨文件累加）。
    """
    out, n = [], 0
    guid = 0
    for fp in _iter_ultrachat_jsonl_files(dir_path):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if n >= limit:
                    return out
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sys_txt = ""  # 可选：放固定 system
                user_txt, asst_txt = "", ""

                # 优先从 messages 提取
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    # 找第一轮 user 和紧随的 assistant
                    first_user = next((m for m in msgs if (m.get("role") == "user" and m.get("content"))), None)
                    first_asst = next((m for m in msgs if (m.get("role") == "assistant" and m.get("content"))), None)
                    if first_user:
                        user_txt = str(first_user.get("content") or "")
                    if first_asst:
                        asst_txt = str(first_asst.get("content") or "")
                else:
                    # 退化到 prompt
                    user_txt = str(obj.get("prompt") or "")
                    # UltraChat 这路通常没有标准 "output" 字段，这里不强求
                    asst_txt = str(obj.get("response") or obj.get("output") or "")

                # 过滤完全空的样本
                if not user_txt and not asst_txt:
                    continue

                out.append({
                    "id": f"ultra-{guid}",
                    "source": "ultrachat",
                    "system": sys_txt,
                    "user": user_txt,
                    "assistant_raw": asst_txt
                })
                guid += 1
                n += 1
    return out


# ----------------------------
# 加载 instruction 数据（用于 SFT）
# ----------------------------
def load_instruction_data(data_dirs: List[str]) -> List[Dict]:
    """
    加载 instruction tuning 数据
    支持 .json 和 .jsonl 格式
    """
    data = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"[WARN] Data directory not found: {data_dir}")
            continue
            
        print(f"[INFO] Loading instruction data from {data_dir}...")
        
        # 遍历目录中的所有文件
        for filename in os.listdir(data_dir):
            if not (filename.endswith('.json') or filename.endswith('.jsonl')):
                continue
                
            file_path = os.path.join(data_dir, filename)
            try:
                if filename.endswith('.jsonl'):
                    # JSONL 格式
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if "instruction" in obj or "data" in obj:
                                    # 支持不同的数据格式
                                    if "data" in obj and isinstance(obj["data"], list):
                                        # 格式：{"data": ["System: ...", "User: ...", "Assistant: ..."]}
                                        if len(obj["data"]) >= 3:
                                            data.append({
                                                "id": f"inst-{filename}-{i}",
                                                "source": "instruction",
                                                "system": obj["data"][0].replace("System: ", "").strip(),
                                                "user": obj["data"][1].replace("User: ", "").strip(),
                                                "assistant_raw": obj["data"][2]
                                            })
                                    else:
                                        # 标准格式：{"instruction": "...", "input": "...", "output": "..."}
                                        data.append({
                                            "id": f"inst-{filename}-{i}",
                                            "source": "instruction",
                                            "system": obj.get("instruction", "").strip(),
                                            "user": obj.get("input", "").strip(),
                                            "assistant_raw": obj.get("output", "")
                                        })
                            except json.JSONDecodeError:
                                continue
                else:
                    # JSON 格式
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            for i, obj in enumerate(json_data):
                                if isinstance(obj, dict) and ("instruction" in obj or "data" in obj):
                                    if "data" in obj and isinstance(obj["data"], list):
                                        if len(obj["data"]) >= 3:
                                            data.append({
                                                "id": f"inst-{filename}-{i}",
                                                "source": "instruction",
                                                "system": obj["data"][0].replace("System: ", "").strip(),
                                                "user": obj["data"][1].replace("User: ", "").strip(),
                                                "assistant_raw": obj["data"][2]
                                            })
                                    else:
                                        data.append({
                                            "id": f"inst-{filename}-{i}",
                                            "source": "instruction",
                                            "system": obj.get("instruction", "").strip(),
                                            "user": obj.get("input", "").strip(),
                                            "assistant_raw": obj.get("output", "")
                                        })
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")
                continue
                
        print(f"[INFO] Loaded {len([d for d in data if data_dir in d['id']])} samples from {data_dir}")
    
    print(f"[INFO] Total instruction samples loaded: {len(data)}")
    return data

# ----------------------------
# 加载化学 QA 数据（用于 pre-training）
# ----------------------------
def load_chem_qa_data(qa_dir: str, max_samples_per_file: int = None) -> List[Dict]:
    """
    加载化学 QA 数据用于 pre-training
    支持 .json 和 .jsonl 格式
    """
    data = []
    qa_files = [
        "pubchem_gptQA.jsonl",
        "open_question.json", 
        "multi_choice_question.json"
    ]
    
    for filename in qa_files:
        file_path = os.path.join(qa_dir, filename)
        if not os.path.exists(file_path):
            print(f"[WARN] QA file not found: {file_path}")
            continue
            
        print(f"[INFO] Loading QA data from {filename}...")
        
        try:
            if filename.endswith('.jsonl'):
                # JSONL 格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if max_samples_per_file and count >= max_samples_per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "instruction" in obj and "output" in obj:
                                data.append({
                                    "id": f"qa-{filename}-{count}",
                                    "source": "chem_qa",
                                    "system": "You are a chemistry expert. Answer the following question accurately and concisely.",
                                    "user": obj["instruction"] + (" " + obj.get("input", "")).strip(),
                                    "assistant_raw": obj["output"]
                                })
                                count += 1
                        except json.JSONDecodeError:
                            continue
            else:
                # JSON 格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        count = 0
                        for obj in json_data:
                            if max_samples_per_file and count >= max_samples_per_file:
                                break
                            if isinstance(obj, dict) and "instruction" in obj:
                                data.append({
                                    "id": f"qa-{filename}-{count}",
                                    "source": "chem_qa", 
                                    "system": "You are a chemistry expert. Answer the following question accurately and concisely.",
                                    "user": obj["instruction"] + (" " + obj.get("input", "")).strip(),
                                    "assistant_raw": obj.get("output", "")
                                })
                                count += 1
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue
            
        print(f"[INFO] Loaded {len([d for d in data if filename in d['id']])} samples from {filename}")
    
    print(f"[INFO] Total QA samples loaded: {len(data)}")
    return data

def load_molnet_ultra_data(molnet_ultra_path: str, limit: int = None) -> List[Dict]:
    """
    加载 MoleculeNet UltraChat 格式数据并转换为 sft_collate 期望的格式
    """
    if not os.path.exists(molnet_ultra_path):
        print(f"[WARN] MoleculeNet UltraChat file not found: {molnet_ultra_path}")
        return []
    
    data = []
    with open(molnet_ultra_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(data) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 转换格式：从 messages 格式转换为 sft_collate 期望的格式
                messages = obj.get("messages", [])
                if len(messages) >= 2:
                    user_msg = messages[0] if messages[0]["role"] == "user" else messages[1]
                    assistant_msg = messages[1] if messages[1]["role"] == "assistant" else messages[0]
                    
                    converted_obj = {
                        "source": "chem",  # 改为 "chem"，这样可以使用化学数据的处理逻辑
                        "system": "",
                        "user": user_msg["content"],
                        "assistant_raw": assistant_msg["content"],
                        "chem_stage": "reason_plus_smiles",  # MoleculeNet 数据全监督
                        "dataset": obj.get("dataset", "unknown")
                    }
                    data.append(converted_obj)
            except Exception as e:
                print(f"[WARN] Failed to parse line: {e}")
                continue
    
    print(f"[MOLNET] Loaded {len(data)} samples from {molnet_ultra_path}")
    return data

def load_cpt_data(cpt_data_path: str, limit: int = None) -> List[Dict]:
    """
    加载 CPT 纯文本预训练数据（jsonl格式，每行包含 text 和 channel 字段）
    将纯文本转换为预训练格式（system 为空，user 为空，assistant_raw 为文本内容）
    """
    if not os.path.exists(cpt_data_path):
        print(f"[WARN] CPT data file not found: {cpt_data_path}")
        return []
    
    data = []
    guid = 0
    with open(cpt_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(data) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                channel = obj.get("channel", "unknown")
                
                if not text:
                    continue
                
                # 将纯文本转换为预训练格式
                # system 和 user 为空，assistant_raw 为文本内容
                converted_obj = {
                    "id": f"cpt-{guid}",
                    "source": "cpt",  # 新数据源标识
                    "system": "",
                    "user": "",  # 纯文本预训练不需要用户输入
                    "assistant_raw": text,  # 文本内容作为 assistant 回复
                    "chem_stage": "reason_plus_smiles",  # 全监督训练
                    "channel": channel  # 保留 channel 信息用于追踪
                }
                data.append(converted_obj)
                guid += 1
            except Exception as e:
                print(f"[WARN] Failed to parse CPT line: {e}")
                continue
    
    print(f"[CPT] Loaded {len(data)} samples from {cpt_data_path}")
    return data

def load_pretrain_mixed_data(
    qa_dir: str,
    ultrachat_dir: str,
    molnet_ultra_path: str = None,
    cpt_data_path: str = None,
    qa_limit: int = 50000,
    ultrachat_limit: int = 20000,
    molnet_limit: int = 50000,
    cpt_limit: int = None,
    seed: int = 42
) -> List[Dict]:
    """
    加载 pre-training 混合数据：化学 QA + MoleculeNet + 通用对话 + CPT 数据
    """
    all_data = []
    
    # 加载化学 QA 数据
    if qa_dir and os.path.exists(qa_dir):
        qa_data = load_chem_qa_data(qa_dir, max_samples_per_file=qa_limit//3)
        all_data.extend(qa_data)
        print(f"[PRETRAIN] QA data: {len(qa_data)} samples")
    
    # 加载 MoleculeNet 数据
    if molnet_ultra_path and os.path.exists(molnet_ultra_path):
        molnet_data = load_molnet_ultra_data(molnet_ultra_path, limit=molnet_limit)
        all_data.extend(molnet_data)
        print(f"[PRETRAIN] MoleculeNet data: {len(molnet_data)} samples")
    
    # 加载通用对话数据
    if ultrachat_dir and os.path.exists(ultrachat_dir):
        ultra_data = load_ultrachat_first_n(ultrachat_dir, limit=ultrachat_limit)
        all_data.extend(ultra_data)
        print(f"[PRETRAIN] UltraChat data: {len(ultra_data)} samples")
    
    # 加载 CPT 纯文本预训练数据
    if cpt_data_path and os.path.exists(cpt_data_path):
        cpt_data = load_cpt_data(cpt_data_path, limit=cpt_limit)
        all_data.extend(cpt_data)
        print(f"[PRETRAIN] CPT data: {len(cpt_data)} samples")
    
    # 打乱顺序
    random.seed(seed)
    random.shuffle(all_data)
    
    print(f"[PRETRAIN] Total mixed data: {len(all_data)} samples")
    return all_data

# ----------------------------
# 混合数据（chem 按倍数重复，ultra 仅取前 N）
# ----------------------------
def load_mixed_data(
    chem_jsonl_path: str,
    ultrachat_dir: str,
    chem_repeat: int = 3,
    ultrachat_limit: int = 20000,
    seed: int = 42,
    chem_stage: str = "reason_plus_smiles",   # ★ 新增
) -> List[Dict]:
    chem  = load_chem_reasoning_jsonl(chem_jsonl_path, chem_stage=chem_stage)
    ultra = load_ultrachat_first_n(ultrachat_dir, limit=ultrachat_limit)
    chem_rep = []
    for _ in range(chem_repeat):
        chem_rep.extend(chem)
    data = chem_rep + ultra
    random.seed(seed); random.shuffle(data)
    return data



# ----------------------------
# collate：只忽略 prompt，assistant 全监督；保留最少 prompt
# ----------------------------
def sft_collate(batch, tokenizer, max_length=None, add_eos=True, smiles_only: bool = False):
    if not batch:
        raise RuntimeError("[sft_collate] got an EMPTY batch from DataLoader. Check sampler.")

    ids_list, labs_list = [], []
    dropped = 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for ex in batch:
        source = ex.get("source", "ultrachat")
        system = ex.get("system", "")
        user = ex.get("user", "")
        assistant = normalize_output_by_domain(ex.get("assistant_raw", ""), source=source)
        chem_stage = ex.get("chem_stage", "reason_plus_smiles")

        # 构造 prompt
        # 对于 Mistral 等模型，chat template 可能对格式要求严格，如果失败则回退到自定义格式
        prompt = None
        if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role":"system","content":system},{"role":"user","content":user}],
                    add_generation_prompt=True, tokenize=False
                )
            except Exception as e:
                # 如果 chat template 失败（如格式不符合要求），回退到自定义格式
                prompt = None
        
        if prompt is None:
            prompt = f"System: {system}\nUser: {user}\nAssistant: "

        p_ids = tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
        a_text = assistant + (tokenizer.eos_token if (add_eos and tokenizer.eos_token) else "")
        a_ids = tokenizer(a_text, add_special_tokens=False).get("input_ids", [])

        # 长度裁剪（如果开了 cap）
        if max_length is not None:
            prompt_keep_ratio = 0.35
            domain_min_keep = {"chem": 512, "ultrachat": 256}
            domain_max_keep = {"chem": 1024, "ultrachat": 512}
            src = source if source in domain_min_keep else "ultrachat"

            target_keep_p = int(max_length * prompt_keep_ratio)
            min_keep_p = domain_min_keep[src]
            max_keep_p = domain_max_keep[src]
            keep_p = max(min_keep_p, min(target_keep_p, max_keep_p))

            remain_for_a = max_length - keep_p
            if remain_for_a <= 0:
                keep_p = min_keep_p
                remain_for_a = max(0, max_length - keep_p)

            if len(a_ids) > remain_for_a:
                a_ids = a_ids[-remain_for_a:]

            keep_p = max(0, max_length - len(a_ids))
            if keep_p < min_keep_p and (len(a_ids) + min_keep_p) <= max_length:
                keep_p = min_keep_p
            if len(p_ids) > keep_p:
                p_ids = p_ids[-keep_p:]

        ids = p_ids + a_ids
        if len(ids) == 0:
            dropped += 1
            continue

        # 关键修改：根据chem_stage和smiles_only决定标签策略

        if source == "chem" and smiles_only:
      # 只对SMILES部分计算损失，reasoning部分设为-100
            labs = [IGNORE_INDEX] * len(p_ids)  # prompt部分忽略
      
      # 用正则鲁棒定位 SMILES 起始
            m = _SMILES_LINE.search(assistant)
            if m:
                reasoning_prefix = assistant[:m.start()].strip()
            else:
                reasoning_prefix = assistant  # 若没找到，保守忽略全部
                print(f"[WARN] SMILES tag not found in chem sample id={ex.get('id')}")
      
            reasoning_ids = tokenizer(reasoning_prefix, add_special_tokens=False).get("input_ids", [])
            reasoning_end_pos = len(reasoning_ids)
      
            labs.extend([IGNORE_INDEX] * reasoning_end_pos)
            smiles_ids = a_ids[reasoning_end_pos:]
            labs.extend(smiles_ids)
      
        else:
      # 原有逻辑：prompt忽略，assistant全监督
            labs = [IGNORE_INDEX] * len(p_ids) + a_ids

        

        ids_list.append(torch.tensor(ids, dtype=torch.long))
        labs_list.append(torch.tensor(labs, dtype=torch.long))

    if not ids_list:
        raise RuntimeError("[sft_collate] all samples in the batch became empty after processing "
                           f"(dropped={dropped}). Check tokenization/normalization or max_length={max_length} too small.")

    input_ids = pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labs_list, batch_first=True, padding_value=IGNORE_INDEX)
    attention = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def sft_collate_with_packing(batch, tokenizer, max_length=None, add_eos=True, smiles_only: bool = False):
    """
    支持序列打包的 collate 函数
    将多个短序列打包到 max_length，提高 GPU 利用率
    """
    if not batch:
        raise RuntimeError("[sft_collate_with_packing] got an EMPTY batch from DataLoader. Check sampler.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if max_length is None:
        raise ValueError("[sft_collate_with_packing] max_length must be specified for packing")

    # 先处理所有样本，得到 ids 和 labels
    all_sequences = []
    dropped = 0

    for ex in batch:
        source = ex.get("source", "ultrachat")
        system = ex.get("system", "")
        user = ex.get("user", "")
        assistant = normalize_output_by_domain(ex.get("assistant_raw", ""), source=source)
        chem_stage = ex.get("chem_stage", "reason_plus_smiles")

        # 构造 prompt
        # 对于 Mistral 等模型，chat template 可能对格式要求严格，如果失败则回退到自定义格式
        prompt = None
        if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role":"system","content":system},{"role":"user","content":user}],
                    add_generation_prompt=True, tokenize=False
                )
            except Exception as e:
                # 如果 chat template 失败（如格式不符合要求），回退到自定义格式
                prompt = None
        
        if prompt is None:
            prompt = f"System: {system}\nUser: {user}\nAssistant: "

        p_ids = tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
        a_text = assistant + (tokenizer.eos_token if (add_eos and tokenizer.eos_token) else "")
        a_ids = tokenizer(a_text, add_special_tokens=False).get("input_ids", [])

        # 长度裁剪
        prompt_keep_ratio = 0.35
        domain_min_keep = {"chem": 512, "ultrachat": 256}
        domain_max_keep = {"chem": 1024, "ultrachat": 512}
        src = source if source in domain_min_keep else "ultrachat"

        target_keep_p = int(max_length * prompt_keep_ratio)
        min_keep_p = domain_min_keep[src]
        max_keep_p = domain_max_keep[src]
        keep_p = max(min_keep_p, min(target_keep_p, max_keep_p))
        remain_for_a = max_length - keep_p

        if len(a_ids) > remain_for_a:
            a_ids = a_ids[-remain_for_a:]
        keep_p = max(0, max_length - len(a_ids))
        if keep_p < min_keep_p and (len(a_ids) + min_keep_p) <= max_length:
            keep_p = min_keep_p
        if len(p_ids) > keep_p:
            p_ids = p_ids[-keep_p:]

        ids = p_ids + a_ids
        if len(ids) == 0 or len(ids) > max_length:
            dropped += 1
            continue

        # 生成 labels
        if source == "chem" and smiles_only:
            labs = [IGNORE_INDEX] * len(p_ids)
            m = _SMILES_LINE.search(assistant)
            if m:
                reasoning_prefix = assistant[:m.start()].strip()
            else:
                reasoning_prefix = assistant
            reasoning_ids = tokenizer(reasoning_prefix, add_special_tokens=False).get("input_ids", [])
            reasoning_end_pos = len(reasoning_ids)
            labs.extend([IGNORE_INDEX] * reasoning_end_pos)
            smiles_ids = a_ids[reasoning_end_pos:]
            labs.extend(smiles_ids)
        else:
            labs = [IGNORE_INDEX] * len(p_ids) + a_ids

        all_sequences.append({
            "ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "length": len(ids)
        })

    if not all_sequences:
        raise RuntimeError("[sft_collate_with_packing] all samples in the batch became empty after processing "
                         f"(dropped={dropped}). Check tokenization/normalization or max_length={max_length} too small.")

    # 打包序列：尽可能将多个短序列合并到一个 max_length 的序列中
    packed_sequences = []
    current_ids = []
    current_labels = []
    current_length = 0

    for seq in all_sequences:
        seq_ids = seq["ids"].tolist()
        seq_labels = seq["labels"].tolist()
        seq_len = seq["length"]

        # 如果当前序列加上新序列不超过 max_length，则打包
        if current_length + seq_len <= max_length:
            # 添加 eos_token 作为分隔符（如果当前已有内容）
            if current_length > 0 and tokenizer.eos_token_id is not None:
                current_ids.append(tokenizer.eos_token_id)
                current_labels.append(IGNORE_INDEX)  # eos_token 分隔符不计算损失
                current_length += 1

            current_ids.extend(seq_ids)
            current_labels.extend(seq_labels)
            current_length += seq_len
        else:
            # 当前序列已满，保存并开始新的
            if current_length > 0:
                # pad 到 max_length
                while len(current_ids) < max_length:
                    current_ids.append(tokenizer.pad_token_id)
                    current_labels.append(IGNORE_INDEX)
                packed_sequences.append({
                    "ids": torch.tensor(current_ids[:max_length], dtype=torch.long),
                    "labels": torch.tensor(current_labels[:max_length], dtype=torch.long)
                })
            
            # 开始新序列
            current_ids = seq_ids[:max_length]
            current_labels = seq_labels[:max_length]
            current_length = len(current_ids)
            
            # pad 到 max_length
            while len(current_ids) < max_length:
                current_ids.append(tokenizer.pad_token_id)
                current_labels.append(IGNORE_INDEX)

    # 处理最后一个序列
    if current_length > 0:
        while len(current_ids) < max_length:
            current_ids.append(tokenizer.pad_token_id)
            current_labels.append(IGNORE_INDEX)
        packed_sequences.append({
            "ids": torch.tensor(current_ids[:max_length], dtype=torch.long),
            "labels": torch.tensor(current_labels[:max_length], dtype=torch.long)
        })

    # 转换为 batch 格式
    input_ids = torch.stack([s["ids"] for s in packed_sequences])
    labels = torch.stack([s["labels"] for s in packed_sequences])
    attention = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


from torch.utils.data import Sampler

def build_sorted_indices(dataset, tokenizer):
    lengths = []
    for i, ex in enumerate(dataset):
        try:
            total_len = _token_len_of(ex, tokenizer)
        except Exception as e:
            # 出问题就跳过，并打印一次
            if i < 5:
                print(f"[LEN-ERR] sample idx={i} error: {e}")
            continue
        lengths.append((i, total_len))

    if not lengths:
        raise RuntimeError("[build_sorted_indices] got 0 valid samples after length computing")

    lengths.sort(key=lambda x: x[1])
    idx = [i for (i, _) in lengths]
    # 调试：打印最小/最大长度
    mins = lengths[0][1]; maxs = lengths[-1][1]
    print(f"[LEN] after sort: n={len(idx)} min={mins} max={maxs}")
    return idx


class SortedBatchSampler(Sampler):
    def __init__(self, sorted_indices, batch_size):
        self.sorted_indices = list(sorted_indices)
        self.batch_size = int(batch_size)

    def __iter__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        n = len(self.sorted_indices)
        # 调试：前几组 size
        for i in range(0, n, self.batch_size):
            chunk = self.sorted_indices[i:i + self.batch_size]
            if not chunk:         # 兜底：跳过空块
                continue
            yield chunk

    def __len__(self):
        if self.batch_size <= 0:
            return 0
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


# ----------------------------
# 直接运行：测试加载 & 统计
# ----------------------------
if __name__ == "__main__":
    CHEM_PATH = "/data2/liuhaoran/clean/reasoning_process_cleaned_valid.jsonl"
    ULTRA_DIR = "/data2/liuhaoran/General_Instructions/HuggingFaceH4___ultrachat_200k/default/0.0.0/ultrachat/jsonl"

    data = load_mixed_data(
        chem_jsonl_path=CHEM_PATH,
        ultrachat_dir=ULTRA_DIR,
        chem_repeat=3,         # 可改为 4/5
        ultrachat_limit=20000, # 只取前 2 万条
        seed=42,
    )

    print(f"[INFO] mixed samples: {len(data)}")
    # 打印几条检查
    for i in range(min(3, len(data))):
        ex = data[i]
        print("----")
        print("source:", ex["source"])
        print("system:", ex["system"][:80].replace("\n"," ") + ("..." if len(ex["system"])>80 else ""))
        print("user  :", ex["user"][:80].replace("\n"," ") + ("..." if len(ex["user"])>80 else ""))
        print("asst  :", normalize_output_by_domain(ex["assistant_raw"], ex["source"])[:120].replace("\n"," | ") + "...")

    # 训练脚本中用法：
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(<your-model-or-tokenizer>)
    # collate_fn = lambda batch: sft_collate(batch, tokenizer, max_length=2048, add_eos=True)
    # dataloader = DataLoader(data, batch_size=..., shuffle=True, collate_fn=collate_fn)
