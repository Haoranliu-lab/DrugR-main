from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm


from admet import compute_reward_with_bad_feats
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


gen_device = 6    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 1000
Q_batch_size = 1
num_pre_Q = 2
train_batch_size = 1
gen_update_steps = 16
save_steps = 200
compute_gen_logps = True
clip_param = 0.2


from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.float().log_softmax(dim=-1)  # <<< cast to float32
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps, dim=0)

#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather
"""
def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device)#.unsqueeze(1) TODO: (B, L-1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss
"""
def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device)

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length-1:]

    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # 只保留生成长度>0的样本
    valid = completion_mask.sum(dim=1) > 0
    if not torch.any(valid):
        return torch.zeros((), device=engine.device, requires_grad=True)

    per_token_logps = per_token_logps[valid]
    advantages      = advantages[valid]
    per_token_kl    = per_token_kl[valid]
    completion_mask = completion_mask[valid]

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device)[valid])
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    denom = completion_mask.sum(dim=1).clamp_min(1)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()
    return loss
def gen_worker(Q, physics_device):
    import re, json, time, random
    from vllm_train import LLM, SamplingParams
    from transformers import AutoTokenizer
    from torch.nn.utils.rnn import pad_sequence

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{physics_device}"
    torch.cuda.set_device(0)
    print(f"[vLLM] Generation worker process uses GPU {physics_device}")

    os.makedirs(LOG_DIR, exist_ok=True)
    ROLLOUT_LOG = os.path.join(LOG_DIR, "rollouts2.jsonl")

    # === vLLM 实例 ===
    VLLM_MAXLEN = 2096          # 必须与 LLM(..., max_model_len=2096) 一致
    BASE_MAX_NEW = 656          # 你的目标生成上限
    GEN_MIN_TOKENS = 64         # 至少给生成留的 token 数（避免 prompt 占满导致不出字）
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5, max_model_len=VLLM_MAXLEN)

    # === Tokenizer（与主进程无共享）===
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 统一 STOP 标记（建议用 "<END>"）===
    STOP_TOKEN = "<END>"

    # ====== 任务相关：数据/模板/解析 ======

    def build_prompt_text(instruction: str, user_text: str) -> str:
        sys_msg = normalize_instruction(instruction)
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True
        )

    STRICT_TWO_LINES = re.compile(
        r"(?s)^(?:.*?</THINK>\s*)?"                # 可选 scratchpad
        r"\s*Reasoning\s*:\s*(?P<reason>[^\n]+?)\s*"
        r"\n\s*Optimized_SMILES\s*:\s*(?P<smiles>\S+?)\s*"
        r"(?:</?END>)?\s*$",                      # 兼容 <END> 与 </END>
        re.IGNORECASE
    )

    def strict_format_ok(text: str) -> bool:
        return STRICT_TWO_LINES.search(text or "") is not None

    def parse_two_lines(text: str):
        m = STRICT_TWO_LINES.search(text or "")
        if not m:
            return "", ""
        return m.group("reason").strip(), m.group("smiles").strip()

    def extract_orig_smiles_from_input(user_text: str) -> str:
        m = re.search(r"^\s*Original\s+SMILES\s*:\s*(\S+)", user_text or "", flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    # ====== 载入训练/生成数据 ======

    assert os.path.exists(TRAIN_FILE), f"File not found: {TRAIN_FILE}"

    def load_reasoning_file(file_path: str):
        out = []
        with open(file_path, "r", encoding="utf-8") as f:
            # JSON list or JSONL
            first_non_ws = None
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
            f.seek(0)
            if first_non_ws == "[":
                objs = json.load(f)
                for obj in objs:
                    ins = str(obj.get("instruction", "")).strip()
                    inp = str(obj.get("input", "")).strip()
                    outp = str(obj.get("output", "")).strip()
                    out.append({"instruction": ins, "input": inp, "output": outp})
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ins = str(obj.get("instruction", "")).strip()
                    inp = str(obj.get("input", "")).strip()
                    outp = str(obj.get("output", "")).strip()
                    out.append({"instruction": ins, "input": inp, "output": outp})
        return out

    dataset = load_reasoning_file(TRAIN_FILE)
    QAs = [{"I": ex["instruction"], "Q": ex["input"], "A": ex.get("output", "")} for ex in dataset]
    print("[vLLM] dataset loaded. example:", QAs[0] if QAs else None)

    # ====== 生成核心 ======
    def gen_answers(items):
        """对每条样本动态裁剪 prompt 并设置 max_tokens；Q_batch_size=1 时开销可忽略。"""
        tip_texts, sps = [], []
        for ex in items:
            t = build_prompt_text(ex["I"], ex["Q"])

            # 计算 prompt token 长度并（如必要）裁剪到 VLLM_MAXLEN - GEN_MIN_TOKENS
            ids = tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            plen = int(ids.size(0))
            max_prompt = VLLM_MAXLEN - GEN_MIN_TOKENS
            if plen > max_prompt:
                ids = ids[-max_prompt:]                                  # 保留结尾更可能含关键信息
                t = tokenizer.decode(ids, skip_special_tokens=False)
                plen = max_prompt
                print(f"[clip] prompt too long, keep tail to {plen} tokens")

            # 动态决定本条生成上限；再预留 1 个 token
            allowed = max(GEN_MIN_TOKENS, min(BASE_MAX_NEW, VLLM_MAXLEN - plen - 1))
            print(f"[gen] plen={plen} allowed_gen={allowed}")

            sp = SamplingParams(
                n=num_pre_Q,
                temperature=0.7,
                max_tokens=int(allowed),
                stop=[STOP_TOKEN],
                logprobs=1,
            )
            tip_texts.append(t)
            sps.append(sp)

        answers, ans_token_ids, ans_token_logps = [], [], []
        for t, sp in zip(tip_texts, sps):
            voutputs = vllm_gen.generate([t], sp, use_tqdm=False)
            for v in voutputs:
                for z in v.outputs:
                    txt = (z.text or "")
                    answers.append(txt)
                    ans_token_ids.append(z.token_ids)

                    if hasattr(z, "token_logprobs") and z.token_logprobs is not None:
                        lp = [tok.logprob for tok in z.token_logprobs]
                        ans_token_logps.append(torch.tensor(lp, dtype=torch.float32))
                    else:
                        ans_token_logps.append(torch.zeros(len(z.token_ids), dtype=torch.float32))

                    # 对齐长度（保险）
                    if len(ans_token_logps[-1]) != len(ans_token_ids[-1]):
                        L = min(len(ans_token_logps[-1]), len(ans_token_ids[-1]))
                        ans_token_logps[-1] = ans_token_logps[-1][:L]
                        ans_token_ids[-1] = ans_token_ids[-1][:L]

                    # 可视化检查（定位“空输出/过早stop”）
                    head = txt.strip()[:80].replace("\n", " ")
                    print(f"[gen one] out_len={len(txt.strip())} head={head!r}")

        print(f"[gen] generated {len(answers)} answers for {len(items)} prompts")
        return tip_texts, answers, ans_token_ids, ans_token_logps

    def reward_admet(orig_smiles: str, answer_text: str) -> float:
        # 先严格校验两行输出
        if not strict_format_ok(answer_text):
            return -2.0
        reasoning, opt_smi = parse_two_lines(answer_text)
        if not reasoning or not opt_smi:
            return -2.0
        try:
            total_reward, _ = compute_reward_with_bad_feats(orig_smiles, opt_smi, reasoning)
            return float(max(min(total_reward, 10.0), -10.0))
        except Exception as e:
            print("[reward_admet] error:", e)
            return -2.0

    def spread_to_tokens(ans_token_ids, avg_reward: float):
        L = len(ans_token_ids)
        if L <= 0:
            return torch.tensor([], dtype=torch.float32)
        return torch.full((L,), float(avg_reward), dtype=torch.float32)

    def gen_samples(inputs):
        prompts_text, answers, ans_token_ids, ans_token_logps = gen_answers(inputs)

        rewards = []
        per_token_rewards = []
        for i, ex in enumerate(inputs):
            orig_smiles = extract_orig_smiles_from_input(ex["Q"])
            for j in range(num_pre_Q):
                a = answers[i * num_pre_Q + j]
                t = ans_token_ids[i * num_pre_Q + j]
                r = reward_admet(orig_smiles, a)          # 不合格自动 -2.0
                rewards.append(r)
                per_token_rewards.append(spread_to_tokens(t, r))

        return prompts_text, torch.tensor(rewards, dtype=torch.float32), per_token_rewards, answers, ans_token_ids, ans_token_logps

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print("[vLLM] recving new model ...")
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print("[vLLM] model updated")
            del new_state_dict
        except Exception:
            return

    # ====== 主循环 ======
    for it in range(10**9):
        if it % 3 == 0:
            try_update_model()

        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, per_token_rewards, answers, ans_token_ids, ans_token_logps = gen_samples(inputs)
        print(f"time: {time.time()-tic:.2f}s    rewards: {rewards}")

        # --- 把本 batch 的候选写入 logs/rollouts.jsonl ---
        try:
            with open(ROLLOUT_LOG, "a", encoding="utf-8") as fw:
                for i, ex in enumerate(inputs):
                    orig_smiles = extract_orig_smiles_from_input(ex["Q"])
                    for j in range(num_pre_Q):
                        idx = i * num_pre_Q + j
                        # 兜底防越界/空输出
                        ans_txt = answers[idx] if idx < len(answers) else ""
                        out_ids = ans_token_ids[idx] if idx < len(ans_token_ids) else []
                        ok = strict_format_ok(ans_txt)
                        reason, smi = parse_two_lines(ans_txt) if ok else ("", "")
                        rec = {
                            "time": time.time(),
                            "iter": it,
                            "prompt": prompt_inputs[i],  # 实际用于生成的 prompt（已被模板+截断）
                            "orig_smiles": orig_smiles,
                            "answer": ans_txt,
                            "reasoning": reason,
                            "smiles": smi,
                            "reward": float(rewards[idx].item()) if torch.is_tensor(rewards[idx]) else float(
                                rewards[idx]),
                            "ok_format": bool(ok),
                            "out_len": int(len(out_ids)),
                        }
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[log] rollout write failed: {e.__class__.__name__}: {e}")

        if it % 5 == 0 and answers:
            print("answers[0]:", answers[0])

        # 批次 prompt 长度统计（当前 batch）
        batch_prompt_lens = []
        for pt in prompt_inputs:
            _ids = tokenizer(pt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            batch_prompt_lens.append(int(_ids.size(1)))
        if batch_prompt_lens:
            _avg = sum(batch_prompt_lens) / len(batch_prompt_lens)
            print(f"[batch prompt] count={len(batch_prompt_lens)}  mean={_avg:.1f}  "
                  f"max={max(batch_prompt_lens)}  head={batch_prompt_lens[:4]}")

        # 逐 prompt 打包上传
        for i, prompt_text in enumerate(prompt_inputs):
            # 1) 计算 prompt token 数
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = int(prompt_ids.size(1))

            # 2) 取出该 prompt 的 num_pre_Q 个候选
            start = i * num_pre_Q
            end = start + num_pre_Q
            curr_ans_ids = ans_token_ids[start:end]
            curr_rewards = rewards[start:end].tolist()
            curr_token_rewards = per_token_rewards[start:end]
            sub_ans_logps = ans_token_logps[start:end]

            # 过滤空样本（避免 pad 报错）
            keep_idx = [k for k, ids in enumerate(curr_ans_ids) if len(ids) > 0]
            if len(keep_idx) == 0:
                print("[gen] all candidates empty -> skip this prompt")
                continue
            curr_ans_ids = [curr_ans_ids[k] for k in keep_idx]
            curr_rewards = [curr_rewards[k] for k in keep_idx]
            curr_token_rewards = [curr_token_rewards[k] for k in keep_idx]
            sub_ans_logps = [sub_ans_logps[k] for k in keep_idx]

            # 3) 组内标准化 per-token 奖励（提升稳定性）
            rew_vec = torch.tensor(curr_rewards, dtype=torch.float32)
            mean, std = rew_vec.mean(), rew_vec.std()
            if float(std) < 1e-4:
                std = torch.tensor(1e-4)
            norm_token_rewards = [(tr - mean) / float(std) for tr in curr_token_rewards]

            # 4) pad 成等长
            pad_rewards = pad_sequence(norm_token_rewards, batch_first=True, padding_value=0.0)
            output_ids = pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in curr_ans_ids],
                                      batch_first=True, padding_value=tokenizer.pad_token_id)
            pad_gen_logps = pad_sequence(sub_ans_logps, batch_first=True, padding_value=0.0)

            # 5) 拼接完整输入
            Qrep = prompt_ids.repeat(output_ids.size(0), 1)
            merged_ids = torch.cat([Qrep, output_ids], dim=1)

            # 6) 上传（inputs / rewards / gen_logps）
            payload = make_bytes_list([
                json.dumps({"plen": plen}).encode(),
                tensor_to_bytes(merged_ids.cpu()),
                tensor_to_bytes(pad_rewards.cpu()),
                tensor_to_bytes(pad_gen_logps.cpu()),
            ])
            try:
                resp = requests.post(f"{ref_server}/upload", data=payload, timeout=10)
                print(f"[upload] ok: status={resp.status_code} plen={plen} "
                      f"B={output_ids.size(0)} L={output_ids.size(1)}")
            except Exception as e:
                print(f"[upload] failed: {e.__class__.__name__}: {e}")







#tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()



