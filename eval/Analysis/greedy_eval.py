import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

def compute_perplexity(logits, input_ids, start_pos=0):
    """
    计算从 start_pos 开始（使用 logits[start_pos:seq_len-1] 预测 tokens start_pos+1:seq_len）的 perplexity。
    logits: model outputs logits, shape (1, seq_len, vocab)
    input_ids: tensor shape (1, seq_len)
    start_pos: int
    返回 float perplexity 或 NaN（如果没有可计算的 target）
    """
    seq_len = input_ids.shape[1]
    if seq_len <= start_pos + 1:
        return float("nan")  # 没有可计算的 target

    pred_logits = logits[0, start_pos:seq_len-1, :]  # (checked_len, vocab)
    target_ids = input_ids[0, start_pos+1:seq_len]  # (checked_len,)

    # 为数值稳定性，必要时将 logits 转为 float32
    pred_logits = pred_logits.float()

    log_probs = F.log_softmax(pred_logits, dim=-1)  # (checked_len, vocab)
    target_log_probs = log_probs[range(target_ids.shape[0]), target_ids]  # (checked_len,)
    nll = -target_log_probs.mean().item()
    ppl = float(torch.exp(torch.tensor(nll)))
    return ppl

def check_greedy_decisions(tokenizer, model, pre_text, text, default_skip_tokens=100, context_window=20):
    """
    检查模型在给定文本中是否在给定区间采用贪心选择（即模型 argmax 是否等于实际下一个 token）。
    返回：ratio, non_greedy_tokens, position_percents, perplexity
      - ratio: non-greedy token 数 / checked token 数
      - non_greedy_tokens: list of decoded token strings (过滤空字符串)
      - position_percents: list of corresponding relative positions (%)
      - perplexity: perplexity（针对从 start_pos 开始的 checked 区间）
    """
    # 1. 计算实际 skip 长度
    pre_inputs = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False) if pre_text else None
    actual_skip_tokens = pre_inputs['input_ids'].shape[1] if pre_inputs is not None else default_skip_tokens

    # 2. 编码 text
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)  # shape (1, seq_len)
    seq_len = input_ids.shape[1]

    # 3. 确定 start_pos（与原逻辑一致）
    start_pos = min(actual_skip_tokens, seq_len - 2)
    if start_pos >= seq_len - 1:
        # 无可检查 tokens
        return 0.0, [], [], float("nan")

    # 4. 取得 logits（在 no_grad 下）
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    # 5. 计算困惑度（从 start_pos 开始预测）
    ppl = compute_perplexity(logits, input_ids, start_pos=start_pos)

    # 6. 准备用于向量化比较的张量（转到 CPU）
    #    check_logits 用于预测 start_pos .. seq_len-2 的下一个 token（对应 actual_tokens start_pos+1 .. seq_len-1）
    check_logits = logits[0, start_pos:seq_len-1, :].cpu()  # (checked_len, vocab)
    max_tokens = torch.argmax(check_logits, dim=-1)  # (checked_len,)
    actual_tokens = input_ids[0, start_pos+1:seq_len].cpu()  # (checked_len,)
    total_checked = actual_tokens.shape[0]

    # 7. 向量化找到不匹配的位置
    mismatch_mask = actual_tokens != max_tokens  # (checked_len,)
    mismatch_count = int(mismatch_mask.sum().item())

    # 8. 批量解码不匹配 token ids（并过滤空解码）
    if mismatch_count > 0:
        mismatch_ids = actual_tokens[mismatch_mask].tolist()  # list of ints
        # batch_decode 并去掉空/空白
        decoded = tokenizer.batch_decode(mismatch_ids, skip_special_tokens=True)
        non_greedy_tokens = [t.strip() for t in decoded if t.strip()]
    else:
        non_greedy_tokens = []

    # 9. 计算相对位置百分比并筛选不匹配位置
    #    positions range: start_pos .. start_pos + total_checked - 1
    positions = torch.arange(start_pos, start_pos + total_checked)
    relative_positions = (positions.float() / seq_len) * 100.0
    position_percents = relative_positions[mismatch_mask].tolist() if mismatch_count > 0 else []

    # 10. 计算 ratio（使用 total_checked 与原实现一致）
    ratio = mismatch_count / total_checked if total_checked > 0 else 0.0

    return ratio, non_greedy_tokens, position_percents, ppl


if __name__ == "__main__":
    model_path = ''

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for data in ["math"]:

        input_file = f""
        with open(input_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        all_tokens, all_positions, all_ratios = [], [], []
        all_ppls = []

        for result in tqdm(results, desc="Analyzing greedy decisions"):
            ratio, tokens, positions, ppl = check_greedy_decisions(
                tokenizer=tokenizer,
                model=model,
                pre_text=result["prompt"],
                text=result["full_text"],
                default_skip_tokens=100,
                context_window=30
            )
            all_ratios.append(ratio)
            all_tokens.extend(tokens)
            all_positions.extend(positions)
            all_ppls.append(ppl)

        overall_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0
        # 统计有效 perplexities（去掉 nan）
        valid_ppls = [p for p in all_ppls if not (p is None or (isinstance(p, float) and (p != p)))]  # remove NaN
        mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan")
        median_ppl = sorted(valid_ppls)[len(valid_ppls)//2] if valid_ppls else float("nan")

        print(f"Dataset: {data}")
        print(f"Overall non-greedy ratio: {overall_ratio:.2%}")
        print(f"Unique non-greedy tokens: {len(set(all_tokens))}")
        print(f"Total non-greedy tokens: {len(all_tokens)}")
        print(f"Mean perplexity (per-sample, checked region): {mean_ppl:.4f}")
        print(f"Median perplexity (per-sample, checked region): {median_ppl:.4f}")

