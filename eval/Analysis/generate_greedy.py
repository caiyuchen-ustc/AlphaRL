import sys
import os
import argparse
sys.path.append('/project/ugmathllm/caiyuchen/LIMO/eval')

from utils.data_loader import load_data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ====== 解析命令行参数 ======
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="1", help="使用的GPU编号，如 '0' 或 '0,1'")
parser.add_argument("--output_prefix", type=str, default="base_aime_results", help="输出文件名前缀")
args = parser.parse_args()

# 设置可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cdevice = 0

# ====== 加载模型 ======
model_path = ''
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": cdevice}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== 函数 ======
def parse_question(example, data_name=None):
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            return example[key].strip()
    return ""

def apply_template(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"content": question, "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
    )

# ====== 加载数据 ======
for data in ["math",'aime24','aime25','minerva','gpqa','gsm8k']:
    examples = load_data(data, "test", "")
    prompt_batch = []

    for idx, example in tqdm(enumerate(examples), desc="准备提示词"):
        question = parse_question(example, "math")
        #str_to_remove1 = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        #str_to_remove2 = "Remember to put your answer on its own line after \"Answer:\".\n\n"
        question_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
        cur_prompt = question + question_prompt
        # cur_prompt = apply_template(tokenizer=tokenizer, question=question)
        # cur_prompt = apply_r1_template(question)
        prompt_batch.append(cur_prompt)

    # ====== 生成 ======
    results = []
    for prompt in tqdm(prompt_batch[:10], desc="生成回答"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=6000,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0])
        results.append({
            "prompt": prompt,
            "full_text": generated_text
        })

    print(f"生成完成，共处理 {len(results)} 个样本")
    print("第一个样本的合并结果预览：")
    # print(results[0]["full_text"][:500] + "...")

    # ====== 保存到文件 ======
    output_file = f""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到 {output_file}")
