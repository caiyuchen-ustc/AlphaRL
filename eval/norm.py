import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from tqdm import tqdm

def save_svd_components(model1_path, model2_path, base_output_path, start_step=9, end_step=10):
    # 创建基础输出路径（如果不存在）
    norm = 0
    count = 0
    os.makedirs(base_output_path, exist_ok=True)

    # 加载原始tokenizer和模型
    print(f"加载原始模型 {model1_path}")
    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    model1 = AutoModelForCausalLM.from_pretrained(model1_path)
    model1.to('cuda')
    print("原始模型和分词器加载完成。")

    # 遍历不同的global step
    print(start_step)
    print(end_step)
    for global_step in tqdm(range(start_step, end_step + 1)):
        print(f"\n正在处理 global_step {global_step}")
        
        # 构建当前步骤的模型路径和输出路径
        if "aa" in model2_path:
            current_model2_path = model2_path.format(i=global_step)
            current_output_path = os.path.join(base_output_path, f"global_step_{global_step}")
            
            # 创建当前步骤的输出目录
            os.makedirs(current_output_path, exist_ok=True)
        else:
            current_model2_path = model2_path
            current_output_path = model2_path

        
        # 加载当前步骤的模型
        print(f"加载模型 {current_model2_path}")
        model2 = AutoModelForCausalLM.from_pretrained(current_model2_path)
        model2.to('cuda')
        # 存储SVD分解结果的字典
        svd_components = {}

        # 无梯度计算以提高性能
        with torch.no_grad():
            # 遍历所有层
            for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):
                print(f"\n处理 layer {layer_idx}")
                layer_svd = {}
                
                # 处理self_attn模块
                print("  处理 self_attn 模块:")
                for name, param in layer2.self_attn.named_parameters():
                    if name.endswith('.weight'):
                        print(f"    处理 self_attn.{name}")
                        ref_param = layer1.self_attn.get_parameter(name)
                        weight_update = param - ref_param

                        norm += torch.norm(weight_update)
                        count += 1
                
                # 处理mlp模块
                print("  处理 mlp 模块:")
                for name, param in layer2.mlp.named_parameters():
                    if name.endswith('.weight'):
                        print(f"    处理 mlp.{name}")
                        ref_param = layer1.mlp.get_parameter(name)
                        weight_update = param - ref_param
                        norm += torch.norm(weight_update)
                        count += 1
    print(norm/count)  
    return norm/count

# 设置路径
model1_path = ""
model2_path = ""
base_output_path  = ""
# 执行保存
start_step=1
end_step=1

save_svd_components(model1_path, model2_path, base_output_path, start_step, end_step)
