import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def reconstruct_and_save_rank1(model1_path, model2_path, svd_components_base_path, start_step, end_step):
    # 提前加载配置和tokenizer（只加载一次，提升效率）
    print("Loading base config and tokenizer...")
    original_config = AutoConfig.from_pretrained(model1_path)
    target_dtype = original_config.torch_dtype  # 获取原始模型数据类型
    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    # 确保tokenizer有pad_token（避免生成时警告）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device ='cpu'
    print(f"Using device: {device}")
    norm=0

    count=0
    svg_norm=0
    svg_count=0
    for top_k in range(0, 2, 10):
        # 处理每个global_step
        for global_step in range(start_step, end_step + 1):
            print(f'\n===== Processing top_k: {top_k} =====')
            # 加载原始模型（每个top_k重新加载一次确保参数纯净）
            print("Loading original models...")
            model1 = AutoModelForCausalLM.from_pretrained(
                model1_path,
                torch_dtype=target_dtype,
                device_map=device  # 自动分配设备
            )
            model2_paths = os.path.join(model2_path)
            model2 = AutoModelForCausalLM.from_pretrained(
                model2_paths,
                torch_dtype=target_dtype,
                device_map=device  # 自动分配设备
            )
            print("Original models loaded successfully")
            try:
                print(f"\n----- Processing global_step {global_step} -----")
                # 构建文件路径

                if "aa" in model2_path:
                    svd_file = os.path.join(svd_components_base_path, f"global_step_{global_step}", 'svd_components.pt')
                    output_dir = os.path.join(svd_components_base_path, f"global_step_{global_step}", f'rank_1')
                
                # elif "hf_model" in model2_path:
                #     svd_file = os.path.join(svd_components_base_path, 'svd_components.pt')
                #     output_dir = os.path.join(svd_components_base_path, f'top{top_k}')

                else:
                    svd_file = os.path.join(svd_components_base_path,'svd_components.pt')
                    output_dir = os.path.join(svd_components_base_path, f'rank_1')

                # 检查SVD文件是否存在
                if not os.path.exists(svd_file):
                    print(f"SVD file not found: {svd_file} - skipping")
                    continue

                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)

                # 加载SVD组件（指定map_location避免设备冲突）
                svd_components = torch.load(svd_file, map_location=device)

                print("Starting model parameter updates...")

                # 逐层处理
                for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):
                    print(f"  Processing layer {layer_idx}")
                    layer_svd = svd_components.get(f'layer_{layer_idx}', {})
                    if not layer_svd:
                        print(f"  No SVD data for layer {layer_idx} - skipping")
                        continue

                    # 处理自注意力模块
                    for (name1, param1), (name2, param2) in zip(layer1.self_attn.named_parameters(), 
                                                               layer2.self_attn.named_parameters()):
                        # 确保参数名称匹配
                        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
                        
                        if name1.endswith('.weight'):
                            key_U = f'self_attn_{name1}_U'
                            key_S = f'self_attn_{name1}_S'
                            key_Vt = f'self_attn_{name1}_Vt'
                            
                            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:
                                # 加载并转移到目标设备
                                U = layer_svd[key_U].to(device, dtype=target_dtype)
                                S = layer_svd[key_S].to(device, dtype=target_dtype)
                                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)

                                # 计算top k分量
                                k = max(1, int(len(S) * top_k / 100))
                                #k= top_k
                                print(f"    Using top {k}/{len(S)} components for self_attn.{name1}")

                                # 提取top k分量
                                U_top = U[:, :k]
                                S_top = S[:k]
                                V_top = Vt[:k, :]
                                # import pdb
                                # pdb.set_trace()
                                # 计算USV并更新model2的参数
                                S_matrix = torch.diag(S_top)
                                USV_top = U_top @ S_matrix @ V_top  # 直接使用V，因为torch.svd返回的V已经是转置后的

                                norm_factor = torch.norm(U@torch.diag(S)@Vt)  # Compute the norm
                                
                                count+=1
                                

                                USV_top_norm = torch.norm(USV_top)
                                norm+=USV_top_norm/norm_factor
                                #print(norm_factor/USV_top_norm)
                                svg_norm +=S_top.sum()/S.sum()
                                param1.data = param1.data + 0.7* (norm_factor/USV_top_norm)*USV_top

                                # S_matrix = torch.diag(S_top)
                                # USV_top = U_top @ S_matrix @ V_top
                                #param2.data = param1.data + USV_top    # 将model2参数设为model1参数加上USV_top
                                print(f"    Updated model1 self_attn.{name1} parameters")

                    # 处理MLP模块
                    for (name1, param1), (name2, param2) in zip(layer1.mlp.named_parameters(), 
                                                               layer2.mlp.named_parameters()):
                        # 确保参数名称匹配
                        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
                        
                        if name1.endswith('.weight'):
                            key_U = f'mlp_{name1}_U'
                            key_S = f'mlp_{name1}_S'
                            key_Vt = f'mlp_{name1}_Vt'
                            
                            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:
                                # 加载并转移到目标设备
                                U = layer_svd[key_U].to(device, dtype=target_dtype)
                                S = layer_svd[key_S].to(device, dtype=target_dtype)
                                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)

                                # 计算top k分量
                                k = max(1, int(len(S) * top_k / 100))
                                #k=top_k
                                print(f"    Using top {k}/{len(S)} components for mlp.{name1}")

                                # 提取top k分量
                                U_top = U[:, :k]
                                S_top = S[:k]
                                V_top = Vt[:k, :]

                                # 计算USV并更新model2的参数
                                S_matrix = torch.diag(S_top)
                                USV_top = U_top @ S_matrix @ V_top  # 直接使用V，因为torch.svd返回的V已经是转置后的

                                norm_factor = torch.norm(U@torch.diag(S)@Vt)  # Compute the norm
                                USV_top_norm = torch.norm(USV_top)
                                print('---------')
                                print(norm_factor)
                                print(USV_top_norm)
                                print('---------')
                                param1.data = param1.data + 0.4* (norm_factor/USV_top_norm)*USV_top
                                norm+=USV_top_norm/norm_factor
                                # norm+=norm_factor
                                #svg_norm +=S_top.sum()/S.sum()
                                count+=1
                                # S_matrix = torch.diag(S_top)
                                # USV_top = U_top @ S_matrix @ V_top
                                #param2.data = param1.data + USV_top    # 将model2参数设为model1参数加上USV_top
                                print(f"    Updated model1 mlp.{name1} parameters")

                # 保存模型（确保数据类型一致）
                # print(f"Saving model to {output_dir} with dtype {target_dtype}")
                model1 = model1.to('cpu')  # 转移到CPU再保存
                model1.save_pretrained(
                    output_dir,
                    torch_dtype=target_dtype,
                    safe_serialization=True
                )
                # tokenizer.save_pretrained(output_dir)
                # print(f"Successfully saved rank-1 model for global_step {global_step} (top_k={top_k})")

            except KeyboardInterrupt:
                print(f"\nUser interrupted - skipping global_step {global_step}")
                continue
            except Exception as e:
                print(f"\nError processing global_step {global_step}: {str(e)}")
                continue
            finally:
                torch.cuda.empty_cache()  # 清理GPU缓存

            # 每个top_k循环结束后清理模型
            del model1, model2, svd_components
            torch.cuda.empty_cache()  # 强制清理GPU缓存
            print(f"Released GPU memory after top_k={top_k}\n")
    print(norm/count)
    # print(svg_norm/count)
    print("All processes completed")


# 配置路径

model1_path = ""
model2_path = ""
svd_components_base_path = ""

# 执行任务（处理step 1024）
start_step = 1
end_step = 1
reconstruct_and_save_rank1(model1_path, model2_path, svd_components_base_path, start_step, end_step)
