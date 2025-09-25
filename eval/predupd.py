import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def reconstruct_and_save_rank1_with_pred_u(
    model1_path,
    model2_path,
    svd_components_base_path,
    predicted_u_file,                      # <- 新增：predicted_vectors.pt 路径
    output_subdir="rank1_predu"            # 输出目录后缀，避免覆盖
):
    print("Loading base config and tokenizer...")
    original_config = AutoConfig.from_pretrained(model1_path)
    target_dtype = original_config.torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading original models...")
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_path, torch_dtype=target_dtype, device_map=device
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model2_path, torch_dtype=target_dtype, device_map=device
    )
    print("Original models loaded successfully")

    # SVD 组件与输出目录
    svd_file = os.path.join(svd_components_base_path, "svd_components.pt")
    if not os.path.exists(svd_file):
        raise FileNotFoundError(f"SVD file not found: {svd_file}")

    output_dir = os.path.join(svd_components_base_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 载入 SVD 字典与 预测的 u 向量字典
    print(f"Loading SVD components from: {svd_file}")
    svd_components = torch.load(svd_file, map_location=device)

    if not os.path.exists(predicted_u_file):
        raise FileNotFoundError(f"Predicted-u file not found: {predicted_u_file}")
    print(f"Loading predicted u vectors from: {predicted_u_file}")
    predicted_u_dict = torch.load(predicted_u_file, map_location=device)

    print("Starting model parameter updates (using predicted u)...")
    for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):
        print(f"  Processing layer {layer_idx}")
        layer_svd = svd_components.get(f"layer_{layer_idx}", {})
        if not layer_svd:
            print(f"  No SVD data for layer {layer_idx} - skipping")
            continue

        # ---------- Self-Attention ----------
        for (name1, param1), (name2, param2) in zip(
            layer1.self_attn.named_parameters(),
            layer2.self_attn.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            if not name1.endswith(".weight"):
                continue

            # keys in SVD dict
            key_U  = f"self_attn_{name1}_U"
            key_S  = f"self_attn_{name1}_S"
            key_Vt = f"self_attn_{name1}_Vt"

            # 预测 u 的 key（与你 earlier 的 first_u / r2/predict 代码保持一致）
            module_key = f"layer_{layer_idx}_self_attn_{name1}"

            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:
                U  = layer_svd[key_U ].to(device, dtype=target_dtype)
                S  = layer_svd[key_S ].to(device, dtype=target_dtype)
                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)

                # 归一化系数（与原逻辑一致）
                norm_factor = torch.norm(U @ torch.diag(S) @ Vt)

                # 取预测的 u（长度应与 U[:,0] 相同）
                u_pred = predicted_u_dict.get(module_key, None)
                if u_pred is None:
                    print(f"    ⏩ Pred u missing for {module_key} - skipping this param")
                    continue

                u_pred = u_pred.to(device, dtype=target_dtype).view(-1, 1)   # (out_dim, 1)
                # if u_pred.shape[0] != U.shape[0]:
                #     print(f"    ⚠️ Shape mismatch for {module_key}: pred {u_pred.shape[0]} vs U {U.shape[0]} - skipping")
                #     continue

                # 用预测的 u 与 Vt 第一行做外积，并按 full ΔW 的 Frobenius 范数缩放
                # if torch.norm(u_pred) >0.03:
                #     u_pred = (u_pred/torch.norm(u_pred))*0.03
                print(torch.norm(u_pred))
                update = 1 * (u_pred @ Vt[0:1, :])                 # (out_dim, in_dim)
                param1.data += update
                print(f"    ✅ Updated self_attn.{name1} using predicted u")

        # ---------- MLP ----------
        for (name1, param1), (name2, param2) in zip(
            layer1.mlp.named_parameters(),
            layer2.mlp.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            if not name1.endswith(".weight"):
                continue

            key_U  = f"mlp_{name1}_U"
            key_S  = f"mlp_{name1}_S"
            key_Vt = f"mlp_{name1}_Vt"
            module_key = f"layer_{layer_idx}_mlp_{name1}"

            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:
                U  = layer_svd[key_U ].to(device, dtype=target_dtype)
                S  = layer_svd[key_S ].to(device, dtype=target_dtype)
                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)

                norm_factor = torch.norm(U @ torch.diag(S) @ Vt)

                u_pred = predicted_u_dict.get(module_key, None)
                if u_pred is None:
                    print(f"    ⏩ Pred u missing for {module_key} - skipping this param")
                    continue

                u_pred = u_pred.to(device, dtype=target_dtype).view(-1, 1)
                # if u_pred.shape[0] != U.shape[0]:
                #     print(f"    ⚠️ Shape mismatch for {module_key}: pred {u_pred.shape[0]} vs U {U.shape[0]} - skipping")
                #     continue
                # if torch.norm(u_pred) >0.03:
                #     u_pred = (u_pred/torch.norm(u_pred))*0.03
                print(torch.norm(u_pred))
                update = 1* (u_pred @ Vt[0:1, :])                 # (out_dim, in_dim)
                param1.data += update
                print(f"    ✅ Updated mlp.{name1} using predicted u")

    print(f"Saving model to {output_dir} with dtype {target_dtype}")
    model1 = model1.to("cpu")
    model1.save_pretrained(output_dir, torch_dtype=target_dtype, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # 清理
    torch.cuda.empty_cache()
    del model1, model2, svd_components, predicted_u_dict
    torch.cuda.empty_cache()
    print("All processes completed.")


# ==================== 配置与调用 ====================
if __name__ == "__main__":
    for i in range(10,12):
        model1_path = ""
        model2_path = f""
        svd_components_base_path = f""


        predicted_u_file = f""
        reconstruct_and_save_rank1_with_pred_u(
            model1_path=model1_path,
            model2_path=model2_path,
            svd_components_base_path=svd_components_base_path,
            predicted_u_file=predicted_u_file,
            output_subdir=f""   # 输出到 PRIME/rank1_predu
        )
