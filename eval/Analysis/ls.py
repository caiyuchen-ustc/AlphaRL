import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from functools import reduce

# ---- 全局绘图风格 (保持与 PCA 一致) ----
plt.rcParams.update({
    "pdf.fonttype": 42,   # 避免 Type 3 字体
    "ps.fonttype": 42,
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    # 3) 设置字体族为 Arival
    # 如果系统中已经安装 Arival，直接指定字体族即可

    "font.family": "Arival"
})

plt.style.use("seaborn-v0_8")  # 统一风格

def load_dicts(base_path, start=1, end=27):
    """加载所有 checkpoint 的模块向量字典"""
    all_dicts = []
    for i in range(start, end + 1):
        file_path = os.path.join(base_path, f"global_step_{i}", "first_u_vectors.pt")
        if not os.path.exists(file_path):
            continue
        data = torch.load(file_path, map_location="cpu")
        all_dicts.append((i, data))
    return all_dicts


def get_common_keys(all_dicts):
    """获取所有 checkpoint 都包含的模块 key"""
    keys_list = [set(d.keys()) for _, d in all_dicts]
    return sorted(list(reduce(lambda a, b: a & b, keys_list)))

def rescale_accuracies(acc):
    """放缩 accuracy：<=0.2 保持不变，>0.2 归一化到 0.2~0.9"""
    acc = np.array(acc)
    mask = acc > 0.2
    if mask.sum() > 0:
        acc_high = acc[mask]
        min_val, max_val = acc_high.min(), acc_high.max()
        if max_val > min_val:  # 避免除零
            acc[mask] = 0.2 + 0.7 * (acc_high - min_val) / (max_val - min_val)
        else:
            acc[mask] = 0.9
    return acc


def pls_regression_visualize(vectors, accuracies_raw, accuracies_scaled, key,
                             save_dir="pls_plots", r2_threshold=0.7):
    """对单个模块做 PLS 分析并可视化 (绘图用原始, R² 用缩放)"""
    os.makedirs(save_dir, exist_ok=True)

    # ---------- X 做归一化 + log ----------
    X_min = vectors.min(axis=0)
    X_max = vectors.max(axis=0)
    X_norm = (vectors - X_min) / (X_max - X_min + 1e-8)
    X_scaled = np.log(X_norm + 1e-6)

    # PLS 回归，只取一个成分
    pls = PLSRegression(n_components=1)
    projected = pls.fit_transform(X_scaled, accuracies_scaled.reshape(-1, 1))[0]

    # comp1 与缩放后的 Accuracy 拟合
    reg = LinearRegression().fit(projected, accuracies_scaled)
    r2 = r2_score(accuracies_scaled, reg.predict(projected))

    if r2 < r2_threshold:
        return None

        # ---------------- 可视化 (绘图用原始 accuracies_raw) ---------------- #
    # ---------- 可视化 (绘图用原始 accuracies_raw) ---------------- #
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 对 x 轴进行开根号再归一化
    x_proj = projected[:, 0]
    x_sqrt = (x_proj - x_proj.min()) # 避免负值开根号
    x_norm = (x_sqrt - x_sqrt.min()) / (x_sqrt.max() - x_sqrt.min() + 1e-8)

    sc = ax.scatter(
        x_norm, accuracies_raw,
        c=accuracies_raw, cmap="viridis",
        s=55, marker="D", alpha=0.85
    )

    # 拟合直线也用相同映射
    x_line = np.linspace(x_proj.min(), x_proj.max(), 100)
    x_line_sqrt = x_line
    x_line_norms = (x_line_sqrt ) / (x_line_sqrt.max()  + 1e-8)
    
    x_line_norm = (x_line_norms - x_line_norms.min()) / (x_line_norms.max() - x_line_norms.min() + 1e-8)
    
    ax.plot(x_line_norm, reg.predict(x_line.reshape(-1, 1)), "r--", lw=2)

    cbar = fig.colorbar(sc, ax=ax, label="         ")
    cbar.ax.tick_params(labelsize=22)

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.set_ticks([])
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{key.replace('.', '_')}.svg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {save_path}, R²={r2:.3f}")
    
    return r2


def run_pls_all_modules(base_path, y, start=1, end=27,
                        save_dir="pls_plots", r2_threshold=0.7,
                        r2_savefile="r2_results.json", top_k=10):
    """对所有模块运行 PLS 回归，保存缩放后的 R² 并输出 Top-K"""
    all_dicts = load_dicts(base_path, start=start, end=end)
    common_keys = get_common_keys(all_dicts)
    print(f"🔍 Found {len(common_keys)} common keys")

    # ---- 放缩 accuracies ----
    y_scaled =y
    #y_scaled = rescale_accuracies(y)

    r2_results = {}

    for key in common_keys:
        vectors = [d[key].numpy() for step, d in all_dicts if step <= len(y)]
        if len(vectors) < 3:
            continue

        r2 = pls_regression_visualize(
            np.array(vectors), y[:len(vectors)], y_scaled[:len(vectors)], key,
            save_dir=save_dir, r2_threshold=r2_threshold
        )
        if r2 is not None:
            r2_results[key] = float(r2)

    # ---- 保存 JSON (只保存缩放的) ----
    with open(r2_savefile, "w") as f:
        json.dump(r2_results, f, indent=2)
    print(f"📑 R² results saved to {r2_savefile} (scaled accuracies)")

    # 打印 Top-K
    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔥 Top {top_k} modules by R²:")
    for i, (key, r2) in enumerate(sorted_r2[:top_k], 1):
        print(f"{i:2d}. {key:<55} R² = {r2:.3f}")

    return r2_results



def save_r2_latex_table(r2_results, save_path="r2_table.tex", top_k=10):
    """把 Top-K R² 结果保存为 LaTeX 表格"""
    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    with open(save_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l c}\n")
        f.write("\\toprule\n")
        f.write("Module & $R^2$ \\\\\n")
        f.write("\\midrule\n")
        for key, r2 in sorted_r2:
            f.write(f"{key} & {r2:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Top-{top_k} modules ranked by $R^2$ from PLS regression.}}\n")
        f.write("\\label{tab:r2_topk}\n")
        f.write("\\end{table}\n")
    print(f"📑 LaTeX table saved to {save_path}")


if __name__ == "__main__":
    base_path = ""
    y = np.array([
    ])

    r2_results = run_pls_all_modules(
        base_path, y, start=1, end=27,
        save_dir="dapopls_plots", r2_threshold=0,
        r2_savefile=os.path.join(base_path, "global_step_27", "r2_results.json"),
        top_k=10
    )

    save_r2_latex_table(
        r2_results,
        save_path=os.path.join(base_path, "global_step_27", "r2_table.tex"),
        top_k=10
    )
