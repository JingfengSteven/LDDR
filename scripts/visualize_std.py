#!/usr/bin/env python3
"""
可视化 relevance std 分布
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_std_distribution(json_path: str, output_path: str = None):
    """绘制 std 分布图"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    std_data = data['std']
    bins = np.array(std_data['histogram']['bins'])
    counts = np.array(std_data['histogram']['counts'])

    # 创建图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Relevance Std Distribution (N={data["count"]})', fontsize=16)

    # 1. 直方图
    ax = axes[0, 0]
    ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
    ax.axvline(std_data['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {std_data["mean"]:.3f}')
    ax.axvline(std_data['percentiles']['p25'], color='orange', linestyle=':', label='P25')
    ax.axvline(std_data['percentiles']['p75'], color='orange', linestyle=':', label='P75')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Count')
    ax.set_title('Histogram')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. CDF
    ax = axes[0, 1]
    cumsum = np.cumsum(counts)
    cumsum = cumsum / cumsum[-1]
    ax.plot(bins[:-1], cumsum, linewidth=2, color='blue')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(std_data['percentiles']['p50'], color='red', linestyle='--', label=f'P50: {std_data["percentiles"]["p50"]:.3f}')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Box plot + percentiles table
    ax = axes[1, 0]
    p = std_data['percentiles']
    percentiles = ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
    values = [p[k] for k in percentiles]

    bars = ax.barh(percentiles, values, color='steelblue', alpha=0.7)
    ax.axvline(std_data['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {std_data["mean"]:.3f}')
    ax.set_xlabel('Std Value')
    ax.set_title('Percentiles')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

    # 4. 策略分区可视化
    ax = axes[1, 1]
    p25, p50, p75 = p['p25'], p['p50'], p['p75']

    # 绘制当前 simple 方法的线性映射
    x = np.linspace(0, 0.35, 100)
    y_simple_beta = 0.5 * (1 - np.clip(x / 0.3, 0, 1))  # max_beta=0.5, threshold=0.3
    y_simple_alpha = 1.0 * (0.6 + 0.9 * np.clip(x / 0.3, 0, 1))  # base_alpha=1.0

    ax.plot(x, y_simple_beta, 'r-', linewidth=2, label='β (simple)')
    ax.plot(x, y_simple_alpha, 'b-', linewidth=2, label='α (simple)')

    # 标记数据集的分布区间
    ax.axvspan(0, p25, alpha=0.1, color='green', label=f'Low std (25%)')
    ax.axvspan(p25, p75, alpha=0.1, color='yellow')
    ax.axvspan(p75, 0.35, alpha=0.1, color='red', label=f'High std (25%)')

    ax.axvline(p25, color='green', linestyle=':', alpha=0.5)
    ax.axvline(p50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(p75, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Std')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Current Simple Method vs Data Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 2.0)

    plt.tight_layout()

    if output_path is None:
        output_path = json_path.replace('.json', '.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize std distribution")
    parser.add_argument("--input", type=str, required=True, help="Path to std_distribution.json")
    parser.add_argument("--output", type=str, default=None, help="Output image path")

    args = parser.parse_args()
    plot_std_distribution(args.input, args.output)
