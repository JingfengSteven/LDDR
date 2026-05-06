#!/usr/bin/env python3
"""
从 CLIP 缓存中统计 relevance std 分布
用于指导 adaptive alpha-beta 策略设计
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict


def load_cache_files(cache_dir: str = "/path/to/your/workspace/cache_center/clip-cache"):
    """
    加载所有缓存文件，提取 relevance 特征

    Returns:
        stats: {
            'stds': list of float,
            'means': list of float,
            'maxs': list of float,
            'mins': list of float,
            'num_frames': list of int,
        }
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    stats = {
        'stds': [],
        'means': [],
        'maxs': [],
        'mins': [],
        'num_frames': [],
    }

    cache_files = list(cache_path.glob("*.pt"))
    print(f"Found {len(cache_files)} cache files")

    for i, cache_file in enumerate(cache_files):
        try:
            data = torch.load(cache_file, map_location='cpu')
            full_feats = data['full_feats']      # [T, D]
            text_embed = data['text_embed']       # [D]

            # 计算 relevance
            relevance = (full_feats @ text_embed).cpu().numpy()

            # 归一化到 [0,1]
            rel_min = relevance.min()
            rel_max = relevance.max()
            relevance = (relevance - rel_min) / (rel_max - rel_min + 1e-6)

            # 统计
            stats['stds'].append(relevance.std())
            stats['means'].append(relevance.mean())
            stats['maxs'].append(relevance.max())
            stats['mins'].append(relevance.min())
            stats['num_frames'].append(len(relevance))

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(cache_files)} files")

        except Exception as e:
            print(f"Error loading {cache_file}: {e}")

    return stats


def compute_statistics(stats: dict) -> dict:
    """计算详细的统计信息"""
    stds = np.array(stats['stds'])

    result = {
        'count': len(stds),
        'std': {
            'mean': float(stds.mean()),
            'std': float(stds.std()),
            'min': float(stds.min()),
            'max': float(stds.max()),
            'percentiles': {
                'p1': float(np.percentile(stds, 1)),
                        'p5': float(np.percentile(stds, 5)),
                        'p10': float(np.percentile(stds, 10)),
                        'p25': float(np.percentile(stds, 25)),
                        'p50': float(np.percentile(stds, 50)),
                        'p75': float(np.percentile(stds, 75)),
                        'p90': float(np.percentile(stds, 90)),
                        'p95': float(np.percentile(stds, 95)),
                        'p99': float(np.percentile(stds, 99)),
                    }
                },
        'num_frames': {
            'mean': float(np.array(stats['num_frames']).mean()),
            'min': int(np.array(stats['num_frames']).min()),
            'max': int(np.array(stats['num_frames']).max()),
        }
    }

    # 直方图数据（用于可视化）
    hist, bins = np.histogram(stds, bins=50, range=(0, 0.4))
    result['std']['histogram'] = {
        'counts': hist.tolist(),
        'bins': bins.tolist()
    }

    return result


def print_summary(stats_result: dict):
    """打印统计摘要"""
    print("\n" + "=" * 60)
    print("RELEVANCE STD DISTRIBUTION SUMMARY")
    print("=" * 60)

    s = stats_result['std']
    print(f"\nTotal samples: {stats_result['count']}")
    print(f"\nStd statistics:")
    print(f"  Mean:     {s['mean']:.4f}")
    print(f"  Std:      {s['std']:.4f}")
    print(f"  Range:    [{s['min']:.4f}, {s['max']:.4f}]")

    print(f"\nPercentiles:")
    p = s['percentiles']
    print(f"  1%:  {p['p1']:.4f}")
    print(f"  5%:  {p['p5']:.4f}")
    print(f"  10%: {p['p10']:.4f}")
    print(f"  25%: {p['p25']:.4f}")
    print(f"  50%: {p['p50']:.4f}")
    print(f"  75%: {p['p75']:.4f}")
    print(f"  90%: {p['p90']:.4f}")
    print(f"  95%: {p['p95']:.4f}")
    print(f"  99%: {p['p99']:.4f}")

    print(f"\nNum frames: {stats_result['num_frames']['min']} - {stats_result['num_frames']['max']} "
          f"(mean: {stats_result['num_frames']['mean']:.1f})")

    # 策略建议
    print("\n" + "-" * 60)
    print("STRATEGY RECOMMENDATIONS")
    print("-" * 60)

    p25, p50, p75 = p['p25'], p['p50'], p['p75']

    print(f"\n当前 simple 方法阈值: 0.3")
    print(f"建议分段:")
    print(f"  低方差区间:   std < {p25:.3f}  (25%)")
    print(f"  中等区间:      {p25:.3f} ≤ std < {p75:.3f}  (50%)")
    print(f"  高方差区间:   std ≥ {p75:.3f}  (25%)")

    print(f"\nSigmoid 参数建议:")
    center = p50
    scale = (p75 - p25) / 4  # 过渡宽度
    print(f"  center = {center:.4f}")
    print(f"  scale  = {scale:.4f}")

    print("\n" + "=" * 60)


def save_results(stats_result: dict, output_path: str):
    """保存统计结果到 JSON"""
    with open(output_path, 'w') as f:
        json.dump(stats_result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze relevance std distribution from CLIP cache")
    parser.add_argument("--cache_dir", type=str,
                       default="/path/to/your/workspace/cache_center/clip-cache",
                       help="Path to CLIP cache directory")
    parser.add_argument("--output", type=str,
                       default="./stats/std_distribution.json",
                       help="Output JSON file path")

    args = parser.parse_args()

    # 加载缓存
    stats = load_cache_files(args.cache_dir)

    # 计算统计
    stats_result = compute_statistics(stats)

    # 打印摘要
    print_summary(stats_result)

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(stats_result, args.output)

    print("\nNext steps:")
    print(f"1. 查看 JSON: cat {args.output}")
    print(f"2. 可视化: python scripts/visualize_std.py --input {args.output}")
