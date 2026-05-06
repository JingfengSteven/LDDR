# Relevance Std Distribution Analysis

从 CLIP 缓存中统计图文相似度的方差分布，用于指导 adaptive alpha-beta 策略设计。

## 快速使用

```bash
# 一键运行
./scripts/analyze_std_cache.sh
```

## 输出

1. **JSON 统计**: `stats/std_distribution.json`
   ```json
   {
     "count": 168,
     "std": {
       "mean": 0.1873,
       "percentiles": {
         "p25": 0.12,
         "p50": 0.18,
         "p75": 0.24,
         ...
       }
     }
   }
   ```

2. **可视化图表**: `stats/std_distribution.png`
   - 直方图
   - CDF
   - 百分位数
   - 当前方法 vs 数据分布对比

## 使用统计结果优化策略

### 方式 1: 校准阈值

```python
# 原来 (硬编码)
normalized_std = min(std / 0.3, 1.0)

# 改为 (基于数据)
std_min = percentiles['p25']  # 0.12
std_max = percentiles['p75']  # 0.24
normalized_std = (std - std_min) / (std_max - std_min + eps)
```

### 方式 2: 分段策略

```python
if std < p25:
    # 低方差：均匀分布，需要时间覆盖
    alpha = 0.6
    beta = 0.5
elif std > p75:
    # 高方差：有明显关键帧
    alpha = 1.5
    beta = 0.0
else:
    # 中等：线性过渡
    ratio = (std - p25) / (p75 - p25)
    alpha = 0.6 + 0.9 * ratio
    beta = 0.5 * (1 - ratio)
```

### 方式 3: Sigmoid 平滑过渡

```python
def continuous_beta_from_std(std, center=0.18, scale=0.03, max_beta=0.5):
    return max_beta / (1 + np.exp((std - center) / scale))

# center = P50 (中位数)
# scale = (P75 - P25) / 4 (过渡宽度)
```

## 手动运行

```bash
# 只统计
python3 scripts/analyze_std_from_cache.py \
    --cache_dir /path/to/your/workspace/cache_center/clip-cache \
    --output ./stats/std_distribution.json

# 只可视化
python3 scripts/visualize_std.py \
    --input ./stats/std_distribution.json \
    --output ./stats/std_distribution.png
```

## 注意事项

1. **确保缓存存在**: 先运行一次评估生成缓存
2. **缓存路径**: 默认 `/path/to/your/workspace/cache_center/clip-cache`
3. **不同数据集**: 每个数据集可能有不同的分布，分别统计
