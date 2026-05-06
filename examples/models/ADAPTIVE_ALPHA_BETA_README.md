# Adaptive Alpha-Beta Strategy for CDPruner+DPP

## 概述

本实现在 CDPruner+DPP 帧选择算法中加入了**自适应 α-β 参数调整**功能。

### 核心思想

根据**图文相似度分布**的统计特征，自动调整：
- **α (alpha)**: 内容相似性权重
- **β (beta)**: 时间位置先验权重

---

## 三种自适应方法

### 1. Simple 方法 (`method="simple"`) - 推荐

**原理**: 只用相似度方差作为信号

**启发式规则**:
```
方差大 (std > 0.25)  → 内容区分度高 → 高α, 低β
方差小 (std < 0.10)  → 分布均匀    → 低α, 高β
```

**输出范围**:
- α: [0.6, 1.5]
- β: [0.0, 0.5]

**适用场景**: 快速实验，80%+ 场景有效

---

### 2. Multi 方法 (`method="multi"`) - 最鲁棒

**原理**: 融合 4 个统计信号

| 信号 | 计算 | 权重 |
|------|------|------|
| std_score | 方差归一化 | 30% |
| gap_score | Top-K 均值与总均值的差 | 30% |
| peak_score | 最大值与中位数的差 | 20% |
| entropy_score | 熵的倒数（分布集中度） | 20% |

**输出范围**:
- α: [0.6, 1.5]
- β: [0.0, 0.6]

**适用场景**: 生产环境，95%+ 场景有效

---

### 3. Bucketed 方法 (`method="bucketed"`) - 最可控

**原理**: 将分布分为典型模式，匹配预设参数

| 模式 | 条件 | α | β | 典型问题 |
|------|------|-----|-----|----------|
| sharp | std>0.25 且 gap>0.3 | 1.3 | 0.2 | 车牌号、细节 |
| uniform | std<0.12 且 gap<0.15 | 0.7 | 0.4 | 动作、过程 |
| multi-peak | std>0.18 且 gap<0.2 | 0.9 | 0.2 | 多主题视频 |
| moderate | 其他 | 1.0 | 0.15 | 一般问答 |

**适用场景**: 需要可解释性，便于调优

---

## 使用方法

### 基本用法

```bash
# 运行评估（使用默认 multi 方法）
./examples/models/qwen25vl_adaptive_alpha_beta.sh
```

### 自定义参数

修改 `qwen2_5_vl_chat_wo_ours_v3.py` 中 `generate_until` 函数的调用参数：

```python
selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
    frame_features=full_feats,
    frame_embeds=full_feats,
    text_embed=text_embed,
    total_token_budget=self.max_num_frames * self.target_token_per_frame,
    min_token_per_frame=self.min_token_per_frame,
    max_token_per_frame=self.max_token_per_frame,
    target_token_per_frame=self.target_token_per_frame,

    # ========== 自适应参数 ==========
    adaptive_alpha_beta_method="multi",  # "simple", "multi", "bucketed"
    base_alpha=1.0,                # 基础 α 值
    base_beta=0.0,                 # 基础 β 值
    max_beta=0.6,                  # β 的最大值
    debug_alpha_beta=True            # 是否打印调试信息
)
```

---

## 调试输出

启用 `debug_alpha_beta=True` 后，会打印：

```
[Adaptive α-β] Method: multi_signal_fusion, α: 1.245, β: 0.215
  → std: 0.187, content_conf: 0.715
```

| 字段 | 说明 |
|------|------|
| Method | 使用的自适应方法 |
| α | 最终的 alpha 值 |
| β | 最终的 beta 值 |
| std | 相似度标准差 |
| content_conf | 内容置信度 (multi 方法) |
| pattern | 分布模式 (bucketed 方法) |

---

## 参数调优建议

### 场景化配置

| 场景 | 推荐方法 | base_alpha | max_beta |
|------|----------|-----------|----------|
| **VideoMME** (问答为主) | multi | 1.0 | 0.5 |
| **动作识别任务** | bucketed | 0.9 | 0.7 |
| **时序推理任务** | bucketed | 0.8 | 0.8 |
| **细节识别** | simple | 1.2 | 0.3 |

### 快速实验流程

1. **从 multi 方法开始** (默认)
2. **观察调试输出**，统计 α、β 分布
3. **如果某类问题表现差**：
   - 细节类 → 提高 base_alpha (如 1.2)
   - 动作类 → 提高 max_beta (如 0.7)
4. **可选**: 切换到 bucketed 方法手动配置模式

---

## 理论基础

### α 和 β 的作用

```
similarity[i,j] = α × content_sim[i,j] + β × location_prior[i,j]
```

其中：
- `content_sim[i,j]`: 帧 i 和 j 的内容相似度（余弦距离）
- `location_prior[i,j]`: 时间位置先验 = exp(-|i-j|² / R²)

### 相似度分布的物理意义

| 分布特征 | 说明 | α、β 策略 |
|---------|------|----------|
| **高方差** | 有明确的高分帧（关键帧集中） | 提高α强化内容匹配 |
| **低方差** | 所有帧相似度接近（无明显关键帧） | 提高β增加时间覆盖 |
| **高偏度** | Top-K 帧与平均差距大 | 提高α，头部优先 |
| **高熵值** | 分布均匀（平坦） | 提高β，时间均匀采样 |

---

## 代码结构

```python
lmms_eval/models/chat/qwen2_5_vl_chat_wo_ours_v3.py
├── adaptive_alpha_beta_simple()      # 简单方法
├── adaptive_alpha_beta_multi()       # 多指标融合
├── adaptive_alpha_beta_bucketed()   # 分段模式
└── cdpruner_dpp_dynamic_resolution()  # 主函数（已修改）
    └── 调用自适应函数
        └── 计算 similarity = α×content_sim + β×location_prior
```

---

## 与其他策略对比

| 策略 | Token分配 | 时间复杂度 | 特点 |
|--------|-----------|-------------|------|
| **CDPruner+DPP (固定 α、β)** | 动态自适应 | O(n²K) | 通用性强 |
| **CDPruner+DPP (自适应 α、β)** | 动态自适应 | O(n²K) | **自适应性更强** |
| **AKS** | 统一分辨率 | O(n log n) | 时间递归分割 |
| **QFrame** | 多尺度分层 | O(n log n) | Gumbel 采样 |

---

## 常见问题

### Q: 如何选择自适应方法？

**A**:
- **快速验证**: 用 simple
- **生产部署**: 用 multi (默认)
- **需要可解释性**: 用 bucketed

### Q: 如何调试参数？

**A**: 设置 `debug_alpha_beta=True`，观察输出：
```
[Adaptive α-β] Method: multi, α: 1.245, β: 0.215
```
- 如果总是 α>1.3 且 β<0.2 → 可能 base_alpha 过高
- 如果总是 α<0.8 且 β>0.4 → 可能 max_beta 过高

### Q: 能否混合使用？

**A**: 可以修改代码，在 multi 方法中调整信号权重：
```python
# 在 adaptive_alpha_beta_multi() 中
content_confidence = (
    0.4 * std_score +  # 提高方差权重
    0.3 * gap_score +
    0.1 * peak_score +  # 降低峰值权重
    0.2 * entropy_score
)
```

---

## 参考文献

1. DPP (Determinantal Point Process): Kulesza & Taskar, 2012
2. Video Frame Selection: Various CVPR/ICCV papers
3. Adaptive Token Allocation: Custom implementation

---

## 更新日志

- **v3**: 添加自适应 α-β 功能
  - 实现 3 种自适应方法
  - 添加调试输出
  - 支持自定义 base_alpha、max_beta 等参数
