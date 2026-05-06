import time
from typing import List, Optional, Tuple, Union
import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
import heapq
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
import torch.nn.functional as F
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages
from lmms_eval.metadata_manager import metadata_manager
import cv2
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via pip install qwen-vl-utils")


import sys
sys.path.append("/path/to/your/workspace/EACL/lmms-eval/lmms_eval/models/simple/Longclip")
from model import longclip

USE_AKS = False
USE_Q_FRAME = False
TEST_CDP_MIN_ALLOCATION = False
TEST_MIN_TOKEN_MAX_COVERAGE = False
USE_FOCUS = False

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import SiglipModel, SiglipProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from tqdm import tqdm
import argparse
import re
from PIL import Image
import torch
import numpy as np

def focus_sampler(
    full_feats: torch.Tensor,
    text_embed: torch.Tensor,
    fps: float,
    max_num_frames: int,
    target_token_per_frame: int = 1024,
    coarse_every_sec: float = 16.0,
    fine_every_sec: float = 1.0,
    zoom_ratio: float = 0.25,
    min_coarse_segments: int = 8,
    min_zoom_segments: int = 4,
    min_gap_sec: float = 1.0,
    top_ratio: float = 0.2,
    temperature: float = 0.06,
    exploration_weight: float = 1.5,
    disable_gap_below_sec: float = 0.2,
    gap_ratio_of_avg: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del fine_every_sec
    total_frames = full_feats.shape[0]
    video_duration = total_frames / max(1.0, fps)
    scores = (full_feats @ text_embed).squeeze().cpu().numpy()

    avg_spacing_sec = video_duration / max(1, max_num_frames)
    if avg_spacing_sec <= disable_gap_below_sec:
        auto_min_gap_sec = 0.0
    else:
        auto_min_gap_sec = min(gap_ratio_of_avg * avg_spacing_sec, min_gap_sec)
    min_gap_frames = int(auto_min_gap_sec * fps)

    actual_coarse_step = coarse_every_sec
    if video_duration / coarse_every_sec < min_coarse_segments:
        actual_coarse_step = max(1.0 / max(fps, 1e-6), video_duration / max(1, min_coarse_segments))
    step = max(1, int(actual_coarse_step * fps))
    coarse_indices = np.arange(0, total_frames, step)

    half_win = int(actual_coarse_step * fps // 2)
    arms = []
    for idx in coarse_indices:
        start, end = max(0, idx - half_win), min(total_frames, idx + half_win)
        arm_scores = scores[start:end]
        mu = np.mean(arm_scores)
        sigma = np.std(arm_scores) + 1e-6
        ucb = mu + exploration_weight * sigma
        arms.append({"range": (start, end), "ucb": ucb, "scores": arm_scores})

    num_zoom = max(int(len(arms) * zoom_ratio), min_zoom_segments)
    num_zoom = min(num_zoom, len(arms))
    arms.sort(key=lambda x: x["ucb"], reverse=True)
    selected_arms = arms[:num_zoom]

    num_top = int(max_num_frames * top_ratio)
    num_ucb_budget = max_num_frames - num_top
    ucb_vals = np.array([a["ucb"] for a in selected_arms])
    weights = ucb_vals / (ucb_vals.sum() + 1e-9)
    arm_budgets = np.round(weights * num_ucb_budget).astype(int)

    final_candidates = []
    for i, arm in enumerate(selected_arms):
        budget = arm_budgets[i]
        if budget <= 0:
            continue
        probs = np.exp((arm["scores"] - np.max(arm["scores"])) / temperature)
        probs /= probs.sum()
        local_indices = np.random.choice(len(arm["scores"]), size=min(budget, len(arm["scores"])), replace=False, p=probs)
        final_candidates.extend(np.arange(arm["range"][0], arm["range"][1])[local_indices])

    if num_top > 0:
        global_top_idx = np.argsort(scores)[-num_top * 2 :]
        final_candidates.extend(global_top_idx)

    final_candidates = sorted(set(final_candidates), key=lambda x: scores[x], reverse=True)
    selected_indices = []
    for candidate_idx in final_candidates:
        if len(selected_indices) >= max_num_frames:
            break
        if all(abs(candidate_idx - selected_idx) >= min_gap_frames for selected_idx in selected_indices):
            selected_indices.append(candidate_idx)

    final_indices = sorted(selected_indices)
    selected_resolution = [int(target_token_per_frame)] * len(final_indices)
    return final_indices, selected_resolution
'''
def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=1024,
    max_token_per_frame=4096,
    target_token_per_frame=512
):
    # 保存原始 device
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # ===== 把计算搬到 CPU =====
    frame_features = frame_features.cpu()
    frame_embeds = frame_embeds.cpu()
    text_embed = text_embed.cpu()

    # 算力预算
    K_max = total_token_budget // min_token_per_frame

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)
    

    # 3. 构建 DPP Kernel
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    similarity = (feat @ feat.T).clamp(min=0)

    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)
    k = min(K_max, T)
    _, topk_indices = torch.topk(relevance, k=k, sorted=False)

    # 4. 保持时间顺序输出 (这是为了后续视频处理的连续性)
    final_indices = topk_indices.sort()[0]

    # 5. 分配 Token 预算 (此处保持你原始逻辑的 min_token)
    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    return final_indices.to(original_device), selected_token_counts.to(original_device)
  
    kernel = relevance[:, None] * similarity * relevance[None, :]

    # 4. DPP greedy
    cis = torch.zeros((K_max, T))
    di2s = torch.diag(kernel).clone()
    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break

        selected_indices.append(j.item())

        if i == 0:
            eis = kernel[j] / torch.sqrt(current_energy + eps)
        else:
            eis = (kernel[j] - cis[:i, j] @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis
        di2s = di2s - eis ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]
        selected_token_counts = selected_token_counts[sort_idx]


    return final_indices.to(original_device), selected_token_counts.to(original_device)



'''
def cdpruner_dpp_chunk_ablation(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    chunk_cnt=16,  # 消融实验参数: 2 或 4
    min_token_per_frame=1024,
    max_token_per_frame=4096,
    target_token_per_frame=512
):
    import torch
    import math

    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # 数据预处理 (CPU/Float32 保证数值稳定性)
    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    # 计算全局总共能选多少帧，以及每个 Chunk 的帧数预算
    total_k_max = total_token_budget // min_token_per_frame
    k_per_chunk = total_k_max // chunk_cnt
    
    if k_per_chunk < 1:
        k_per_chunk = 1 # 确保每个 chunk 至少选一帧

    # 特征归一化
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()
    
    # 归一化相关性
    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)
    Phi_all = relevance[:, None] * feat

    all_selected_indices = []

    # ===== 开始分块处理 =====
    chunk_size = math.ceil(T / chunk_cnt)
    
    for c in range(chunk_cnt):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, T)
        if start >= T: break
        
        # 提取当前 Chunk 的特征
        Phi = Phi_all[start:end]
        curr_chunk_len = Phi.shape[0]
        d = Phi.shape[1]
        
        # 局部 DPP 变量
        actual_k = min(k_per_chunk, curr_chunk_len)
        cis = torch.zeros((actual_k, d))
        di2s = (Phi * Phi).sum(dim=1)
        
        chunk_selected = []
        
        for i in range(actual_k):
            j = torch.argmax(di2s)
            current_energy = di2s[j]
            
            if current_energy <= 1e-7:
                break
                
            # 保存相对于当前 chunk 的索引，稍后转换
            chunk_selected.append(j.item() + start)
            
            phi_j = Phi[j]
            if i == 0:
                eis = phi_j / torch.sqrt(current_energy + eps)
            else:
                proj = cis[:i] @ phi_j
                eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)
            
            cis[i] = eis
            di2s = di2s - (Phi @ eis) ** 2
            di2s = torch.clamp(di2s, min=0)
            di2s[j] = -float("inf")
            
        all_selected_indices.extend(chunk_selected)

    # ===== 合并并排序 =====
    final_indices = torch.tensor(all_selected_indices).long()
    final_indices, _ = torch.sort(final_indices)

    # Token 分配 (保持与你原始代码一致)
    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    print(f"Chunk Ablation (cnt={chunk_cnt}): selected {len(final_indices)} frames")
    
    return final_indices.to(original_device), selected_token_counts.to(original_device)
'''def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=1024,
    max_token_per_frame=4096,
    target_token_per_frame=512
):
    import torch
    import math

    # 保存原始 device
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # ===== 保持 CPU =====
    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    # 算力预算
    K_max = total_token_budget // min_token_per_frame

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)


    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)


    Phi = relevance[:, None] * feat
  
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)

    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break
        

        selected_indices.append(j.item())
        
        phi_j = Phi[j]

        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis


        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== token allocation =====
    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]
        selected_token_counts = selected_token_counts[sort_idx]
    print(final_indices)
    
    return final_indices.to(original_device), selected_token_counts.to(original_device)

'''
def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=256,
    max_token_per_frame=1024,
    target_token_per_frame=512
):
    import torch
    import math

    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    K_max = total_token_budget//(min_token_per_frame*2)

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    # ===== normalize =====
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)

    # ===== relevance =====
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)

    # ===== low-rank Phi =====
    Phi = relevance[:, None] * feat
  
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)

    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break
        
        selected_indices.append(j.item())
        
        phi_j = Phi[j]

        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis

        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== 排序 =====
    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]


    if len(final_indices) > 0:
        Phi_S = Phi[final_indices]
        k = Phi_S.shape[0]

        # ===== Gram =====
        G = Phi_S @ Phi_S.T
        G = G + eps * torch.eye(k)

        sign, logdet_full = torch.slogdet(G)

        importance = []

        for i in range(k):
            mask = torch.ones(k, dtype=torch.bool)
            mask[i] = False

            Phi_sub = Phi_S[mask]
            G_sub = Phi_sub @ Phi_sub.T
            G_sub = G_sub + eps * torch.eye(k - 1)

            sign_sub, logdet_sub = torch.slogdet(G_sub)

            importance_i = torch.exp(logdet_full - logdet_sub)
            importance.append(importance_i)

        importance = torch.stack(importance)

        # ===== density =====
        density = (Phi_S * Phi_S).sum(dim=1)
        density = density / (density.mean() + eps)

        importance = importance * ((density))

        # normalize
        importance = importance / (importance.sum() + eps)

        print("importance (with density):", importance)


        # 先排序（很关键）
        sorted_imp, sorted_idx = torch.sort(importance, descending=True)

        left, right = 1, len(sorted_imp)
        best_k = 1

        while left <= right:
            mid = (left + right) // 2

            imp_sub = sorted_imp[:mid]
            imp_sub = imp_sub / (imp_sub.sum() + eps)

            tokens = total_token_budget * imp_sub
            tokens = torch.clamp(tokens, min=min_token_per_frame, max=max_token_per_frame)

            if tokens.sum() <= total_token_budget:
                best_k = mid
                left = mid + 1
            else:
                right = mid - 1

        # ===== 最终保留 =====
        keep_idx = sorted_idx[:best_k]

        final_indices = final_indices[keep_idx]
        importance = importance[keep_idx]

        # 重新 normalize
        importance = importance / (importance.sum() + eps)

        # ===== allocation（最终版）=====
        base_tokens = torch.full((best_k,), min_token_per_frame)
        remaining_budget = total_token_budget - best_k * min_token_per_frame

        extra_tokens = remaining_budget * importance

        tokens = base_tokens + extra_tokens
        tokens = torch.clamp(tokens, max=max_token_per_frame)

        selected_token_counts = tokens.int()
        if len(final_indices) > 0:
            time_sort_idx = torch.argsort(final_indices)
            final_indices = final_indices[time_sort_idx]
            importance = importance[time_sort_idx]
            selected_token_counts = selected_token_counts[time_sort_idx]

    else:
        selected_token_counts = torch.full(
            (len(final_indices),),
            min_token_per_frame
        ).int()
    print(selected_token_counts)
    return final_indices.to(original_device), selected_token_counts.to(original_device)



def cdpruner_ablation_dpp_then_relevance(
    frame_features,
    frame_embeds,
    text_embed,
    # 阶梯配置
    tiers = [
        (16, 512)#, # Top 2: 1024
       # (4, 512),  # Next 4: 512
       # (8, 256)  # Next 16: 256
    ]
):
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # 为了确保有足够的候选帧供后续 Relevance 排序
    total_requested_frames = sum(t[0] for t in tiers)
    K_max = max(total_requested_frames, min(32, T)) # 稍微多选一点候选帧

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    # ===== 1. DPP Kernel 筛选候选帧 (保持多样性) =====
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    
    # 原始相关性分数
    raw_relevance = (img @ txt).squeeze()
    norm_relevance = (raw_relevance - raw_relevance.min()) / (raw_relevance.max() - raw_relevance.min() + eps)
    
    Phi = norm_relevance[:, None] * feat
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)
    
    candidate_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        if di2s[j] <= 1e-7: break
        candidate_indices.append(j.item())
        phi_j = Phi[j]
        if i == 0:
            eis = phi_j / torch.sqrt(di2s[j] + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(di2s[j] + eps)
        cis[i] = eis
        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    # 转为 tensor
    candidate_indices = torch.tensor(candidate_indices)

    # ===== 2. 在候选帧中，仅根据 Relevance Score 进行排序 =====
    if len(candidate_indices) > 0:
        # 提取候选帧对应的相关性分数
        candidate_relevance = raw_relevance[candidate_indices]
        
        # 对候选帧按相关性降序排列
        _, sorted_ranking_idx = torch.sort(candidate_relevance, descending=True)
        
        assigned_indices = []
        assigned_tokens = []
        
        current_rank = 0
        for num_frames, token_val in tiers:
            # 确定当前梯队能从候选帧中拿多少
            end_rank = min(current_rank + num_frames, len(candidate_indices))
            
            if current_rank >= len(candidate_indices):
                break
                
            # 从 candidate_indices 中选出当前梯队的索引
            tier_idx = sorted_ranking_idx[current_rank:end_rank]
            assigned_indices.append(candidate_indices[tier_idx])
            assigned_tokens.append(torch.full((len(tier_idx),), token_val))
            
            current_rank = end_rank

        # 合并
        final_selected_indices = torch.cat(assigned_indices)
        final_selected_tokens = torch.cat(assigned_tokens).int()

        # 3. 按时间顺序重排
        time_sort_idx = torch.argsort(final_selected_indices)
        
        return (final_selected_indices[time_sort_idx].to(original_device), 
                final_selected_tokens[time_sort_idx].to(original_device))

    else:
        return (torch.tensor([0]).to(original_device), 
                torch.tensor([1024]).int().to(original_device))

def cdpruner_ablation_only(
    frame_features,
    frame_embeds,
    text_embed,
    # 按照你的要求：(帧数, token数) 的阶梯配置
    tiers = [
        (2, 1024), # Top 2: 1024
        (4, 512),  # Next 4: 512
        (16, 256)   # Next 8: 256
    ]
):
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # 为了能选出足够的帧（4+4+8=16），K_max 至少要 16
    total_requested_frames = sum(t[0] for t in tiers)
    K_max = max(total_requested_frames, min(16, T))

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    # ===== 1. DPP 候选帧选取 =====
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + eps)
    
    Phi = relevance[:, None] * feat
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)
    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        if di2s[j] <= 1e-7: break
        selected_indices.append(j.item())
        phi_j = Phi[j]
        if i == 0:
            eis = phi_j / torch.sqrt(di2s[j] + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(di2s[j] + eps)
        cis[i] = eis
        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== 2. 计算 Importance 评分 =====
    if len(final_indices) > 0:
        Phi_S = Phi[final_indices]
        k = Phi_S.shape[0]
        G = Phi_S @ Phi_S.T + eps * torch.eye(k)
        _, logdet_full = torch.slogdet(G)
        
        importance = []
        for i in range(k):
            mask = torch.ones(k, dtype=torch.bool)
            mask[i] = False
            G_sub = Phi_S[mask] @ Phi_S[mask].T + eps * torch.eye(k - 1)
            _, logdet_sub = torch.slogdet(G_sub)
            importance.append(torch.exp(logdet_full - logdet_sub))
        
        importance = torch.stack(importance)
        density = (Phi_S * Phi_S).sum(dim=1)
        importance = importance * (density / (density.mean() + eps))

        # ===== 3. 分级分配逻辑 (Tiered Allocation) =====
        # 按照 Importance 降序排列索引
        _, sorted_ranking_idx = torch.sort(importance, descending=True)
        
        assigned_indices = []
        assigned_tokens = []
        
        current_rank = 0
        for num_frames, token_val in tiers:
            # 确定当前梯队能拿多少帧 (考虑到视频可能不够长)
            end_rank = min(current_rank + num_frames, len(final_indices))
            
            # 取出当前梯队的原始帧索引
            tier_idx = sorted_ranking_idx[current_rank:end_rank]
            if len(tier_idx) == 0:
                break
                
            assigned_indices.append(final_indices[tier_idx])
            assigned_tokens.append(torch.full((len(tier_idx),), token_val))
            
            current_rank = end_rank
            if current_rank >= len(final_indices):
                break

        # 合并结果
        final_selected_indices = torch.cat(assigned_indices)
        final_selected_tokens = torch.cat(assigned_tokens).int()

        # 最后按时间顺序重排，保持视频帧序列正确
        time_sort_idx = torch.argsort(final_selected_indices)
        
        return (final_selected_indices[time_sort_idx].to(original_device), 
                final_selected_tokens[time_sort_idx].to(original_device))

    else:
        return (torch.tensor([0]).to(original_device), 
                torch.tensor([1024]).int().to(original_device))

def cdpruner_ablation_relevance_allocation(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=256,
    max_token_per_frame=1024,
    target_token_per_frame=512
):
    import torch
    import math

    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    K_max = total_token_budget // (min_token_per_frame * 2)

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    # ===== 1. 特征归一化 & 相关性计算 =====
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    
    # 原始的 relevance 分数 (未经过 min-max 缩放前，用于分配)
    raw_relevance = (img @ txt).squeeze()
    
    # 用于 DPP 选取的归一化相关性
    rel_min, rel_max = raw_relevance.min(), raw_relevance.max()
    norm_relevance = (raw_relevance - rel_min) / (rel_max - rel_min + eps)

    # ===== 2. DPP 候选帧选取 (保持原有逻辑) =====
    Phi = norm_relevance[:, None] * feat
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)
    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]
        if current_energy <= 1e-7:
            break
        selected_indices.append(j.item())
        phi_j = Phi[j]
        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)
        cis[i] = eis
        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== 3. Token 分配逻辑 (替换 Importance 为 Relevance) =====
    if len(final_indices) > 0:
        # 获取候选帧对应的相关性分数作为分配依据
        # 注意：这里使用 raw_relevance 确保分布差异被保留
        importance = raw_relevance[final_indices]
        
        # 简单平移确保 importance 为正数 (处理相关性可能为负的情况)
        if importance.min() < 0:
            importance = importance - importance.min() + eps
            
        # 归一化，使其和为 1
        importance = importance / (importance.sum() + eps)

        # 按照相关性分数排序进行二分查找
        sorted_imp, sorted_idx = torch.sort(importance, descending=True)

        left, right = 1, len(sorted_imp)
        best_k = 1

        while left <= right:
            mid = (left + right) // 2
            imp_sub = sorted_imp[:mid]
            imp_sub = imp_sub / (imp_sub.sum() + eps)

            tokens = total_token_budget * imp_sub
            tokens = torch.clamp(tokens, min=min_token_per_frame, max=max_token_per_frame)

            if tokens.sum() <= total_token_budget:
                best_k = mid
                left = mid + 1
            else:
                right = mid - 1

        # ===== 4. 最终保留与分配 =====
        keep_idx = sorted_idx[:best_k]
        final_indices = final_indices[keep_idx]
        importance = importance[keep_idx]

        # 重新归一化保留帧的权重
        importance = importance / (importance.sum() + eps)

        base_tokens = torch.full((best_k,), min_token_per_frame)
        remaining_budget = total_token_budget - (best_k * min_token_per_frame)

        # 使用 relevance 权重分配剩余 budget
        extra_tokens = remaining_budget * importance
        tokens = base_tokens + extra_tokens
        tokens = torch.clamp(tokens, max=max_token_per_frame)

        selected_token_counts = tokens.int()

        # 恢复时间顺序
        time_sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[time_sort_idx]
        selected_token_counts = selected_token_counts[time_sort_idx]

    else:
        selected_token_counts = torch.full(
            (len(final_indices),),
            min_token_per_frame
        ).int()

    return final_indices.to(original_device), selected_token_counts.to(original_device)

def estimate_hw_from_resolution(orig_h, orig_w, target_token_count, patch_size=14):
    aspect_ratio = orig_h / orig_w
    grid_w = max(1, int(round((target_token_count / aspect_ratio) ** 0.5)))
    grid_h = max(1, int(round(grid_w * aspect_ratio)))
    new_h = grid_h * patch_size
    new_w = grid_w * patch_size
    return new_h, new_w

def load_and_resize_images(frame_paths, resolutions, patch_size=14):
    images = []
    metadata = []
    for idx, (path, token_count) in enumerate(zip(frame_paths, resolutions)):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        new_h, new_w = estimate_hw_from_resolution(orig_h, orig_w, token_count, patch_size)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        images.append(img_resized)
        actual_patches = (new_h // 14) * (new_w // 14)

        metadata.append({
            "frame_idx": idx,
            "path": os.path.basename(path),
            "resolution": f"{new_w}x{new_h}",
            "patches": actual_patches,
            "token_count": token_count,
        })

    return images, metadata

import hashlib
import pickle


def get_cache_key(frame_paths, question, clip_type):
    """
    clip_type: 'openai_clip', 'long_clip', or 'siglip_clip'
    """
    combined_str = "".join(frame_paths) + question + clip_type
    return hashlib.md5(combined_str.encode()).hexdigest()

@register_model("qwen2_5_vl_chat_wo_ours_v4")
class Qwen2_5_VL_Chat_WO_Ours_v4(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(
        self,
        test_cdp_min_allocation: bool = False,
        test_min_token_max_coverage: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.test_cdp_min_allocation = test_cdp_min_allocation
        self.test_min_token_max_coverage = test_min_token_max_coverage

    def generate_until(self, requests: List[Instance]) -> List[str]:

        print(f"[INFO] Total dataset samples: {len(requests)}")
        res = []


        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # Create a mapping from ctx id to the original instance
        # This allows us to attach metadata back to the correct instances later
        # We use ctx (first element of args) as the key since chunks are grouped by args
        req_to_instance = {id(req.args[0]): req for req in requests}

        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:

            if len(chunk) == 1:
                ctx = [chunk[0][0]]
                doc_to_messages = [chunk[0][1]]
                all_gen_kwargs = [chunk[0][2]]
                doc_id = [chunk[0][3]]
                task = [chunk[0][4]]
                split = [chunk[0][5]]
            else:

                ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
                ctx = list(ctx)
                doc_to_messages = list(doc_to_messages)
                all_gen_kwargs = list(all_gen_kwargs)
                doc_id = list(doc_id)
                task = list(task)
                split = list(split)


            actual_docs = [self.task_dict[task[idx]][split[idx]][doc_id[idx]] for idx in range(len(doc_id))]

            chat_messages = [doc_to_messages[idx](actual_docs[idx]) for idx in range(len(doc_id))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]


            video_frame_paths = []
            for video in videos:
                frame_dir = video[:-4]
                video_frame_path = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
                video_frame_paths.append(video_frame_path)


            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames

            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]


            print("$$$" * 30)
            
            print(batched_messages)
            
            print("$$$" * 30)

        
            
            original_questions = []
            for msg in batched_messages:
                for content in msg[0]['content']:
                    if content['type'] == 'text':
                        if "videomme" in self.task_name:
                            text = content['text']
                            text = text.replace("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n","")
                            question = text.split("\nA.")[0]
                            original_questions.append(question)
                        elif "mlvu" in self.task_name:
                            text = content['text']
                            question = text.split("""\n(A) """)[0]
                            original_questions.append(question)
                        elif "longvideobench" in self.task_name:
                            text = content['text']
                            text = text.replace("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n","")
                            question = text.split("\nA.")[0]
                            original_questions.append(question)
                        elif "lvbench" in self.task_name:
                            text = content['text']
                            text = text.split("\n(A)")[0]
                            original_questions.append(text)
                            

            print("###" * 30)

            print(original_questions)

            print("###" * 30)

            full_feats_list = []
            text_embed_list = []



            cache_dir = "/path/to/your/workspace/cache_center/clip-cache"
            os.makedirs(cache_dir, exist_ok=True)
            clip_type = getattr(self, "frame_encoder_type", "long_clip" if self.use_longclip else "openai_clip")
            clip_type = "long_clip"
            full_feats_list = []
            text_embed_list = []

            for frame_paths, question in zip(video_frame_paths, original_questions):
                combined_str = "".join(frame_paths) + question + clip_type
                cache_key = hashlib.md5(combined_str.encode()).hexdigest()
                cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

                if os.path.exists(cache_path):
                    eval_logger.debug(f"Loading cached {clip_type} features: {cache_key}")
                    cached_data = torch.load(cache_path, map_location='cpu')
                    full_feats_list.append(cached_data['full_feats'])
                    text_embed_list.append(cached_data['text_embed'])
                    print("exist")
                    continue

                if clip_type == "openai_clip":
                    eval_logger.debug("Running OpenAI-CLIP encoding...")
                    with torch.no_grad():

                        full_feats = self.clip_encoder.encode(
                            frame_paths, batch_size=32, convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=False, device='cuda:0'
                        ).float().cpu()

                        text_embed = self.clip_encoder.encode(
                            [question], convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=False, device='cuda:0'
                        )[0].float().cpu()
                elif clip_type == "long_clip":
                   
                    eval_logger.debug("Running Long-CLIP encoding with batch processing (FP32)...")
                    batch_size_clip = 1024

                    with torch.no_grad():
                        text_tokens = longclip.tokenize([question]).to(self.model.device)
                        text_embed = self.long_clip_model.encode_text(text_tokens)[0]
                        text_embed = text_embed / (text_embed.norm() + 1e-6)
                        # 立即转回 CPU，释放 GPU 引用
                        text_embed = text_embed.detach().cpu().float()

                        # 清理文本中间变量
                        del text_tokens

                        # 2. 分批次处理图像特征 (Image Features)
                        full_feats_batches = []

                        for i in range(0, len(frame_paths), batch_size_clip):
                            batch_paths = frame_paths[i : i + batch_size_clip]
                            images_list = []

                            # CPU 端预处理
                            for path in batch_paths:
                                try:
                                    image = Image.open(path).convert("RGB")
                                    processed = self.long_clip_processor(image)
                                    images_list.append(processed)
                                except Exception as e:
                                    eval_logger.error(f"Error loading image {path}: {e}")
                                    continue

                            if not images_list:
                                continue

                            image_batch = torch.stack(images_list).to(self.model.device)

                            batch_feats = self.long_clip_model.encode_image(image_batch)
                            batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)

                            # 必须执行 .cpu()，否则 full_feats_batches 会一直占用 GPU 显存
                            full_feats_batches.append(batch_feats.detach().cpu().float())

                            del image_batch
                            del batch_feats

                        # 3. 合并所有特征块
                        if full_feats_batches:
                            full_feats = torch.cat(full_feats_batches, dim=0)
                        else:
                            full_feats = None
                elif clip_type == "siglip_clip":
                    eval_logger.debug("Running SigLIP encoding...")
                    self.siglip_model = self.siglip_model.to(self.model.device)
                    batch_size_siglip = 32

                    with torch.no_grad():
                        text_inputs = self.siglip_processor(
                            text=question,
                            padding="max_length",
                            return_tensors="pt",
                        ).to(self.siglip_model.device)

                        stride_num = (int(text_inputs["input_ids"].shape[-1]) + 63) // 64
                        stride = (text_inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

                        input_id_heads, input_id_tails = [], []
                        left, right = 0, text_inputs["input_ids"].shape[-1]
                        while left < right:
                            input_id_heads.append(text_inputs["input_ids"][:, left : left + stride])
                            left += stride
                            if left < right:
                                input_id_tails.append(text_inputs["input_ids"][:, right - stride : right])
                                right -= stride

                        text_input_ids = torch.cat(input_id_heads + input_id_tails[::-1])
                        text_outputs = self.siglip_model.get_text_features(input_ids=text_input_ids)
                        text_embed = text_outputs.mean(dim=0, keepdim=True)[0]
                        text_embed = text_embed / (text_embed.norm() + 1e-6)
                        text_embed = text_embed.detach().cpu().float()

                        full_feats_batches = []
                        for i in range(0, len(frame_paths), batch_size_siglip):
                            batch_paths = frame_paths[i : i + batch_size_siglip]
                            images_list = []
                            for path in batch_paths:
                                try:
                                    image = Image.open(path).convert("RGB")
                                    images_list.append(image)
                                except Exception as e:
                                    eval_logger.error(f"Error loading image {path}: {e}")
                                    continue

                            if not images_list:
                                continue

                            image_inputs = self.siglip_processor(
                                images=images_list,
                                return_tensors="pt",
                            ).to(self.siglip_model.device)
                            batch_feats = self.siglip_model.get_image_features(
                                pixel_values=image_inputs["pixel_values"]
                            )
                            batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)
                            full_feats_batches.append(batch_feats.detach().cpu().float())

                        full_feats = torch.cat(full_feats_batches, dim=0) if full_feats_batches else None
                else:
                    raise ValueError(f"Unsupported frame_encoder_type: {clip_type}")
                torch.save({
                    'full_feats': full_feats,
                    'text_embed': text_embed,
                    'clip_type': clip_type,   # 额外保存元数据，方便排查
                    'question': question      # 可选
                }, cache_path)
                full_feats_list.append(full_feats)
                text_embed_list.append(text_embed)

            physical_patch_limits = []
            for video_frame_path in video_frame_paths:
                with Image.open(video_frame_path[0]) as sample_img:
                    orig_w, orig_h = sample_img.size
                physical_patch_limit = (orig_h // 14) * (orig_w // 14)
                physical_patch_limits.append(physical_patch_limit)


            print("Video Frame Paths and Physical Patch Limits:")
            print(f"physical_patch_limits: {physical_patch_limits}")

            print("Start DPP pruning and dynamic token allocation...")



            final_daynamic_video_frame_paths = []
            all_frame_metadata = []


            for full_feats, text_embed, video_frame_path, physical_patch_limit in zip(full_feats_list, text_embed_list, video_frame_paths, physical_patch_limits):
                full_feats = full_feats.to(self.model.device)
                text_embed = text_embed.to(self.model.device)
                if USE_AKS==True:
                    with torch.no_grad():
                        raw_scores = (full_feats @ text_embed).cpu().numpy()

                    T_current = full_feats.shape[0]
                    D_feat = full_feats.shape[1]
                    log_dir = "/path/to/your/workspace/EACL/effi"
                    log_file = os.path.join(log_dir, "dpp_efficiency_log_old_8.jsonl")
                    start_pruning = time.perf_counter()
                    selected_images = aks(
                        scores=raw_scores,
                        frame_paths=video_frame_path,
                        max_num_frames=self.max_num_frames,
                        target_token_per_frame=self.target_token_per_frame,
                        t1=0.8
                    )
                    end_pruning = time.perf_counter()
                    duration = end_pruning - start_pruning
                    fps = T_current / (duration + 1e-6)
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "video_path": video_frame_path[0].split('/')[-2] if video_frame_path else "unknown",
                        "num_frames_T": T_current,
                        "feature_dim_D": D_feat,
                        "selected_k": len(selected_idx),
                        "latency_sec": round(duration, 6),
                        "fps": round(fps, 2)
                    }
                    try:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except Exception as e:
                        eval_logger.error(f"Failed to save efficiency log to {log_file}: {e}")

                    print(f"\n[DPP Efficiency] T={T_current} | Latency={duration:.4f}s | FPS={fps:.2f}")

                    final_daynamic_video_frame_paths.append(selected_images)

                elif USE_Q_FRAME==True:

                    selected_indices, selected_resolution = q_frames(
                        full_feats=full_feats,
                        text_embed=text_embed,
                        tau=1.0
                    )
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, _ = load_and_resize_images(selected_frames, selected_resolution)
                    final_daynamic_video_frame_paths.append(selected_images)
                elif USE_FOCUS==True:
                    T_current = full_feats.shape[0]
                    D_feat = full_feats.shape[1]
                    log_dir = "/path/to/your/workspace/EACL/effi"
                    log_file = os.path.join(log_dir, "dpp_efficiency_log_focus_32.jsonl")
                    start_pruning = time.perf_counter()
                    selected_indices, selected_resolution = focus_sampler(
                        full_feats=full_feats,
                        text_embed=text_embed,
                        fps=1.0,
                        max_num_frames=self.max_num_frames,
                        target_token_per_frame=self.target_token_per_frame
                    )
                    end_pruning = time.perf_counter()
                    duration = end_pruning - start_pruning
                    fps = T_current / (duration + 1e-6)
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "video_path": video_frame_path[0].split('/')[-2] if video_frame_path else "unknown",
                        "num_frames_T": T_current,
                        "feature_dim_D": D_feat,
                        "latency_sec": round(duration, 6),
                        "fps": round(fps, 2)
                    }
                    try:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except Exception as e:
                        eval_logger.error(f"Failed to save efficiency log to {log_file}: {e}")

                    print(f"\n[DPP Efficiency] T={T_current} | Latency={duration:.4f}s | FPS={fps:.2f}")
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, _ = load_and_resize_images(selected_frames, selected_resolution)
                    final_daynamic_video_frame_paths.append(selected_images)
                else:
                    '''T_current = full_feats.shape[0]
                    D_feat = full_feats.shape[1]
                    log_dir = "/path/to/your/workspace/EACL/effi"
                    log_file = os.path.join(log_dir, "dpp_efficiency_log_old_8.jsonl")
                    start_pruning = time.perf_counter()
                    '''
                    selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
                        frame_features=full_feats,
                        frame_embeds=full_feats,
                        text_embed=text_embed,
                        total_token_budget=self.max_num_frames * self.target_token_per_frame,
                        min_token_per_frame=self.min_token_per_frame,
                        max_token_per_frame=self.max_token_per_frame,
                        target_token_per_frame=self.target_token_per_frame
                    )
                    '''
                    end_pruning = time.perf_counter()
                    duration = end_pruning - start_pruning
                    fps = T_current / (duration + 1e-6)
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "video_path": video_frame_path[0].split('/')[-2] if video_frame_path else "unknown",
                        "num_frames_T": T_current,
                        "feature_dim_D": D_feat,
                        "selected_k": len(selected_idx),
                        "latency_sec": round(duration, 6),
                        "fps": round(fps, 2)
                    }
                    try:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except Exception as e:
                        eval_logger.error(f"Failed to save efficiency log to {log_file}: {e}")

                    print(f"\n[DPP Efficiency] T={T_current} | Latency={duration:.4f}s | FPS={fps:.2f}")
                    '''
                    selected_indices = selected_idx.tolist()
                    sorted_pairs = sorted(zip(selected_indices, selected_resolution.tolist()), key=lambda x: x[0])
                    selected_indices = [i for i, _ in sorted_pairs]
                    print(selected_indices)
                    selected_resolution = [r for _, r in sorted_pairs]
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, frame_metadata = load_and_resize_images(selected_frames, selected_resolution)
                    final_daynamic_video_frame_paths.append(selected_images)


            new_batched_messages = []
            for msg_idx, msg in enumerate(batched_messages):

                content = []
                for frame_path in final_daynamic_video_frame_paths[msg_idx]:
                    content.append({
                        "type": "image",
                        "image": frame_path,
                    })

                content.append(batched_messages[msg_idx][0]['content'][-1])
                new_batched_messages.append([{
                    "role": "user",
                    "content": content
                }])


            texts = self.processor.apply_chat_template(new_batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(new_batched_messages)


            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]

                if total_frames > self.max_num_frames:
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)
                    video_inputs[0] = video_inputs[0][indices]
            else:

                pass

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                do_resize=False,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            image_grid_thw = inputs['image_grid_thw']


            for i, (t, h, w) in enumerate(image_grid_thw):
                num_tokens = int(h * w)
                #print(f"视觉单元 {i}: 时间维(T)={t}, 高度(H)={h}, 宽度(W)={w}, Token总数={num_tokens}")

            total_calculated = torch.sum(image_grid_thw[:, 1] * image_grid_thw[:, 2])
            print(f"Check Token: {total_calculated.item()}")
            # continue

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)


            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }

            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res













'''



import time
from typing import List, Optional, Tuple, Union
import numpy as np
from loguru import logger as eval_logger

# ===== Timing Helpers =====
def _safe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _append_jsonl(log_file, payload):
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        eval_logger.error(f"Failed to save timing log to {log_file}: {e}")


from PIL import Image
from tqdm import tqdm
import heapq
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
import torch.nn.functional as F
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages
from lmms_eval.metadata_manager import metadata_manager
import cv2
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via pip install qwen-vl-utils")


import sys
sys.path.append("/path/to/your/workspace/EACL/lmms-eval/lmms_eval/models/simple/Longclip")
from model import longclip

USE_AKS = False
USE_Q_FRAME = False
TEST_CDP_MIN_ALLOCATION = False
TEST_MIN_TOKEN_MAX_COVERAGE = False
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info
import os
import json
from tqdm import tqdm
import argparse
import re
from PIL import Image
import torch
import numpy as np


def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=1024,
    max_token_per_frame=4096,
    target_token_per_frame=512
):
    # 保存原始 device
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # ===== 把计算搬到 CPU =====
    frame_features = frame_features.cpu()
    frame_embeds = frame_embeds.cpu()
    text_embed = text_embed.cpu()

    # 算力预算
    K_max = total_token_budget // min_token_per_frame

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    # 3. 构建 DPP Kernel
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    similarity = (feat @ feat.T).clamp(min=0)

    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)

    kernel = relevance[:, None] * similarity * relevance[None, :]

    # 4. DPP greedy
    cis = torch.zeros((K_max, T))
    di2s = torch.diag(kernel).clone()
    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break

        selected_indices.append(j.item())

        if i == 0:
            eis = kernel[j] / torch.sqrt(current_energy + eps)
        else:
            eis = (kernel[j] - cis[:i, j] @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis
        di2s = di2s - eis ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]
        selected_token_counts = selected_token_counts[sort_idx]


    return final_indices.to(original_device), selected_token_counts.to(original_device)



def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=1024,
    max_token_per_frame=4096,
    target_token_per_frame=512
):
    import torch
    import math

    # 保存原始 device
    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    # ===== 保持 CPU =====
    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    # 算力预算
    K_max = total_token_budget // min_token_per_frame

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)


    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)


    Phi = relevance[:, None] * feat
  
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)

    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break
        

        selected_indices.append(j.item())
        
        phi_j = Phi[j]

        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis


        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== token allocation =====
    selected_token_counts = torch.full(
        (len(final_indices),),
        min_token_per_frame
    ).int()

    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]
        selected_token_counts = selected_token_counts[sort_idx]
    print(final_indices)
    
    return final_indices.to(original_device), selected_token_counts.to(original_device)


def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=256,
    max_token_per_frame=1024,
    target_token_per_frame=512
):
    import torch
    import math

    original_device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    K_max = total_token_budget // min_token_per_frame

    if T <= 1:
        final_indices = torch.tensor([0])
        actual_res = min(total_token_budget, max_token_per_frame)
        selected_token_counts = torch.tensor([actual_res]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    # ===== normalize =====
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)

    # ===== relevance =====
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()

    rel_min, rel_max = relevance.min(), relevance.max()
    relevance = (relevance - rel_min) / (rel_max - rel_min + eps)

    # ===== low-rank Phi =====
    Phi = relevance[:, None] * feat
  
    d = Phi.shape[1]
    cis = torch.zeros((K_max, d))
    di2s = (Phi * Phi).sum(dim=1)

    selected_indices = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]

        if current_energy <= 1e-7:
            break
        
        selected_indices.append(j.item())
        
        phi_j = Phi[j]

        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis

        di2s = di2s - (Phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)

    # ===== 排序 =====
    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]


    if len(final_indices) > 0:
        Phi_S = Phi[final_indices]
        k = Phi_S.shape[0]

        # ===== Gram =====
        G = Phi_S @ Phi_S.T
        G = G + eps * torch.eye(k)

        sign, logdet_full = torch.slogdet(G)

        importance = []

        for i in range(k):
            mask = torch.ones(k, dtype=torch.bool)
            mask[i] = False

            Phi_sub = Phi_S[mask]
            G_sub = Phi_sub @ Phi_sub.T
            G_sub = G_sub + eps * torch.eye(k - 1)

            sign_sub, logdet_sub = torch.slogdet(G_sub)

            importance_i = torch.exp(logdet_full - logdet_sub)
            importance.append(importance_i)

        importance = torch.stack(importance)

        # ===== density =====
        density = (Phi_S * Phi_S).sum(dim=1)
        density = density / (density.mean() + eps)

        alpha = 1.0
        importance = importance * (density ** alpha)

        # normalize
        importance = importance / (importance.sum() + eps)

        print("importance (with density):", importance)

        # ============================================================
        # 🔥 Binary Search Eviction
        # ============================================================

        # 先排序（很关键）
        sorted_imp, sorted_idx = torch.sort(importance, descending=True)

        left, right = 1, len(sorted_imp)
        best_k = 1

        while left <= right:
            mid = (left + right) // 2

            imp_sub = sorted_imp[:mid]
            imp_sub = imp_sub / (imp_sub.sum() + eps)

            tokens = total_token_budget * imp_sub
            tokens = torch.clamp(tokens, min=min_token_per_frame, max=max_token_per_frame)

            if tokens.sum() <= total_token_budget:
                best_k = mid
                left = mid + 1
            else:
                right = mid - 1

        # ===== 最终保留 =====
        keep_idx = sorted_idx[:best_k]

        final_indices = final_indices[keep_idx]
        importance = importance[keep_idx]

        # 重新 normalize
        importance = importance / (importance.sum() + eps)

        # ===== allocation（最终版）=====
        base_tokens = torch.full((best_k,), min_token_per_frame)
        remaining_budget = total_token_budget - best_k * min_token_per_frame

        extra_tokens = remaining_budget * importance

        tokens = base_tokens + extra_tokens
        tokens = torch.clamp(tokens, max=max_token_per_frame)

        selected_token_counts = tokens.int()
        if len(final_indices) > 0:
            time_sort_idx = torch.argsort(final_indices)
            final_indices = final_indices[time_sort_idx]
            importance = importance[time_sort_idx]
            selected_token_counts = selected_token_counts[time_sort_idx]

    else:
        selected_token_counts = torch.full(
            (len(final_indices),),
            min_token_per_frame
        ).int()
    print(selected_token_counts)
    return final_indices.to(original_device), selected_token_counts.to(original_device)


def estimate_hw_from_resolution(orig_h, orig_w, target_token_count, patch_size=14):
    aspect_ratio = orig_h / orig_w
    grid_w = max(1, int(round((target_token_count / aspect_ratio) ** 0.5)))
    grid_h = max(1, int(round(grid_w * aspect_ratio)))
    new_h = grid_h * patch_size
    new_w = grid_w * patch_size
    return new_h, new_w

def load_and_resize_images(frame_paths, resolutions, patch_size=14):
    images = []
    metadata = []
    for idx, (path, token_count) in enumerate(zip(frame_paths, resolutions)):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        new_h, new_w = estimate_hw_from_resolution(orig_h, orig_w, token_count, patch_size)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        images.append(img_resized)
        actual_patches = (new_h // 14) * (new_w // 14)

        metadata.append({
            "frame_idx": idx,
            "path": os.path.basename(path),
            "resolution": f"{new_w}x{new_h}",
            "patches": actual_patches,
            "token_count": token_count,
        })

    return images, metadata

import hashlib
import pickle


def get_cache_key(frame_paths, question, clip_type):
    """
    clip_type: 'openai_clip', 'long_clip', or 'siglip_clip'
    """
    combined_str = "".join(frame_paths) + question + clip_type
    return hashlib.md5(combined_str.encode()).hexdigest()

@register_model("qwen2_5_vl_chat_wo_ours_v4")
class Qwen2_5_VL_Chat_WO_Ours_v4(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(
        self,
        test_cdp_min_allocation: bool = False,
        test_min_token_max_coverage: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.test_cdp_min_allocation = test_cdp_min_allocation
        self.test_min_token_max_coverage = test_min_token_max_coverage

    def generate_until(self, requests: List[Instance]) -> List[str]:

        print(f"[INFO] Total dataset samples: {len(requests)}")
        res = []


        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # Create a mapping from ctx id to the original instance
        # This allows us to attach metadata back to the correct instances later
        # We use ctx (first element of args) as the key since chunks are grouped by args
        req_to_instance = {id(req.args[0]): req for req in requests}

        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            stage_times_batch = {
                "message_prep": 0.0,
                "clip_text": 0.0,
                "clip_image_load": 0.0,
                "clip_image_encode": 0.0,
                "clip_total": 0.0,
                "frame_select": 0.0,
                "resize_load": 0.0,
                "processor": 0.0,
                "generation": 0.0,
                
            }
            total_start_time = time.perf_counter()

            if len(chunk) == 1:
                ctx = [chunk[0][0]]
                doc_to_messages = [chunk[0][1]]
                all_gen_kwargs = [chunk[0][2]]
                doc_id = [chunk[0][3]]
                task = [chunk[0][4]]
                split = [chunk[0][5]]
            else:

                ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
                ctx = list(ctx)
                doc_to_messages = list(doc_to_messages)
                all_gen_kwargs = list(all_gen_kwargs)
                doc_id = list(doc_id)
                task = list(task)
                split = list(split)


            _message_prep_start = time.perf_counter()
            actual_docs = [self.task_dict[task[idx]][split[idx]][doc_id[idx]] for idx in range(len(doc_id))]

            chat_messages = [doc_to_messages[idx](actual_docs[idx]) for idx in range(len(doc_id))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]


            video_frame_paths = []
            for video in videos:
                frame_dir = video[:-4]
                video_frame_path = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
                video_frame_paths.append(video_frame_path)


            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames

            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]


            print("$$$" * 30)
            
            print(batched_messages)
            
            print("$$$" * 30)

        
            
            original_questions = []
            for msg in batched_messages:
                for content in msg[0]['content']:
                    if content['type'] == 'text':
                        if "videomme" in self.task_name:
                            text = content['text']
                            text = text.replace("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n","")
                            question = text.split("\nA.")[0]
                            original_questions.append(question)
                        elif "mlvu" in self.task_name:
                            text = content['text']
                            question = text.split("""\n(A) """)[0]
                            original_questions.append(question)
                        elif "longvideobench" in self.task_name:
                            text = content['text']
                            text = text.replace("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n","")
                            question = text.split("\nA.")[0]
                            original_questions.append(question)
                        elif "lvbench" in self.task_name:
                            text = content['text']
                            text = text.split("\n(A)")[0]
                            original_questions.append(text)
                            

            print("###" * 30)

            print(original_questions)

            print("###" * 30)
            stage_times_batch["message_prep"] += time.perf_counter() - _message_prep_start

            
            full_feats_list = []
            text_embed_list = []

            USE_CLIP_CACHE=False

            cache_dir = "/path/to/your/workspace/cache_center/clip-cache"
            os.makedirs(cache_dir, exist_ok=True)
            clip_type = getattr(self, "frame_encoder_type", "long_clip" if self.use_longclip else "openai_clip")

            full_feats_list = []
            text_embed_list = []

            for frame_paths, question in zip(video_frame_paths, original_questions):
                combined_str = "".join(frame_paths) + question + clip_type
                cache_key = hashlib.md5(combined_str.encode()).hexdigest()
                cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

                if os.path.exists(cache_path) and USE_CLIP_CACHE:
                    eval_logger.debug(f"Loading cached {clip_type} features: {cache_key}")
                    cached_data = torch.load(cache_path, map_location='cpu')
                    full_feats_list.append(cached_data['full_feats'])
                    text_embed_list.append(cached_data['text_embed'])
                    print("exist")
                    continue

                if clip_type == "openai_clip":
                    eval_logger.debug("Running OpenAI-CLIP encoding...")
                    with torch.no_grad():

                        full_feats = self.clip_encoder.encode(
                            frame_paths, batch_size=32, convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=False, device='cuda:0'
                        ).float().cpu()

                        text_embed = self.clip_encoder.encode(
                            [question], convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=False, device='cuda:0'
                        )[0].float().cpu()
                elif clip_type == "long_clip":
                   
                    clip_start = time.perf_counter()

                    eval_logger.debug("Running Long-CLIP encoding with batch processing (FP32)...")
                    batch_size_clip = 1024
                    clip_device = torch.device("cuda:1")

                    self.long_clip_model = self.long_clip_model.to(clip_device)

                    with torch.no_grad():
                        # =========================
                        # 1. TEXT
                        # =========================
                        t0 = time.perf_counter()

                        text_tokens = longclip.tokenize([question]).to(clip_device)

                        text_embed = self.long_clip_model.encode_text(text_tokens)[0]
                        stage_times_batch["clip_text"] += time.perf_counter() - t0
                        text_embed = text_embed / (text_embed.norm() + 1e-6)
                        text_embed = text_embed.detach().cpu().float()

                        del text_tokens

                        

                        # =========================
                        # 2. IMAGE
                        # =========================
                        full_feats_batches = []

                        for i in range(0, len(frame_paths), batch_size_clip):
                            batch_paths = frame_paths[i : i + batch_size_clip]

                            # -------- IO --------
                            t_load = time.perf_counter()

                            images_list = []
                            for path in batch_paths:
                                try:
                                    image = Image.open(path).convert("RGB")
                                    processed = self.long_clip_processor(image)
                                    images_list.append(processed)
                                except Exception as e:
                                    eval_logger.error(f"Error loading image {path}: {e}")
                                    continue

                            stage_times_batch["clip_image_load"] += time.perf_counter() - t_load

                            if not images_list:
                                continue

                            # -------- GPU --------
                            

                            image_batch = torch.stack(images_list).to(clip_device)
                            t_encode = time.perf_counter()
                            batch_feats = self.long_clip_model.encode_image(image_batch)
                            stage_times_batch["clip_image_encode"] += time.perf_counter() - t_encode
                            batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)

                            full_feats_batches.append(batch_feats.detach().cpu().float())

                            del image_batch
                            del batch_feats

                            

                        # =========================
                        # 3. CONCAT
                        # =========================
                        if full_feats_batches:
                            full_feats = torch.cat(full_feats_batches, dim=0)
                        else:
                            full_feats = None

                    stage_times_batch["clip_total"] += time.perf_counter() - clip_start
                elif clip_type == "siglip_clip":
                    eval_logger.debug("Running SigLIP encoding...")
                    self.siglip_model = self.siglip_model.to(self.model.device)
                    batch_size_siglip = 32

                    with torch.no_grad():
                        text_inputs = self.siglip_processor(
                            text=question,
                            padding="max_length",
                            return_tensors="pt",
                        ).to(self.siglip_model.device)

                        stride_num = (int(text_inputs["input_ids"].shape[-1]) + 63) // 64
                        stride = (text_inputs["input_ids"].shape[-1] + stride_num - 1) // stride_num

                        input_id_heads, input_id_tails = [], []
                        left, right = 0, text_inputs["input_ids"].shape[-1]
                        while left < right:
                            input_id_heads.append(text_inputs["input_ids"][:, left : left + stride])
                            left += stride
                            if left < right:
                                input_id_tails.append(text_inputs["input_ids"][:, right - stride : right])
                                right -= stride

                        text_input_ids = torch.cat(input_id_heads + input_id_tails[::-1])
                        text_outputs = self.siglip_model.get_text_features(input_ids=text_input_ids)
                        text_embed = text_outputs.mean(dim=0, keepdim=True)[0]
                        text_embed = text_embed / (text_embed.norm() + 1e-6)
                        text_embed = text_embed.detach().cpu().float()

                        full_feats_batches = []
                        for i in range(0, len(frame_paths), batch_size_siglip):
                            batch_paths = frame_paths[i : i + batch_size_siglip]
                            images_list = []
                            for path in batch_paths:
                                try:
                                    image = Image.open(path).convert("RGB")
                                    images_list.append(image)
                                except Exception as e:
                                    eval_logger.error(f"Error loading image {path}: {e}")
                                    continue

                            if not images_list:
                                continue

                            image_inputs = self.siglip_processor(
                                images=images_list,
                                return_tensors="pt",
                            ).to(self.siglip_model.device)
                            batch_feats = self.siglip_model.get_image_features(
                                pixel_values=image_inputs["pixel_values"]
                            )
                            batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)
                            full_feats_batches.append(batch_feats.detach().cpu().float())

                        full_feats = torch.cat(full_feats_batches, dim=0) if full_feats_batches else None
                else:
                    raise ValueError(f"Unsupported frame_encoder_type: {clip_type}")
                if USE_CLIP_CACHE:
                    torch.save({
                        'full_feats': full_feats,
                        'text_embed': text_embed,
                        'clip_type': clip_type,   # 额外保存元数据，方便排查
                        'question': question      # 可选
                    }, cache_path)
                full_feats_list.append(full_feats)
                text_embed_list.append(text_embed)

            

            physical_patch_limits = []
            for video_frame_path in video_frame_paths:
                with Image.open(video_frame_path[0]) as sample_img:
                    orig_w, orig_h = sample_img.size
                physical_patch_limit = (orig_h // 14) * (orig_w // 14)
                physical_patch_limits.append(physical_patch_limit)


            print("Video Frame Paths and Physical Patch Limits:")
            print(f"physical_patch_limits: {physical_patch_limits}")

            print("Start DPP pruning and dynamic token allocation...")



            final_daynamic_video_frame_paths = []
            all_frame_metadata = []


            for full_feats, text_embed, video_frame_path, physical_patch_limit in zip(full_feats_list, text_embed_list, video_frame_paths, physical_patch_limits):
                _frame_select_start = time.perf_counter()
                full_feats = full_feats.to(self.model.device)
                text_embed = text_embed.to(self.model.device)
                if USE_AKS==True:
                    with torch.no_grad():
                        raw_scores = (full_feats @ text_embed).cpu().numpy()
                    selected_images = aks(
                        scores=raw_scores,
                        frame_paths=video_frame_path,
                        max_num_frames=self.max_num_frames,
                        target_token_per_frame=self.target_token_per_frame,
                        t1=0.8
                    )

                    final_daynamic_video_frame_paths.append(selected_images)
                    stage_times_batch["frame_select"] += time.perf_counter() - _frame_select_start

                elif USE_Q_FRAME==True:

                    selected_indices, selected_resolution = q_frames(
                        full_feats=full_feats,
                        text_embed=text_embed,
                        tau=1.0
                    )
                    stage_times_batch["frame_select"] += time.perf_counter() - _frame_select_start
                    _resize_start = time.perf_counter()
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, _ = load_and_resize_images(selected_frames, selected_resolution)
                    stage_times_batch["resize_load"] += time.perf_counter() - _resize_start
                    final_daynamic_video_frame_paths.append(selected_images)
                else:
                    T_current = full_feats.shape[0]
                    D_feat = full_feats.shape[1]
                    log_dir = "/path/to/your/workspace/EACL/effi"
                    log_file = os.path.join(log_dir, "dpp_efficiency_old_log.jsonl")
                    start_pruning = time.perf_counter()
                    selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
                        frame_features=full_feats,
                        frame_embeds=full_feats,
                        text_embed=text_embed,
                        total_token_budget=self.max_num_frames * self.target_token_per_frame,
                        min_token_per_frame=self.min_token_per_frame,
                        max_token_per_frame=self.max_token_per_frame,
                        target_token_per_frame=self.target_token_per_frame
                    )
                    end_pruning = time.perf_counter()
                    duration = end_pruning - start_pruning
                    stage_times_batch["frame_select"] += duration
                    fps = T_current / (duration + 1e-6)
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "video_path": video_frame_path[0].split('/')[-2] if video_frame_path else "unknown",
                        "num_frames_T": T_current,
                        "feature_dim_D": D_feat,
                        "selected_k": len(selected_idx),
                        "latency_sec": round(duration, 6),
                        "fps": round(fps, 2)
                    }
                    try:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except Exception as e:
                        eval_logger.error(f"Failed to save efficiency log to {log_file}: {e}")

                    print(f"\n[DPP Efficiency] T={T_current} | Latency={duration:.4f}s | FPS={fps:.2f}")
                    selected_indices = selected_idx.tolist()
                    _resize_start = time.perf_counter()
                    sorted_pairs = sorted(zip(selected_indices, selected_resolution.tolist()), key=lambda x: x[0])
                    selected_indices = [i for i, _ in sorted_pairs]
                    selected_resolution = [r for _, r in sorted_pairs]
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, frame_metadata = load_and_resize_images(selected_frames, selected_resolution)
                    stage_times_batch["resize_load"] += time.perf_counter() - _resize_start
                    final_daynamic_video_frame_paths.append(selected_images)


            new_batched_messages = []
            for msg_idx, msg in enumerate(batched_messages):

                content = []
                for frame_path in final_daynamic_video_frame_paths[msg_idx]:
                    content.append({
                        "type": "image",
                        "image": frame_path,
                    })

                content.append(batched_messages[msg_idx][0]['content'][-1])
                new_batched_messages.append([{
                    "role": "user",
                    "content": content
                }])


            texts = self.processor.apply_chat_template(new_batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(new_batched_messages)


            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]

                if total_frames > self.max_num_frames:
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)
                    video_inputs[0] = video_inputs[0][indices]
            else:

                pass

            padding_side = "left" if self.batch_size > 1 else "right"
            _processor_start = time.perf_counter()
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                do_resize=False,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            stage_times_batch["processor"] += time.perf_counter() - _processor_start

            image_grid_thw = inputs['image_grid_thw']


            for i, (t, h, w) in enumerate(image_grid_thw):
                num_tokens = int(h * w)
                #print(f"视觉单元 {i}: 时间维(T)={t}, 高度(H)={h}, 宽度(W)={w}, Token总数={num_tokens}")

            total_calculated = torch.sum(image_grid_thw[:, 1] * image_grid_thw[:, 2])
            print(f"Check Token: {total_calculated.item()}")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)


            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }

            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            _safe_cuda_sync()
            _generation_start = time.perf_counter()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            _safe_cuda_sync()
            stage_times_batch["generation"] += time.perf_counter() - _generation_start
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
            stage_times_batch["total"] = time.perf_counter() - total_start_time
            _append_jsonl(
                "/path/to/your/workspace/EACL/effi/full_pipeline_ablation.jsonl",
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "batch_size": len(chunk),
                    **{k: round(v, 6) for k, v in stage_times_batch.items()},
                }
            )
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
'''
