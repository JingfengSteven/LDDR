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
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages
from lmms_eval.metadata_manager import metadata_manager
import math
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via pip install qwen-vl-utils")


import sys
sys.path.append("/path/to/your/workspace/EACL/lmms-eval/lmms_eval/models/simple/Longclip")
from model import longclip

USE_AKS = False
USE_Q_FRAME = False
USE_FOCUS = True

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
import numpy as np

import torch
import numpy as np

def focus_sampler(
    full_feats, 
    text_embed, 
    fps, 
    max_num_frames,
    target_token_per_frame=1024,
    coarse_every_sec=16.0,
    fine_every_sec=1.0,
    zoom_ratio=0.25,
    min_coarse_segments=8,
    min_zoom_segments=4,
    min_gap_sec=1.0,
    top_ratio=0.2,
    temperature=0.06,
    exploration_weight=1.5,
    # 对应代码中的 Adaptive Gap 逻辑
    disable_gap_below_sec=0.2,
    gap_ratio_of_avg=0.25
):
    total_frames = full_feats.shape[0]
    video_duration = total_frames / max(1.0, fps)
    
    # 1. 基础得分计算 (保留 Long-CLIP 预计算优势)
    scores = (full_feats @ text_embed).squeeze().cpu().numpy()
    
    # 2. 自适应 Min-Gap 计算 (对应 I/O 代码第 138-143 行)
    avg_spacing_sec = video_duration / max(1, max_num_frames)
    if avg_spacing_sec <= disable_gap_below_sec:
        auto_min_gap_sec = 0.0
    else:
        auto_min_gap_sec = min(gap_ratio_of_avg * avg_spacing_sec, min_gap_sec)
    min_gap_frames = int(auto_min_gap_sec * fps)

    # 3. 粗采样阶段 (Coarse Sampling)
    actual_coarse_step = coarse_every_sec
    if video_duration / coarse_every_sec < min_coarse_segments:
        actual_coarse_step = max(1.0/fps, video_duration / max(1, min_coarse_segments))
    step = max(1, int(actual_coarse_step * fps))
    coarse_indices = np.arange(0, total_frames, step)
    
    # 4. 区域定义与 UCB 评估 (Arms Generation)
    half_win = int(actual_coarse_step * fps // 2)
    arms = []
    for idx in coarse_indices:
        start, end = max(0, idx - half_win), min(total_frames, idx + half_win)
        arm_scores = scores[start:end]
        mu = np.mean(arm_scores)
        sigma = np.std(arm_scores) + 1e-6
        ucb = mu + exploration_weight * sigma
        arms.append({'range': (start, end), 'ucb': ucb, 'scores': arm_scores})

    # 5. 区域筛选 (Zooming)
    num_zoom = max(int(len(arms) * zoom_ratio), min_zoom_segments)
    num_zoom = min(num_zoom, len(arms))
    arms.sort(key=lambda x: x['ucb'], reverse=True)
    selected_arms = arms[:num_zoom]

    # 6. 预算分配 (Budget Allocation)
    # 分为 Top-Ranked 名额和 UCB 分配名额
    num_top = int(max_num_frames * top_ratio)
    num_ucb_budget = max_num_frames - num_top
    
    ucb_vals = np.array([a['ucb'] for a in selected_arms])
    weights = ucb_vals / (ucb_vals.sum() + 1e-9)
    arm_budgets = np.round(weights * num_ucb_budget).astype(int)

    # 7. 区域内采样 (Within-Arm Softmax Sampling)
    # 对应 I/O 代码参数中的 temperature 逻辑
    final_candidates = []
    for i, arm in enumerate(selected_arms):
        budget = arm_budgets[i]
        if budget <= 0: continue
        
        # Softmax 概率分布
        s = arm['scores']
        probs = np.exp((s - np.max(s)) / temperature)
        probs /= probs.sum()
        
        # 按概率抽取点
        local_indices = np.random.choice(
            len(s), size=min(budget, len(s)), replace=False, p=probs
        )
        final_candidates.extend(np.arange(arm['range'][0], arm['range'][1])[local_indices])

    # 8. 全局 Top 补全 (Top-Ranked Selection)
    # 确保最相关的帧一定被考虑
    global_top_idx = np.argsort(scores)[-num_top*2:] 
    final_candidates.extend(global_top_idx)

    # 9. 最终筛选 (Min-Gap 过滤)
    # 先按原始得分排序，保证高分帧优先占据位置
    final_candidates = sorted(list(set(final_candidates)), key=lambda x: scores[x], reverse=True)
    selected_indices = []
    
    for c_idx in final_candidates:
        if len(selected_indices) >= max_num_frames:
            break
        # 检查是否满足动态 Min-Gap
        if all(abs(c_idx - s) >= min_gap_frames for s in selected_indices):
            selected_indices.append(c_idx)

    # 10. 排序并打包
    final_indices = sorted(selected_indices)
    selected_resolution = torch.full((len(final_indices),), target_token_per_frame).int()
    
    return torch.tensor(final_indices), selected_resolution

def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=512,
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

    # ===== 结果再放回原 device =====
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
        #print(f"Path: {os.path.basename(path)} Org:{orig_w}X{orig_h}| Res: {img_resized.size} | Actual Patches: {actual_patches}")

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
    clip_type: 'openai_clip' or 'long_clip'
    """
    combined_str = "".join(frame_paths) + question + clip_type
    return hashlib.md5(combined_str.encode()).hexdigest()

@register_model("qwen2_5_vl_chat_wo_ours_v4_dynamic")
class Qwen2_5_VL_Chat_WO_Ours_v4_dynamic(Qwen2_5_VLSimple):
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

            ### 打印一下
            print("$$$" * 30)
            
            print(batched_messages)
            
            print("$$$" * 30)
            ###
        
            
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
            clip_type = "long_clip" if self.use_longclip else "openai_clip"

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

                if not self.use_longclip:
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
                else:
                    eval_logger.debug("Running Long-CLIP encoding with batch processing (FP32)...")
                    batch_size_clip = 1024

                    with torch.no_grad():
                        text_tokens = longclip.tokenize([question]).to(self.model.device)
                        text_embed = self.long_clip_model.encode_text(text_tokens)[0]
                        text_embed = text_embed / (text_embed.norm() + 1e-6)
                        # 立即转回 CPU，释放 GPU 引用
                        text_embed = text_embed.detach().cpu().float()

                        del text_tokens

                        full_feats_batches = []

                        for i in range(0, len(frame_paths), batch_size_clip):
                            batch_paths = frame_paths[i : i + batch_size_clip]
                            images_list = []

                          
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

                            
                            full_feats_batches.append(batch_feats.detach().cpu().float())

                            del image_batch
                            del batch_feats

                        if full_feats_batches:
                            full_feats = torch.cat(full_feats_batches, dim=0)
                        else:
                            full_feats = None
                torch.save({
                    'full_feats': full_feats,
                    'text_embed': text_embed,
                    'clip_type': clip_type,   
                    'question': question     
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
                    selected_images = aks(
                        scores=raw_scores,
                        frame_paths=video_frame_path,
                        max_num_frames=self.max_num_frames,
                        target_token_per_frame=self.target_token_per_frame,
                        t1=0.8
                    )

                    final_daynamic_video_frame_paths.append(selected_images)
                elif USE_FOCUS:
                    selected_indices, selected_resolution = focus_sampler(
                        full_feats=full_feats,
                        text_embed=text_embed,
                        fps=1.0,
                        max_num_frames=self.max_num_frames,
                        target_token_per_frame=self.target_token_per_frame
                    )
                    selected_frames = [video_frame_path[i] for i in selected_indices]
                    selected_images, _ = load_and_resize_images(selected_frames, selected_resolution.tolist())
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
                else:

                    selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
                        frame_features=full_feats,
                        frame_embeds=full_feats,     
                        text_embed=text_embed,
                        total_token_budget=self.max_num_frames * self.target_token_per_frame,
                        min_token_per_frame=self.min_token_per_frame,
                        max_token_per_frame=self.max_token_per_frame,
                                  
                    )
                    selected_indices = selected_idx.tolist()
                    sorted_pairs = sorted(zip(selected_indices, selected_resolution.tolist()), key=lambda x: x[0])
                    selected_indices = [i for i, _ in sorted_pairs]
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
