import hashlib
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import (
    Qwen2_5_VL as Qwen2_5_VLSimple,
    longclip,
)
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def cdpruner_dpp_dynamic_resolution(
    frame_features: torch.Tensor,
    frame_embeds: torch.Tensor,
    text_embed: torch.Tensor,
    total_token_budget: int,
    min_token_per_frame: int = 256,
    max_token_per_frame: int = 1024,
    tau: float = 1.0,
):
    original_device = frame_features.device
    eps = 1e-6
    num_frames = frame_features.shape[0]

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    max_selected_frames = max(1, total_token_budget // (min_token_per_frame * 2))

    if num_frames <= 1:
        final_indices = torch.tensor([0])
        selected_token_counts = torch.tensor([min(total_token_budget, max_token_per_frame)]).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + eps)

    weighted_features = relevance[:, None] * feat
    feature_dim = weighted_features.shape[1]
    basis = torch.zeros((max_selected_frames, feature_dim))
    residual_energy = (weighted_features * weighted_features).sum(dim=1)
    selected_indices = []

    for step in range(max_selected_frames):
        index = torch.argmax(residual_energy)
        current_energy = residual_energy[index]
        if current_energy <= 1e-7:
            break

        selected_indices.append(index.item())
        feature = weighted_features[index]

        if step == 0:
            orthogonal = feature / torch.sqrt(current_energy + eps)
        else:
            projection = basis[:step] @ feature
            orthogonal = (feature - projection @ basis[:step]) / torch.sqrt(current_energy + eps)

        basis[step] = orthogonal
        residual_energy = torch.clamp(residual_energy - (weighted_features @ orthogonal) ** 2, min=0)
        residual_energy[index] = -float("inf")

    final_indices = torch.tensor(selected_indices)
    if len(final_indices) == 0:
        selected_token_counts = torch.full((0,), min_token_per_frame).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    final_indices = final_indices[torch.argsort(final_indices)]
    selected_features = weighted_features[final_indices]
    selected_count = selected_features.shape[0]
    gram = selected_features @ selected_features.T
    gram = gram + eps * torch.eye(selected_count)
    _, logdet_full = torch.slogdet(gram)

    importance = []
    for index in range(selected_count):
        mask = torch.ones(selected_count, dtype=torch.bool)
        mask[index] = False
        reduced_features = selected_features[mask]
        reduced_gram = reduced_features @ reduced_features.T
        reduced_gram = reduced_gram + eps * torch.eye(selected_count - 1)
        _, logdet_reduced = torch.slogdet(reduced_gram)
        importance.append(torch.exp(logdet_full - logdet_reduced))
    importance = torch.stack(importance)

    density = (selected_features * selected_features).sum(dim=1)
    density = density / (density.mean() + eps)
    importance = importance * (density**tau)
    importance = importance / (importance.sum() + eps)

    sorted_importance, sorted_indices = torch.sort(importance, descending=True)
    left, right = 1, len(sorted_importance)
    best_count = 1

    while left <= right:
        mid = (left + right) // 2
        current_importance = sorted_importance[:mid]
        current_importance = current_importance / (current_importance.sum() + eps)
        tokens = torch.clamp(
            total_token_budget * current_importance,
            min=min_token_per_frame,
            max=max_token_per_frame,
        )

        if tokens.sum() <= total_token_budget:
            best_count = mid
            left = mid + 1
        else:
            right = mid - 1

    keep_indices = sorted_indices[:best_count]
    final_indices = final_indices[keep_indices]
    importance = importance[keep_indices]
    importance = importance / (importance.sum() + eps)

    base_tokens = torch.full((best_count,), min_token_per_frame)
    remaining_budget = total_token_budget - best_count * min_token_per_frame
    tokens = base_tokens + remaining_budget * importance
    tokens = torch.clamp(tokens, max=max_token_per_frame)
    selected_token_counts = tokens.int()

    time_order = torch.argsort(final_indices)
    final_indices = final_indices[time_order]
    selected_token_counts = selected_token_counts[time_order]
    return final_indices.to(original_device), selected_token_counts.to(original_device)


def estimate_hw_from_resolution(orig_h: int, orig_w: int, target_token_count: int, patch_size: int = 14):
    aspect_ratio = orig_h / orig_w
    grid_w = max(1, int(round((target_token_count / aspect_ratio) ** 0.5)))
    grid_h = max(1, int(round(grid_w * aspect_ratio)))
    return grid_h * patch_size, grid_w * patch_size


def load_and_resize_images(frame_paths: List[str], resolutions: List[int], patch_size: int = 14):
    images = []
    for path, token_count in zip(frame_paths, resolutions):
        with Image.open(path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            new_h, new_w = estimate_hw_from_resolution(orig_h, orig_w, token_count, patch_size)
            images.append(image.resize((new_w, new_h), resample=Image.BICUBIC))
    return images


def get_cache_key(frame_paths: List[str], question: str, clip_type: str):
    return hashlib.md5(("".join(frame_paths) + question + clip_type).encode()).hexdigest()


def extract_question(task_name: str, message) -> str:
    question = ""
    for content in message[0]["content"]:
        if content.get("type") == "text":
            question = content.get("text", "")
            break

    lower_task_name = task_name.lower()
    if "videomme" in lower_task_name or "longvideobench" in lower_task_name:
        prompt = (
            "Select the best answer to the following multiple-choice question based on "
            "the video and the subtitles. Respond with only the letter (A, B, C, or D) "
            "of the correct option.\n"
        )
        question = question.replace(prompt, "").split("\nA.")[0]
    elif "mlvu" in lower_task_name:
        question = question.split("\n(A) ")[0]
    elif "lvbench" in lower_task_name:
        question = question.split("\n(A)")[0]

    return question.strip()


@register_model("qwen2_5_vl_chat_wo_ours_v4")
class Qwen2_5_VL_Chat_WO_Ours_v4(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("frame_encoder_type", "long_clip")
        super().__init__(**kwargs)
        if self.frame_encoder_type != "long_clip":
            raise ValueError("qwen2_5_vl_chat_wo_ours_v4 only supports frame_encoder_type='long_clip'.")

        default_cache_dir = Path.home() / ".cache" / "lmms_eval" / "clip-cache"
        self.clip_cache_dir = Path(os.environ.get("LMMS_EVAL_CLIP_CACHE", default_cache_dir))
        self.clip_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_frame_paths(self, videos: List[str]) -> List[List[str]]:
        video_frame_paths = []
        for video in videos:
            frame_dir = Path(video).with_suffix("")
            frame_paths = sorted(str(path) for path in frame_dir.iterdir() if path.is_file())
            video_frame_paths.append(frame_paths)
        return video_frame_paths

    def _encode_video_frames(self, frame_paths: List[str], question: str):
        cache_key = get_cache_key(frame_paths, question, "long_clip")
        cache_path = self.clip_cache_dir / f"{cache_key}.pt"

        if cache_path.exists():
            cached_data = torch.load(cache_path, map_location="cpu")
            return cached_data["full_feats"], cached_data["text_embed"]

        batch_size = 1024
        with torch.no_grad():
            text_tokens = longclip.tokenize([question]).to(self.model.device)
            text_embed = self.long_clip_model.encode_text(text_tokens)[0]
            text_embed = text_embed / (text_embed.norm() + 1e-6)
            text_embed = text_embed.detach().cpu().float()

            full_feats_batches = []
            for start in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    try:
                        with Image.open(path) as image:
                            images.append(self.long_clip_processor(image.convert("RGB")))
                    except Exception as error:
                        eval_logger.error(f"Error loading image {path}: {error}")

                if not images:
                    continue

                image_batch = torch.stack(images).to(self.model.device)
                batch_feats = self.long_clip_model.encode_image(image_batch)
                batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)
                full_feats_batches.append(batch_feats.detach().cpu().float())

        full_feats = torch.cat(full_feats_batches, dim=0) if full_feats_batches else None
        torch.save({"full_feats": full_feats, "text_embed": text_embed}, cache_path)
        return full_feats, text_embed

    def _build_dynamic_messages(self, batched_messages, selected_images):
        new_batched_messages = []
        for message, images in zip(batched_messages, selected_images):
            content = [{"type": "image", "image": image} for image in images]
            content.append(message[0]["content"][-1])
            new_batched_messages.append([{"role": "user", "content": content}])
        return new_batched_messages

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            _, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            doc_to_messages = list(doc_to_messages)
            all_gen_kwargs = list(all_gen_kwargs)
            doc_id = list(doc_id)
            task = list(task)
            split = list(split)

            actual_docs = [self.task_dict[task[idx]][split[idx]][doc_id[idx]] for idx in range(len(doc_id))]
            chat_messages = [doc_to_messages[idx](actual_docs[idx]) for idx in range(len(doc_id))]
            chat_messages = [ChatMessages(**{"messages": message}) for message in chat_messages]

            videos = []
            for messages in chat_messages:
                _, video, _ = messages.extract_media()
                videos.append(video)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames

            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            original_questions = [extract_question(self.task_name, message) for message in batched_messages]
            video_frame_paths = self._get_video_frame_paths(videos)

            selected_images = []
            for frame_paths, question in zip(video_frame_paths, original_questions):
                full_feats, text_embed = self._encode_video_frames(frame_paths, question)
                if full_feats is None:
                    selected_images.append([])
                    continue

                selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
                    frame_features=full_feats.to(self.model.device),
                    frame_embeds=full_feats.to(self.model.device),
                    text_embed=text_embed.to(self.model.device),
                    total_token_budget=self.max_num_frames * self.target_token_per_frame,
                    min_token_per_frame=self.min_token_per_frame,
                    max_token_per_frame=self.max_token_per_frame,
                )

                selected_indices = selected_idx.tolist()
                selected_resolutions = selected_resolution.tolist()
                ordered_pairs = sorted(zip(selected_indices, selected_resolutions), key=lambda item: item[0])
                ordered_indices = [index for index, _ in ordered_pairs]
                ordered_resolutions = [resolution for _, resolution in ordered_pairs]
                selected_frames = [frame_paths[index] for index in ordered_indices]
                selected_images.append(load_and_resize_images(selected_frames, ordered_resolutions))

            new_batched_messages = self._build_dynamic_messages(batched_messages, selected_images)
            texts = self.processor.apply_chat_template(new_batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(new_batched_messages)

            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                if total_frames > self.max_num_frames:
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    video_inputs[0] = video_inputs[0][np.unique(indices)]

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                do_resize=False,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

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

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

            pbar.update(1)

        res = re_ords.get_original(res)
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
