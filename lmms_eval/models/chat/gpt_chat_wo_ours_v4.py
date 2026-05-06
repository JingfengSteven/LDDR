import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.openai_compatible import (
    OpenAICompatible as OpenAICompatibleSimple,
)
from lmms_eval.models.simple.qwen2_5_vl import (
    CLIP_MODEL_PATH,
    LONG_CLIP_MODEL_PATH,
    SIGLIP_MODEL_PATH,
    longclip,
)
from lmms_eval.protocol import ChatMessages


def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=256,
    max_token_per_frame=1024,
    target_token_per_frame=512,
):
    original_device = frame_features.device
    eps = 1e-6
    total_frames = frame_features.shape[0]

    frame_features = frame_features.cpu().float()
    frame_embeds = frame_embeds.cpu().float()
    text_embed = text_embed.cpu().float()

    max_frames = max(1, total_token_budget // max(1, min_token_per_frame))

    if total_frames <= 1:
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

    phi = relevance[:, None] * feat
    feature_dim = phi.shape[1]
    cis = torch.zeros((max_frames, feature_dim))
    di2s = (phi * phi).sum(dim=1)

    selected_indices = []
    for i in range(max_frames):
        j = torch.argmax(di2s)
        current_energy = di2s[j]
        if current_energy <= 1e-7:
            break

        selected_indices.append(j.item())
        phi_j = phi[j]
        if i == 0:
            eis = phi_j / torch.sqrt(current_energy + eps)
        else:
            proj = cis[:i] @ phi_j
            eis = (phi_j - proj @ cis[:i]) / torch.sqrt(current_energy + eps)

        cis[i] = eis
        di2s = di2s - (phi @ eis) ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf")

    final_indices = torch.tensor(selected_indices)
    if len(final_indices) > 0:
        sort_idx = torch.argsort(final_indices)
        final_indices = final_indices[sort_idx]

    if len(final_indices) == 0:
        selected_token_counts = torch.full((0,), min_token_per_frame).int()
        return final_indices.to(original_device), selected_token_counts.to(original_device)

    phi_selected = phi[final_indices]
    selected_count = phi_selected.shape[0]

    gram = phi_selected @ phi_selected.T
    gram = gram + eps * torch.eye(selected_count)
    _, logdet_full = torch.slogdet(gram)

    importance = []
    for i in range(selected_count):
        mask = torch.ones(selected_count, dtype=torch.bool)
        mask[i] = False
        sub = phi_selected[mask]
        gram_sub = sub @ sub.T
        gram_sub = gram_sub + eps * torch.eye(selected_count - 1)
        _, logdet_sub = torch.slogdet(gram_sub)
        importance.append(torch.exp(logdet_full - logdet_sub))

    importance = torch.stack(importance)
    density = (phi_selected * phi_selected).sum(dim=1)
    density = density / (density.mean() + eps)
    importance = importance * density
    importance = importance / (importance.sum() + eps)

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

    keep_idx = sorted_idx[:best_k]
    final_indices = final_indices[keep_idx]
    importance = importance[keep_idx]
    importance = importance / (importance.sum() + eps)

    base_tokens = torch.full((best_k,), min_token_per_frame)
    remaining_budget = total_token_budget - best_k * min_token_per_frame
    extra_tokens = remaining_budget * importance
    tokens = torch.clamp(base_tokens + extra_tokens, max=max_token_per_frame)
    selected_token_counts = tokens.int()

    time_sort_idx = torch.argsort(final_indices)
    final_indices = final_indices[time_sort_idx]
    selected_token_counts = selected_token_counts[time_sort_idx]
    return final_indices.to(original_device), selected_token_counts.to(original_device)


def estimate_hw_from_resolution(orig_h, orig_w, target_token_count, patch_size=14):
    aspect_ratio = orig_h / max(orig_w, 1)
    grid_w = max(1, int(round((target_token_count / max(aspect_ratio, 1e-6)) ** 0.5)))
    grid_h = max(1, int(round(grid_w * aspect_ratio)))
    new_h = grid_h * patch_size
    new_w = grid_w * patch_size
    return new_h, new_w


def load_and_resize_images(frame_paths, resolutions, patch_size=14):
    images = []
    frame_metadata = []
    for path, token_count in zip(frame_paths, resolutions):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        new_h, new_w = estimate_hw_from_resolution(orig_h, orig_w, token_count, patch_size)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        images.append(img_resized)
        frame_metadata.append(
            {
                "path": path,
                "token_count": int(token_count),
                "width": int(new_w),
                "height": int(new_h),
                "resolution": f"{int(new_w)}x{int(new_h)}",
                "original_width": int(orig_w),
                "original_height": int(orig_h),
            }
        )
    return images, frame_metadata


def get_cache_key(frame_paths, question, clip_type):
    combined_str = "".join(frame_paths) + question + clip_type
    return hashlib.md5(combined_str.encode()).hexdigest()


@register_model("gpt_chat_wo_ours_v4")
class GPT_Chat_WO_Ours_v4(OpenAICompatibleSimple):
    is_simple = False

    def __init__(
        self,
        model_version: str = "gpt-4o",
        frame_encoder_type: str = "openai_clip",
        max_num_frames: int = 32,
        target_token_per_frame: int = 512,
        min_token_per_frame: int = 256,
        max_token_per_frame: int = 1024,
        task_name: str = "",
        clip_cache_dir: str = "/path/to/your/workspace/cache_center/clip-cache",
        video_frame_cache_dir: str = "/path/to/your/workspace/cache_center/video-frame-cache",
        min_completion_tokens_for_reasoning: int = 16000,
        max_retries: int = 5,
        timeout: int = 10,
        batch_size: int = 8,
        **kwargs,
    ) -> None:
        kwargs = dict(kwargs)
        kwargs.setdefault("model_version", model_version)
        kwargs.setdefault("max_frames_num", max_num_frames)
        kwargs.setdefault("max_retries", max_retries)
        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("batch_size", batch_size)
        super().__init__(**kwargs)

        valid_frame_encoder_types = ["openai_clip", "long_clip", "siglip_clip"]
        if frame_encoder_type not in valid_frame_encoder_types:
            raise ValueError(f"frame_encoder_type must be one of {valid_frame_encoder_types}, got {frame_encoder_type}")

        self.frame_encoder_type = frame_encoder_type
        self.max_num_frames = max_num_frames
        self.max_frames_num = max_num_frames
        self.target_token_per_frame = target_token_per_frame
        self.min_token_per_frame = min_token_per_frame
        self.max_token_per_frame = max_token_per_frame
        self.task_name = task_name
        self.clip_cache_dir = clip_cache_dir
        self.video_frame_cache_dir = video_frame_cache_dir
        self.min_completion_tokens_for_reasoning = min_completion_tokens_for_reasoning
        os.makedirs(self.clip_cache_dir, exist_ok=True)
        os.makedirs(self.video_frame_cache_dir, exist_ok=True)

        clip_device = str(self.device) if torch.cuda.is_available() else "cpu"
        self.clip_encoder = None
        self.long_clip_model = None
        self.long_clip_processor = None
        self.siglip_model = None
        self.siglip_processor = None

        if self.frame_encoder_type == "openai_clip":
            self.clip_encoder = SentenceTransformer(CLIP_MODEL_PATH, device=clip_device)
        elif self.frame_encoder_type == "long_clip":
            self.long_clip_model, self.long_clip_processor = longclip.load(LONG_CLIP_MODEL_PATH, device=clip_device)
        else:
            self.siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device).eval()
            self.siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)

    def _extract_question(self, text: str) -> str:
        if "videomme" in self.task_name or "longvideobench" in self.task_name:
            text = text.replace(
                "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n",
                "",
            )
            return text.split("\nA.")[0].strip()
        if "mlvu" in self.task_name:
            return text.split("\n(A) ")[0].strip()
        if "lvbench" in self.task_name:
            return text.split("\n(A)")[0].strip()
        return text.strip()

    def _extract_question_from_messages(self, chat_messages: ChatMessages) -> str:
        text_parts = []
        for message in chat_messages.messages:
            for content in message.content:
                if content.type == "text":
                    text_parts.append(content.text)
        merged = "\n".join(part for part in text_parts if part)
        return self._extract_question(merged)

    def _resolve_frame_paths(self, video_path: str) -> List[str]:
        video_path_obj = Path(video_path)
        candidate_dirs = []

        # Prefer a sibling extracted-frame directory even when the raw .mp4 file
        # is missing. Some datasets, such as LongVideoBench, ship only frames.
        if video_path_obj.suffix:
            candidate_dirs.append(video_path_obj.with_suffix(""))
        candidate_dirs.append(video_path_obj)

        frame_dir = None
        for candidate in candidate_dirs:
            if candidate.is_dir():
                frame_dir = candidate
                break

        if frame_dir is None:
            if video_path_obj.is_file():
                frame_dir = Path(self._extract_video_frames_to_cache(str(video_path_obj)))
            else:
                raise FileNotFoundError(f"Frame directory not found for video: {video_path}")

        frame_paths = sorted(
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in directory: {frame_dir}")
        return frame_paths

    def _extract_video_frames_to_cache(self, video_path: str) -> str:
        if VideoReader is None:
            raise ImportError(
                "decord is required to decode raw video files for gpt_chat_wo_ours_v4. "
                "Please install it or pre-extract frames."
            )

        stat = os.stat(video_path)
        cache_key = hashlib.md5(
            f"{video_path}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
        ).hexdigest()
        frame_dir = os.path.join(self.video_frame_cache_dir, cache_key)
        os.makedirs(frame_dir, exist_ok=True)

        existing_frames = sorted(
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if name.lower().endswith(".jpg")
        )
        if existing_frames:
            return frame_dir

        eval_logger.info(f"Extracting frames from raw video to cache: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        for idx in range(total_frames):
            frame = vr[idx].asnumpy()
            image = Image.fromarray(frame)
            image.save(os.path.join(frame_dir, f"{idx:06d}.jpg"), format="JPEG", quality=95)

        return frame_dir

    def _encode_with_openai_clip(self, frame_paths: List[str], question: str):
        with torch.no_grad():
            full_feats = self.clip_encoder.encode(
                frame_paths,
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=str(self.device) if torch.cuda.is_available() else "cpu",
            ).float().cpu()
            text_embed = self.clip_encoder.encode(
                [question],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=str(self.device) if torch.cuda.is_available() else "cpu",
            )[0].float().cpu()
        return full_feats, text_embed

    def _encode_with_long_clip(self, frame_paths: List[str], question: str):
        clip_device = self.device
        batch_size_clip = 512
        self.long_clip_model = self.long_clip_model.to(clip_device)

        with torch.no_grad():
            text_tokens = longclip.tokenize([question]).to(clip_device)
            text_embed = self.long_clip_model.encode_text(text_tokens)[0]
            text_embed = text_embed / (text_embed.norm() + 1e-6)
            text_embed = text_embed.detach().cpu().float()
            del text_tokens

            batches = []
            for i in range(0, len(frame_paths), batch_size_clip):
                batch_paths = frame_paths[i : i + batch_size_clip]
                images = []
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert("RGB")
                        images.append(self.long_clip_processor(image))
                    except Exception as exc:
                        eval_logger.warning(f"Skipping frame {path}: {exc}")
                if not images:
                    continue
                image_batch = torch.stack(images).to(clip_device)
                batch_feats = self.long_clip_model.encode_image(image_batch)
                batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)
                batches.append(batch_feats.detach().cpu().float())
                del image_batch
                del batch_feats

        full_feats = torch.cat(batches, dim=0) if batches else torch.empty((0, 768))
        return full_feats, text_embed

    def _encode_with_siglip(self, frame_paths: List[str], question: str):
        batch_size_siglip = 32
        self.siglip_model = self.siglip_model.to(self.device)

        with torch.no_grad():
            text_inputs = self.siglip_processor(
                text=question,
                padding="max_length",
                return_tensors="pt",
            ).to(self.siglip_model.device)
            text_outputs = self.siglip_model.get_text_features(input_ids=text_inputs["input_ids"])
            text_embed = text_outputs.mean(dim=0, keepdim=True)[0]
            text_embed = text_embed / (text_embed.norm() + 1e-6)
            text_embed = text_embed.detach().cpu().float()

            batches = []
            for i in range(0, len(frame_paths), batch_size_siglip):
                batch_paths = frame_paths[i : i + batch_size_siglip]
                images = []
                for path in batch_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                    except Exception as exc:
                        eval_logger.warning(f"Skipping frame {path}: {exc}")
                if not images:
                    continue
                image_inputs = self.siglip_processor(images=images, return_tensors="pt").to(self.siglip_model.device)
                batch_feats = self.siglip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
                batch_feats = batch_feats / (batch_feats.norm(dim=1, keepdim=True) + 1e-6)
                batches.append(batch_feats.detach().cpu().float())

        full_feats = torch.cat(batches, dim=0) if batches else torch.empty((0, 768))
        return full_feats, text_embed

    def _get_clip_features(self, frame_paths: List[str], question: str):
        cache_key = get_cache_key(frame_paths, question, self.frame_encoder_type)
        cache_path = os.path.join(self.clip_cache_dir, f"{cache_key}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            return cached["full_feats"], cached["text_embed"]

        if self.frame_encoder_type == "openai_clip":
            full_feats, text_embed = self._encode_with_openai_clip(frame_paths, question)
        elif self.frame_encoder_type == "long_clip":
            full_feats, text_embed = self._encode_with_long_clip(frame_paths, question)
        else:
            full_feats, text_embed = self._encode_with_siglip(frame_paths, question)

        torch.save(
            {
                "full_feats": full_feats,
                "text_embed": text_embed,
                "clip_type": self.frame_encoder_type,
                "question": question,
            },
            cache_path,
        )
        return full_feats, text_embed

    def _build_fallback_frame_metadata(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        frame_metadata = []
        for frame_idx, path in enumerate(frame_paths):
            with Image.open(path) as img:
                width, height = img.size
            frame_metadata.append(
                {
                    "frame_index": frame_idx,
                    "path": path,
                    "token_count": None,
                    "width": int(width),
                    "height": int(height),
                    "resolution": f"{int(width)}x{int(height)}",
                    "original_width": int(width),
                    "original_height": int(height),
                }
            )
        return frame_metadata

    def _select_images_for_video(self, video_path: str, frame_paths: List[str], question: str) -> Tuple[List[Image.Image], Dict[str, Any]]:
        full_feats, text_embed = self._get_clip_features(frame_paths, question)
        if full_feats.numel() == 0:
            fallback = frame_paths[: self.max_num_frames]
            return [Image.open(path).convert("RGB") for path in fallback], {
                "video_path": video_path,
                "frame_source": os.path.dirname(fallback[0]) if fallback else None,
                "num_candidate_frames": len(frame_paths),
                "selection_strategy": "fallback_first_n_without_clip_features",
                "selected_frame_indices": list(range(len(fallback))),
                "selected_frame_paths": fallback,
                "selected_frames": self._build_fallback_frame_metadata(fallback),
            }

        selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
            frame_features=full_feats,
            frame_embeds=full_feats,
            text_embed=text_embed,
            total_token_budget=self.max_num_frames * self.target_token_per_frame,
            min_token_per_frame=self.min_token_per_frame,
            max_token_per_frame=self.max_token_per_frame,
            target_token_per_frame=self.target_token_per_frame,
        )
        sorted_pairs = sorted(zip(selected_idx.tolist(), selected_resolution.tolist()), key=lambda x: x[0])
        selected_indices = [idx for idx, _ in sorted_pairs]
        selected_tokens = [tok for _, tok in sorted_pairs]
        selected_frames = [frame_paths[idx] for idx in selected_indices]
        selected_images, selected_frame_metadata = load_and_resize_images(selected_frames, selected_tokens)
        for frame_metadata, frame_idx in zip(selected_frame_metadata, selected_indices):
            frame_metadata["frame_index"] = int(frame_idx)
        return selected_images, {
            "video_path": video_path,
            "frame_source": os.path.dirname(frame_paths[0]) if frame_paths else None,
            "num_candidate_frames": len(frame_paths),
            "selection_strategy": "cdpruner_dpp_dynamic_resolution",
            "selected_frame_indices": selected_indices,
            "selected_frame_paths": selected_frames,
            "selected_frames": selected_frame_metadata,
        }

    def _build_openai_messages(self, chat_messages: ChatMessages) -> Tuple[List[dict], Dict[str, Any]]:
        question = self._extract_question_from_messages(chat_messages)
        openai_messages = []
        video_counter = 0
        video_paths = chat_messages.extract_media()[1]
        selected_images_per_video = []
        selected_metadata_per_video = []

        for video_path in video_paths:
            frame_paths = self._resolve_frame_paths(video_path)
            selected_images, selected_metadata = self._select_images_for_video(video_path, frame_paths, question)
            selected_images_per_video.append(selected_images)
            selected_metadata_per_video.append(selected_metadata)

        for message in chat_messages.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    base64_img, mime_type = self._encode_image_with_mime(content.url)
                    openai_message["content"].append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}
                    )
                elif content.type == "video":
                    selected_images = selected_images_per_video[video_counter]
                    video_counter += 1
                    for image in selected_images:
                        base64_img, mime_type = self._encode_image_with_mime(image)
                        openai_message["content"].append(
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}
                        )
            openai_messages.append(openai_message)

        return openai_messages, {
            "question": question,
            "frame_encoder_type": self.frame_encoder_type,
            "max_num_frames": self.max_num_frames,
            "target_token_per_frame": self.target_token_per_frame,
            "min_token_per_frame": self.min_token_per_frame,
            "max_token_per_frame": self.max_token_per_frame,
            "videos": selected_metadata_per_video,
        }

    def _encode_image_with_mime(self, image):
        if isinstance(image, str):
            with open(image, "rb") as handle:
                byte_data = handle.read()
            suffix = os.path.splitext(image)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")
            return base64.b64encode(byte_data).decode("utf-8"), mime_type

        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode("utf-8"), "image/png"

    def _build_payload(self, chat_messages: ChatMessages, gen_kwargs: dict) -> Tuple[dict, Dict[str, Any]]:
        openai_messages, sample_metadata = self._build_openai_messages(chat_messages)
        payload = {
            "model": self.model_version,
            "messages": openai_messages,
        }

        max_new_tokens = min(gen_kwargs.get("max_new_tokens", 1024), 4096)
        temperature = gen_kwargs.get("temperature", 0)

        if any(tag in self.model_version for tag in ("o1", "o3", "o4", "gpt-5")):
            # Reasoning models can consume the task's tiny `max_new_tokens` budget
            # internally and end up returning an empty visible answer. Give them a
            # safer completion budget for evaluation tasks that expect a short label.
            payload["max_completion_tokens"] = max(
                max_new_tokens,
                self.min_completion_tokens_for_reasoning,
            )
        else:
            payload["max_tokens"] = max_new_tokens
            payload["temperature"] = temperature

        return payload, sample_metadata

    def _extract_response_text(self, response) -> str:
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
            return "".join(parts)
        return str(content) if content is not None else ""

    def _process_single_request(self, payload):
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**payload)
                end_time = time.time()
                response_text = self._extract_response_text(response)
                latency = end_time - start_time
                tokens = 0
                if hasattr(response, "usage") and response.usage is not None:
                    tokens = getattr(response.usage, "completion_tokens", 0) or 0
                return response_text or "", latency, tokens
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {exc}")
                    return "", 0, 0
                eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {exc}")
                time.sleep(self.timeout)
        return "", 0, 0

    def generate_until(self, requests) -> List[str]:
        res = []

        def _collate(x):
            return x.args[0], x.args[0]

        re_ords = utils.Collator(
            requests,
            _collate,
            group_fn=lambda x: x.args[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            batch_payloads = []
            batch_gen_kwargs = []
            batch_doc_uuids = []
            batch_responses = []

            for req in chunk:
                _, doc_to_messages, gen_kwargs, doc_id, task, split = req.args
                doc_uuid = f"{task}___{split}___{doc_id}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode is True and self.cache_mode == "resume" and doc_uuid in self.response_cache:
                    cached = self.response_cache[doc_uuid]
                    if cached:
                        batch_responses.append(cached)
                        batch_payloads.append(None)
                        batch_gen_kwargs.append(gen_kwargs)
                        continue

                raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages = ChatMessages(**{"messages": raw_messages})
                payload, sample_metadata = self._build_payload(chat_messages, gen_kwargs)
                batch_payloads.append(payload)
                batch_gen_kwargs.append(gen_kwargs)
                batch_responses.append(None)
                req.custom_metadata = sample_metadata

            tasks_to_run = [(payload, idx) for idx, payload in enumerate(batch_payloads) if payload is not None]
            if tasks_to_run:
                max_workers = min(len(tasks_to_run), 8)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._process_single_request, payload): idx
                        for payload, idx in tasks_to_run
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        response_text, latency, tokens = future.result()
                        batch_responses[idx] = response_text
                        e2e_latency += latency
                        total_tokens += tokens

            for response_text, doc_uuid in zip(batch_responses, batch_doc_uuids):
                clean_ans = parse_reasoning_model_answer(response_text or "")
                res.append(clean_ans)
                if self.continual_mode is True:
                    self.response_cache[doc_uuid] = clean_ans

            if self.continual_mode is True:
                with open(self.response_persistent_file, "w") as handle:
                    json.dump(self.response_cache, handle)

            pbar.update(1)

        res = re_ords.get_original(res)
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=avg_speed,
        )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for gpt_chat_wo_ours_v4")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for gpt_chat_wo_ours_v4")
