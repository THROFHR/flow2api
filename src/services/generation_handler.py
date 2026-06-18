"""Generation handler for Flow2API"""
import asyncio
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Dict, Any, Tuple, Callable, Awaitable
from ..core.logger import debug_logger
from ..core.config import config
from ..core.monitoring import record_generation_result
from ..core.models import Task, RequestLog
from ..core.account_tiers import (
    PAYGATE_TIER_NOT_PAID,
    get_paygate_tier_label,
    get_required_paygate_tier_for_model,
    normalize_user_paygate_tier,
    supports_model_for_tier,
)
from .file_cache import FileCache


# Model configuration
MODEL_CONFIG = {
    # 图片生成 - GEM_PIX_2 (Gemini 3.0 Pro)
    "gemini-3.0-pro-image-landscape": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
    },
    "gemini-3.0-pro-image-portrait": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"
    },
    "gemini-3.0-pro-image-square": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE"
    },
    "gemini-3.0-pro-image-four-three": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE"
    },
    "gemini-3.0-pro-image-three-four": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR"
    },

    # 图片生成 - GEM_PIX_2 (Gemini 3.0 Pro) 2K 放大版
    "gemini-3.0-pro-image-landscape-2k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.0-pro-image-portrait-2k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.0-pro-image-square-2k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.0-pro-image-four-three-2k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.0-pro-image-three-four-2k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },

    # 图片生成 - GEM_PIX_2 (Gemini 3.0 Pro) 4K 放大版
    "gemini-3.0-pro-image-landscape-4k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.0-pro-image-portrait-4k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.0-pro-image-square-4k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.0-pro-image-four-three-4k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.0-pro-image-three-four-4k": {
        "type": "image",
        "model_name": "GEM_PIX_2",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },

    # 图片生成 - IMAGEN_3_5 (Imagen 4.0)
    "imagen-4.0-generate-preview-landscape": {
        "type": "image",
        "model_name": "IMAGEN_3_5",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
    },
    "imagen-4.0-generate-preview-portrait": {
        "type": "image",
        "model_name": "IMAGEN_3_5",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"
    },

    # 图片生成 - NARWHAL (新版)
    "gemini-3.1-flash-image-landscape": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE"
    },
    "gemini-3.1-flash-image-portrait": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT"
    },
    "gemini-3.1-flash-image-square": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE"
    },
    "gemini-3.1-flash-image-four-three": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE"
    },
    "gemini-3.1-flash-image-three-four": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR"
    },
    "gemini-3.1-flash-image-landscape-2k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.1-flash-image-portrait-2k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.1-flash-image-square-2k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.1-flash-image-four-three-2k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.1-flash-image-three-four-2k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_2K"
    },
    "gemini-3.1-flash-image-landscape-4k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.1-flash-image-portrait-4k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.1-flash-image-square-4k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_SQUARE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.1-flash-image-four-three-4k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },
    "gemini-3.1-flash-image-three-four-4k": {
        "type": "image",
        "model_name": "NARWHAL",
        "aspect_ratio": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
        "upsample": "UPSAMPLE_IMAGE_RESOLUTION_4K"
    },

    # ========== 文生视频 (T2V - Text to Video) ==========
    # 不支持上传图片，只使用文本提示词生成

    # veo_3_1_t2v_fast_portrait (竖屏)
    # 上游模型名: veo_3_1_t2v_fast_portrait
    "veo_3_1_t2v_fast_portrait": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False
    },
    # veo_3_1_t2v_fast_landscape (横屏)
    # 上游模型名: veo_3_1_t2v_fast
    "veo_3_1_t2v_fast_landscape": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False
    },

    # veo_3_1_t2v_fast_ultra (横竖屏)
    "veo_3_1_t2v_fast_portrait_ultra": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False
    },
    "veo_3_1_t2v_fast_ultra": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False
    },

    # veo_3_1_t2v_fast_ultra_relaxed (横竖屏)
    "veo_3_1_t2v_fast_portrait_ultra_relaxed": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False
    },
    "veo_3_1_t2v_fast_ultra_relaxed": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False
    },

    # veo_3_1_t2v (横竖屏)
    "veo_3_1_t2v_portrait": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False
    },
    "veo_3_1_t2v_landscape": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False
    },
    # veo_3_1_t2v_lite (横竖屏，来自 labs.google.har)
    "veo_3_1_t2v_lite_portrait": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },
    "veo_3_1_t2v_lite_landscape": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },

    # ========== 首尾帧模型 (I2V - Image to Video) ==========
    # 支持1-2张图片：1张作为首帧，2张作为首尾帧

    # veo_3_1_i2v_s_fast_fl (需要新增横竖屏)
    "veo_3_1_i2v_s_fast_portrait_fl": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_portrait_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },
    "veo_3_1_i2v_s_fast_fl": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },

    # veo_3_1_i2v_s_fast_ultra (横竖屏)
    "veo_3_1_i2v_s_fast_portrait_ultra_fl": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_portrait_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },
    "veo_3_1_i2v_s_fast_ultra_fl": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },

    # veo_3_1_i2v_s_fast_ultra_relaxed (需要新增横竖屏)
    "veo_3_1_i2v_s_fast_portrait_ultra_relaxed": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_portrait_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },
    "veo_3_1_i2v_s_fast_ultra_relaxed": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },

    # veo_3_1_i2v_s (需要新增横竖屏)
    "veo_3_1_i2v_s_portrait": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },
    "veo_3_1_i2v_s_landscape": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2
    },
    # veo_3_1_i2v_lite (横竖屏，仅首帧，来自 labs.google.har)
    "veo_3_1_i2v_lite_portrait": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 1,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },
    "veo_3_1_i2v_lite_landscape": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 1,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },
    # veo_3_1_interpolation_lite (横竖屏，首尾帧，来自 labs.google.har)
    "veo_3_1_interpolation_lite_portrait": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_interpolation_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 2,
        "max_images": 2,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },
    "veo_3_1_interpolation_lite_landscape": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_interpolation_lite",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 2,
        "max_images": 2,
        "use_v2_model_config": True,
        "allow_tier_upgrade": False
    },

    # ========== 多图生成 (R2V - Reference Images to Video) ==========
    # 当前上游协议最多支持 3 张参考图

    # veo_3_1_r2v_fast (横竖屏)
    "veo_3_1_r2v_fast_portrait": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_portrait",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },
    "veo_3_1_r2v_fast": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_landscape",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },

    # veo_3_1_r2v_fast_ultra (横竖屏)
    "veo_3_1_r2v_fast_portrait_ultra": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },
    "veo_3_1_r2v_fast_ultra": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_landscape_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },

    # veo_3_1_r2v_fast_ultra_relaxed (横竖屏)
    "veo_3_1_r2v_fast_portrait_ultra_relaxed": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_portrait_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },
    "veo_3_1_r2v_fast_ultra_relaxed": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_landscape_ultra_relaxed",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3
    },

    # ========== 视频放大 (Video Upsampler) ==========
    # 仅 3.1 支持，需要先生成视频后再放大，可能需要 30 分钟

    # T2V 4K 放大版
    "veo_3_1_t2v_fast_portrait_4k": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },
    "veo_3_1_t2v_fast_4k": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },
    "veo_3_1_t2v_fast_portrait_ultra_4k": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },
    "veo_3_1_t2v_fast_ultra_4k": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },

    # T2V 1080P 放大版
    "veo_3_1_t2v_fast_portrait_1080p": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },
    "veo_3_1_t2v_fast_1080p": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },
    "veo_3_1_t2v_fast_portrait_ultra_1080p": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },
    "veo_3_1_t2v_fast_ultra_1080p": {
        "type": "video",
        "video_type": "t2v",
        "model_key": "veo_3_1_t2v_fast_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },

    # I2V 4K 放大版
    "veo_3_1_i2v_s_fast_portrait_ultra_fl_4k": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_portrait_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },
    "veo_3_1_i2v_s_fast_ultra_fl_4k": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },

    # I2V 1080P 放大版
    "veo_3_1_i2v_s_fast_portrait_ultra_fl_1080p": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_portrait_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },
    "veo_3_1_i2v_s_fast_ultra_fl_1080p": {
        "type": "video",
        "video_type": "i2v",
        "model_key": "veo_3_1_i2v_s_fast_ultra_fl",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 1,
        "max_images": 2,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },

    # R2V 4K 放大版
    "veo_3_1_r2v_fast_portrait_ultra_4k": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },
    "veo_3_1_r2v_fast_ultra_4k": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_landscape_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3,
        "upsample": {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
    },

    # R2V 1080P 放大版
    "veo_3_1_r2v_fast_portrait_ultra_1080p": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },
    "veo_3_1_r2v_fast_ultra_1080p": {
        "type": "video",
        "video_type": "r2v",
        "model_key": "veo_3_1_r2v_fast_landscape_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": True,
        "min_images": 0,
        "max_images": 3,
        "upsample": {"resolution": "VIDEO_RESOLUTION_1080P", "model_key": "veo_3_1_upsampler_1080p"}
    },

    # ========== 视频续写 (Extend - Video Continuation) ==========
    # 基于已生成的视频续写7秒，最多续写20次（最长148秒）
    # 需要提供源视频的 mediaGenerationId

    # VEO 3.1 Extend (横竖屏)
    "veo_3_1_extend_portrait": {
        "type": "video",
        "video_type": "extend",
        "model_key": "veo_3_1_extend_fast_portrait_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_PORTRAIT",
        "supports_images": False,
        "requires_video_id": True,
    },
    "veo_3_1_extend": {
        "type": "video",
        "video_type": "extend",
        "model_key": "veo_3_1_extend_fast_ultra",
        "aspect_ratio": "VIDEO_ASPECT_RATIO_LANDSCAPE",
        "supports_images": False,
        "requires_video_id": True,
    },
}


def _make_t2v_config(
    model_key: str,
    aspect_ratio: str,
    *,
    use_v2_model_config: bool = False,
    allow_tier_upgrade: bool = True,
    upsample: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "type": "video",
        "video_type": "t2v",
        "model_key": model_key,
        "aspect_ratio": aspect_ratio,
        "supports_images": False,
    }
    if use_v2_model_config:
        cfg["use_v2_model_config"] = True
    if not allow_tier_upgrade:
        cfg["allow_tier_upgrade"] = False
    if upsample:
        cfg["upsample"] = upsample
    return cfg


def _make_i2v_config(
    model_key: str,
    aspect_ratio: str,
    *,
    min_images: int = 1,
    max_images: int = 2,
    use_v2_model_config: bool = False,
    allow_tier_upgrade: bool = True,
    upsample: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "type": "video",
        "video_type": "i2v",
        "model_key": model_key,
        "aspect_ratio": aspect_ratio,
        "supports_images": True,
        "min_images": min_images,
        "max_images": max_images,
    }
    if use_v2_model_config:
        cfg["use_v2_model_config"] = True
    if not allow_tier_upgrade:
        cfg["allow_tier_upgrade"] = False
    if upsample:
        cfg["upsample"] = upsample
    return cfg


def _apply_veo_3_1_model_updates():
    """Keep the public aliases aligned with the current Veo 3.1 model families."""
    landscape = "VIDEO_ASPECT_RATIO_LANDSCAPE"
    portrait = "VIDEO_ASPECT_RATIO_PORTRAIT"

    def add_alias(alias: str, target: str):
        MODEL_CONFIG[alias] = dict(MODEL_CONFIG[target])

    # Non-fast/non-lite Veo 3.1 aliases must call Quality upstream keys.
    MODEL_CONFIG["veo_3_1_t2v_landscape"].update({"model_key": "veo_3_1_t2v"})
    MODEL_CONFIG["veo_3_1_t2v_portrait"].update({"model_key": "veo_3_1_t2v_portrait"})
    MODEL_CONFIG["veo_3_1_i2v_s_landscape"].update({"model_key": "veo_3_1_i2v_s_fl"})
    MODEL_CONFIG["veo_3_1_i2v_s_portrait"].update({"model_key": "veo_3_1_i2v_s_portrait_fl"})
    MODEL_CONFIG["veo_3_1_extend"].update({"model_key": "veo_3_1_extend_landscape"})
    MODEL_CONFIG["veo_3_1_extend_portrait"].update({"model_key": "veo_3_1_extend_portrait"})

    for seconds in (4, 6):
        suffix = f"{seconds}s"

        # T2V duration variants.
        MODEL_CONFIG[f"veo_3_1_t2v_fast_{suffix}"] = _make_t2v_config(
            f"veo_3_1_t2v_fast_{suffix}", landscape
        )
        MODEL_CONFIG[f"veo_3_1_t2v_fast_portrait_{suffix}"] = _make_t2v_config(
            f"veo_3_1_t2v_fast_{suffix}", portrait
        )
        MODEL_CONFIG[f"veo_3_1_t2v_lite_{suffix}_landscape"] = _make_t2v_config(
            f"veo_3_1_t2v_lite_{suffix}",
            landscape,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_t2v_lite_{suffix}_portrait"] = _make_t2v_config(
            f"veo_3_1_t2v_lite_{suffix}",
            portrait,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_t2v_{suffix}"] = _make_t2v_config(
            f"veo_3_1_t2v_quality_{suffix}", landscape
        )
        MODEL_CONFIG[f"veo_3_1_t2v_portrait_{suffix}"] = _make_t2v_config(
            f"veo_3_1_t2v_quality_{suffix}", portrait
        )

        # I2V duration variants. FL keys are used for 2 images; the single-image path strips "_fl".
        MODEL_CONFIG[f"veo_3_1_i2v_s_fast_{suffix}_fl"] = _make_i2v_config(
            f"veo_3_1_i2v_s_fast_{suffix}_fl", landscape
        )
        MODEL_CONFIG[f"veo_3_1_i2v_s_fast_portrait_{suffix}_fl"] = _make_i2v_config(
            f"veo_3_1_i2v_s_fast_{suffix}_fl", portrait
        )
        MODEL_CONFIG[f"veo_3_1_i2v_lite_{suffix}_landscape"] = _make_i2v_config(
            f"veo_3_1_i2v_s_lite_{suffix}",
            landscape,
            min_images=1,
            max_images=1,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_i2v_lite_{suffix}_portrait"] = _make_i2v_config(
            f"veo_3_1_i2v_s_lite_{suffix}",
            portrait,
            min_images=1,
            max_images=1,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_interpolation_lite_{suffix}_landscape"] = _make_i2v_config(
            f"veo_3_1_i2v_s_lite_{suffix}_fl",
            landscape,
            min_images=2,
            max_images=2,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_interpolation_lite_{suffix}_portrait"] = _make_i2v_config(
            f"veo_3_1_i2v_s_lite_{suffix}_fl",
            portrait,
            min_images=2,
            max_images=2,
            use_v2_model_config=True,
            allow_tier_upgrade=False,
        )
        MODEL_CONFIG[f"veo_3_1_i2v_s_{suffix}"] = _make_i2v_config(
            f"veo_3_1_i2v_s_quality_{suffix}_fl", landscape
        )
        MODEL_CONFIG[f"veo_3_1_i2v_s_portrait_{suffix}"] = _make_i2v_config(
            f"veo_3_1_i2v_s_quality_{suffix}_fl", portrait
        )

        for resolution_name, resolution, upsampler_model_key in (
            ("4k", "VIDEO_RESOLUTION_4K", "veo_3_1_upsampler_4k"),
            ("1080p", "VIDEO_RESOLUTION_1080P", "veo_3_1_upsampler_1080p"),
        ):
            upsample = {"resolution": resolution, "model_key": upsampler_model_key}
            MODEL_CONFIG[f"veo_3_1_t2v_{suffix}_{resolution_name}"] = _make_t2v_config(
                f"veo_3_1_t2v_quality_{suffix}", landscape, upsample=upsample
            )
            MODEL_CONFIG[f"veo_3_1_t2v_portrait_{suffix}_{resolution_name}"] = _make_t2v_config(
                f"veo_3_1_t2v_quality_{suffix}", portrait, upsample=upsample
            )
            MODEL_CONFIG[f"veo_3_1_i2v_s_{suffix}_{resolution_name}"] = _make_i2v_config(
                f"veo_3_1_i2v_s_quality_{suffix}_fl", landscape, upsample=upsample
            )
            MODEL_CONFIG[f"veo_3_1_i2v_s_portrait_{suffix}_{resolution_name}"] = _make_i2v_config(
                f"veo_3_1_i2v_s_quality_{suffix}_fl", portrait, upsample=upsample
            )

    for resolution_name, resolution, upsampler_model_key in (
        ("4k", "VIDEO_RESOLUTION_4K", "veo_3_1_upsampler_4k"),
        ("1080p", "VIDEO_RESOLUTION_1080P", "veo_3_1_upsampler_1080p"),
    ):
        upsample = {"resolution": resolution, "model_key": upsampler_model_key}
        MODEL_CONFIG[f"veo_3_1_t2v_{resolution_name}"] = _make_t2v_config(
            "veo_3_1_t2v", landscape, upsample=upsample
        )
        MODEL_CONFIG[f"veo_3_1_t2v_portrait_{resolution_name}"] = _make_t2v_config(
            "veo_3_1_t2v_portrait", portrait, upsample=upsample
        )
        MODEL_CONFIG[f"veo_3_1_i2v_s_{resolution_name}"] = _make_i2v_config(
            "veo_3_1_i2v_s_fl", landscape, upsample=upsample
        )
        MODEL_CONFIG[f"veo_3_1_i2v_s_portrait_{resolution_name}"] = _make_i2v_config(
            "veo_3_1_i2v_s_portrait_fl", portrait, upsample=upsample
        )

    for seconds in (4, 6):
        suffix = f"{seconds}s"

        # Explicit landscape names for /v1/models; short landscape names remain compatible.
        add_alias(f"veo_3_1_t2v_fast_landscape_{suffix}", f"veo_3_1_t2v_fast_{suffix}")
        add_alias(f"veo_3_1_t2v_landscape_{suffix}", f"veo_3_1_t2v_{suffix}")
        add_alias(f"veo_3_1_i2v_s_fast_landscape_{suffix}_fl", f"veo_3_1_i2v_s_fast_{suffix}_fl")
        add_alias(f"veo_3_1_i2v_s_landscape_{suffix}", f"veo_3_1_i2v_s_{suffix}")

        add_alias(f"veo_3_1_t2v_lite_landscape_{suffix}", f"veo_3_1_t2v_lite_{suffix}_landscape")
        add_alias(f"veo_3_1_t2v_lite_portrait_{suffix}", f"veo_3_1_t2v_lite_{suffix}_portrait")
        add_alias(f"veo_3_1_i2v_lite_landscape_{suffix}", f"veo_3_1_i2v_lite_{suffix}_landscape")
        add_alias(f"veo_3_1_i2v_lite_portrait_{suffix}", f"veo_3_1_i2v_lite_{suffix}_portrait")
        add_alias(
            f"veo_3_1_interpolation_lite_landscape_{suffix}",
            f"veo_3_1_interpolation_lite_{suffix}_landscape",
        )
        add_alias(
            f"veo_3_1_interpolation_lite_portrait_{suffix}",
            f"veo_3_1_interpolation_lite_{suffix}_portrait",
        )

        for resolution_name in ("4k", "1080p"):
            add_alias(
                f"veo_3_1_t2v_landscape_{suffix}_{resolution_name}",
                f"veo_3_1_t2v_{suffix}_{resolution_name}",
            )
            add_alias(
                f"veo_3_1_i2v_s_landscape_{suffix}_{resolution_name}",
                f"veo_3_1_i2v_s_{suffix}_{resolution_name}",
            )

    for resolution_name in ("4k", "1080p"):
        add_alias(f"veo_3_1_t2v_landscape_{resolution_name}", f"veo_3_1_t2v_{resolution_name}")
        add_alias(f"veo_3_1_i2v_s_landscape_{resolution_name}", f"veo_3_1_i2v_s_{resolution_name}")

    add_alias("veo_3_1_r2v_fast_landscape", "veo_3_1_r2v_fast")
    add_alias("veo_3_1_r2v_fast_landscape_ultra", "veo_3_1_r2v_fast_ultra")
    add_alias("veo_3_1_r2v_fast_landscape_ultra_relaxed", "veo_3_1_r2v_fast_ultra_relaxed")
    add_alias("veo_3_1_r2v_fast_landscape_ultra_4k", "veo_3_1_r2v_fast_ultra_4k")
    add_alias("veo_3_1_r2v_fast_landscape_ultra_1080p", "veo_3_1_r2v_fast_ultra_1080p")


_apply_veo_3_1_model_updates()


def _known_video_model_keys() -> set[str]:
    return {
        cfg["model_key"]
        for cfg in MODEL_CONFIG.values()
        if cfg.get("type") == "video" and cfg.get("model_key")
    }


def _resolve_tier_two_model_key(model_key: str) -> str:
    """Only upgrade to an ultra key when that exact upstream key is known valid."""
    if "ultra" in model_key:
        return model_key
    if "_fl" in model_key:
        candidate = model_key.replace("_fl", "_ultra_fl")
    else:
        candidate = model_key + "_ultra"
    return candidate if candidate in _known_video_model_keys() else model_key


class GenerationHandler:
    """统一生成处理器"""

    def __init__(self, flow_client, token_manager, load_balancer, db, concurrency_manager, proxy_manager):
        cache_dir = Path(__file__).resolve().parents[2] / "tmp"
        self.flow_client = flow_client
        self.token_manager = token_manager
        self.load_balancer = load_balancer
        self.db = db
        self.concurrency_manager = concurrency_manager
        self.file_cache = FileCache(
            cache_dir=str(cache_dir),
            default_timeout=config.cache_timeout,
            proxy_manager=proxy_manager,
            flow_client=flow_client,
        )

    def _create_generation_result(self) -> Dict[str, Any]:
        """????????????????"""
        return dict(
            success=False,
            error_message=None,
            error_extra=None,
            error_emitted=False,
            usage_success=False,
            result_status="failed",
        )

    def _create_response_state(self) -> Dict[str, Any]:
        """为单次请求创建独立的响应状态，避免并发请求互相污染。"""
        return {
            "url": None,
            "generated_assets": None,
            "base_url": None,
            "debug_enabled": False,
            "debug_include_media_check": True,
        }

    def _mark_generation_failed(
        self,
        generation_result: Optional[Dict[str, Any]],
        error_message: str,
        error_extra: Optional[Dict[str, Any]] = None,
    ):
        """????????????????????"""
        if isinstance(generation_result, dict):
            generation_result["success"] = False
            generation_result["error_message"] = error_message
            generation_result["error_extra"] = error_extra
            generation_result["error_emitted"] = True
            generation_result["usage_success"] = False
            generation_result["result_status"] = "failed"

    def _mark_generation_succeeded(
        self,
        generation_result: Optional[Dict[str, Any]],
        *,
        usage_success: bool = True,
        result_status: str = "success",
        message: Optional[str] = None,
    ):
        """???????"""
        if isinstance(generation_result, dict):
            generation_result["success"] = True
            generation_result["error_message"] = message
            generation_result["error_emitted"] = False
            generation_result["usage_success"] = usage_success
            generation_result["result_status"] = result_status

    def _normalize_error_message(self, error_message: Any, max_length: int = 1000) -> str:
        """归一化错误文本，避免写入超长内容。"""
        text = str(error_message or "").strip() or "未知错误"
        if len(text) <= max_length:
            return text
        return f"{text[:max_length - 3]}..."

    def _truncate_debug_text(self, value: Any, max_length: int = 240) -> str:
        text = str(value or "")
        if len(text) <= max_length:
            return text
        return f"{text[:max_length]}... (truncated)"

    def _mask_debug_token(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if len(text) <= 16:
            return "***"
        return f"{text[:8]}...{text[-6:]}"

    def _summarize_debug_payload(self, data: Any, depth: int = 0) -> Any:
        if depth > 4:
            return self._truncate_debug_text(data, max_length=120)
        if isinstance(data, dict):
            result: Dict[str, Any] = {}
            for key, value in data.items():
                key_lower = str(key).lower()
                if key_lower in {
                    "authorization",
                    "cookie",
                    "token",
                    "recaptchatoken",
                    "grecaptcharesponse",
                    "clientkey",
                    "apikey",
                    "api_key",
                    "access_token",
                    "st_token",
                    "at_token",
                }:
                    result[key] = self._mask_debug_token(value)
                    continue
                if key_lower in {
                    "imagebytes",
                    "rawimagebytes",
                    "encodedimage",
                    "base64",
                    "data",
                } and isinstance(value, str):
                    result[key] = f"<{key} length={len(value)}>"
                    continue
                result[key] = self._summarize_debug_payload(value, depth + 1)
            return result
        if isinstance(data, list):
            if len(data) > 8:
                preview = [self._summarize_debug_payload(item, depth + 1) for item in data[:8]]
                preview.append(f"... ({len(data)} items total)")
                return preview
            return [self._summarize_debug_payload(item, depth + 1) for item in data]
        if isinstance(data, str):
            return self._truncate_debug_text(data, max_length=240)
        return data

    def _format_debug_json(self, data: Any) -> str:
        summarized = self._summarize_debug_payload(data)
        try:
            return json.dumps(summarized, ensure_ascii=False, indent=2)
        except Exception:
            return self._truncate_debug_text(summarized, max_length=500)

    def _format_debug_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    async def _emit_debug_event(
        self,
        response_state: Optional[Dict[str, Any]],
        stream: bool,
        event: str,
        title: str,
        payload: Any = None,
    ) -> Optional[str]:
        if not isinstance(response_state, dict) or not response_state.get("debug_enabled"):
            return None

        lines = [f"[{self._format_debug_timestamp()}] [DEBUG] {title}"]
        if payload is not None:
            lines.append(self._format_debug_json(payload))
        text = "\n".join(lines) + "\n"

        if stream:
            return self._create_stream_chunk(text)
        return text

    async def _emit_debug_progress(
        self,
        response_state: Optional[Dict[str, Any]],
        stream: bool,
        request_log_state: Optional[Dict[str, Any]],
        token_id: Optional[int],
        event: str,
        title: str,
        payload: Any = None,
        progress: Optional[int] = None,
    ) -> Optional[str]:
        if progress is not None:
            await self._update_request_log_progress(
                request_log_state,
                token_id=token_id,
                status_text=f"debug_{event}",
                progress=progress,
                response_extra={"debug": self._summarize_debug_payload(payload) if payload is not None else {"event": event}},
            )
        return await self._emit_debug_event(
            response_state=response_state,
            stream=stream,
            event=event,
            title=title,
            payload=payload,
        )

    def _is_video_url_candidate(self, value: Any) -> bool:
        return isinstance(value, str) and (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("/")
        )

    def _find_nested_video_url(self, value: Any, keys: Tuple[str, ...]) -> Optional[str]:
        if isinstance(value, dict):
            for key in keys:
                candidate = value.get(key)
                if self._is_video_url_candidate(candidate):
                    return candidate.strip()
            for candidate in value.values():
                found = self._find_nested_video_url(candidate, keys)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_nested_video_url(item, keys)
                if found:
                    return found
        return None

    def _extract_video_url_from_operation(self, operation: Dict[str, Any]) -> Optional[str]:
        return self._find_nested_video_url(
            operation,
            ("fifeUrl", "servingUri", "videoUrl", "outputUri", "downloadUri", "url", "uri"),
        )

    def _find_nested_string_value(self, value: Any, key: str) -> Optional[str]:
        if isinstance(value, dict):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            for nested in value.values():
                found = self._find_nested_string_value(nested, key)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_nested_string_value(item, key)
                if found:
                    return found
        return None

    def _save_encoded_video_to_tmp(
        self,
        encoded_video: str,
        media_id: Optional[str],
        response_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not encoded_video:
            return None

        safe_media_id = "".join(
            ch for ch in str(media_id or "video") if ch.isalnum() or ch in ("-", "_")
        )[:80] or "video"
        tmp_dir = Path(__file__).resolve().parents[2] / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        filename = f"encoded_video_{safe_media_id}_{int(time.time())}.mp4"
        file_path = tmp_dir / filename
        file_path.write_bytes(base64.b64decode(encoded_video))
        return f"{self._get_base_url(response_state)}/tmp/{filename}"

    def _extract_video_details_from_operation(
        self,
        operation: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        operation_body = operation.get("operation") if isinstance(operation.get("operation"), dict) else {}
        metadata = operation_body.get("metadata") if isinstance(operation_body.get("metadata"), dict) else {}
        video_info = metadata.get("video") if isinstance(metadata.get("video"), dict) else {}
        video_url = video_info.get("fifeUrl") or self._extract_video_url_from_operation(operation)
        return metadata, video_info, video_url

    def _extract_media_id_from_operation(
        self,
        operation: Dict[str, Any],
        video_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        operation_body = operation.get("operation") if isinstance(operation.get("operation"), dict) else {}
        candidate = (
            (video_info or {}).get("mediaGenerationId")
            or operation.get("mediaName")
            or operation.get("name")
            or operation_body.get("name")
        )
        return candidate.strip() if isinstance(candidate, str) and candidate.strip() else None

    async def _wait_for_video_url_after_success(
        self,
        token,
        operations: List[Dict[str, Any]],
        initial_operation: Dict[str, Any],
        poll_interval: float,
        request_log_state: Optional[Dict[str, Any]] = None,
        response_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
        latest_operation = initial_operation
        latest_metadata, latest_video_info, latest_video_url = self._extract_video_details_from_operation(latest_operation)
        retry_operations = [latest_operation] if latest_operation else operations
        max_url_attempts = max(1, min(10, int(10 / max(float(poll_interval), 1.0))))

        for attempt in range(max_url_attempts):
            if latest_video_url:
                break

            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="video_url_pending",
                progress=90,
                response_extra={
                    "mediaGenerationId": latest_video_info.get("mediaGenerationId"),
                    "url_retry_attempt": attempt + 1,
                    "url_retry_max": max_url_attempts,
                },
            )
            await asyncio.sleep(poll_interval)

            media_id = self._extract_media_id_from_operation(latest_operation, latest_video_info)
            if media_id:
                try:
                    media_result = await self.flow_client.get_media(token.at, media_id)
                    media_url = self._extract_video_url_from_operation(media_result)
                    if not media_url:
                        encoded_video = self._find_nested_string_value(media_result, "encodedVideo")
                        media_url = self._save_encoded_video_to_tmp(
                            encoded_video or "",
                            media_id,
                            response_state,
                        )
                    if media_url:
                        latest_video_url = media_url
                        latest_operation = {
                            **latest_operation,
                            "mediaLookup": media_result,
                        }
                        latest_metadata, latest_video_info, _ = self._extract_video_details_from_operation(latest_operation)
                        break
                    latest_operation = {
                        **latest_operation,
                        "mediaLookup": media_result,
                    }
                except Exception as e:
                    debug_logger.log_warning(f"[VIDEO URL] get_media failed for {media_id}: {e}")

            result = await self.flow_client.check_video_status(token.at, retry_operations)
            checked_operations = result.get("operations", [])
            if not checked_operations:
                continue

            latest_operation = checked_operations[0]
            latest_metadata, latest_video_info, latest_video_url = self._extract_video_details_from_operation(latest_operation)
            retry_operations = [latest_operation]

        return latest_operation, latest_metadata, latest_video_info, latest_video_url

    def _build_failed_batch_item(self, index: int, prompt: str, error_message: Any) -> Dict[str, Any]:
        """构建批量图片失败项。"""
        return {
            "index": index,
            "prompt": prompt,
            "status": "failed",
            "error": self._normalize_error_message(error_message, max_length=240),
        }

    def _build_batch_summary(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总批量图片结果。"""
        total = len(items)
        succeeded = sum(1 for item in items if item.get("status") == "succeeded" and item.get("url"))
        failed = max(0, total - succeeded)
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "partial_success": bool(succeeded and failed),
            "all_failed": bool(total and not succeeded),
        }

    def _get_first_successful_batch_url(self, items: List[Dict[str, Any]]) -> Optional[str]:
        """返回批量结果中的第一张成功图片链接。"""
        for item in items:
            url = str(item.get("url") or "").strip()
            if item.get("status") == "succeeded" and url:
                return url
        return None

    async def _process_single_batch_image_item(
        self,
        *,
        token,
        project_id: str,
        model_config: Dict[str, Any],
        prompt: str,
        prompt_index: int,
        total_prompts: int,
        image_inputs: Optional[List[Dict[str, Any]]],
        response_state: Dict[str, Any],
        request_log_state: Optional[Dict[str, Any]],
        normalized_tier: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """逐项执行批量图片请求，返回成功或失败结果。"""
        item_trace: Dict[str, Any] = {
            "index": prompt_index,
            "prompt": prompt,
            "status": "failed",
        }
        generated_asset: Dict[str, Any] = {
            "type": "image",
            "prompt": prompt,
            "status": "failed",
        }

        progress_start = 28 + int((62 * prompt_index) / max(1, total_prompts))
        progress_end = 28 + int((62 * (prompt_index + 1)) / max(1, total_prompts))
        progress_end = max(progress_start + 1, min(progress_end, 96))

        async def _item_progress_callback(status_text: str, progress: int):
            bounded_progress = max(0, min(100, int(progress)))
            mapped_progress = progress_start + int((progress_end - progress_start) * bounded_progress / 100)
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text=f"batch_item_{prompt_index + 1}_{status_text}",
                progress=min(96, mapped_progress),
            )

        await self._update_request_log_progress(
            request_log_state,
            token_id=token.id,
            status_text=f"batch_item_{prompt_index + 1}_starting",
            progress=progress_start,
        )

        try:
            generate_started_at = time.time()
            result, generation_session_id, upstream_trace = await self.flow_client.generate_image(
                at=token.at,
                project_id=project_id,
                prompt=prompt,
                model_name=model_config["model_name"],
                aspect_ratio=model_config["aspect_ratio"],
                image_inputs=image_inputs,
                token_id=token.id,
                token_image_concurrency=token.image_concurrency,
                progress_callback=_item_progress_callback,
            )
            item_trace["generate_api_ms"] = int((time.time() - generate_started_at) * 1000)
            item_trace["upstream_trace"] = upstream_trace
        except Exception as exc:
            error_message = self._normalize_error_message(exc, max_length=240)
            item_trace["error"] = error_message
            generated_asset["error"] = error_message
            return self._build_failed_batch_item(prompt_index, prompt, error_message), generated_asset, item_trace

        media_entries = result.get("media", [])
        if isinstance(media_entries, dict):
            media_entries = [media_entries]
        if not isinstance(media_entries, list) or not media_entries:
            error_message = "生成结果为空"
            item_trace["error"] = error_message
            generated_asset["error"] = error_message
            return self._build_failed_batch_item(prompt_index, prompt, error_message), generated_asset, item_trace

        try:
            image_url, media_id = self._extract_generated_image_media(media_entries[0])
        except Exception as exc:
            error_message = self._normalize_error_message(exc, max_length=240)
            item_trace["error"] = error_message
            generated_asset["error"] = error_message
            return self._build_failed_batch_item(prompt_index, prompt, error_message), generated_asset, item_trace

        final_url = image_url
        generated_asset.update({
            "status": "succeeded",
            "origin_image_url": image_url,
        })
        item_trace["origin_image_url"] = image_url
        item_trace["media_id"] = media_id

        upsample_resolution = model_config.get("upsample")
        if upsample_resolution and media_id:
            resolution_name = "4K" if "4K" in upsample_resolution else "2K"
            upsample_started_at = time.time()
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text=f"batch_item_{prompt_index + 1}_upsampling_{resolution_name.lower()}",
                progress=min(98, progress_end + 4),
            )
            for retry_attempt in range(config.flow_max_retries):
                try:
                    encoded_image = await self.flow_client.upsample_image(
                        at=token.at,
                        project_id=project_id,
                        media_id=media_id,
                        target_resolution=upsample_resolution,
                        user_paygate_tier=normalized_tier,
                        session_id=generation_session_id,
                        token_id=token.id,
                    )

                    if not encoded_image:
                        break

                    generated_asset["upscaled_image"] = {"resolution": resolution_name}
                    try:
                        await self._update_request_log_progress(
                            request_log_state,
                            token_id=token.id,
                            status_text="caching_batch_images",
                            progress=99,
                        )
                        cached_filename = await self.file_cache.cache_base64_image(
                            encoded_image,
                            resolution_name,
                        )
                        final_url = f"{self._get_base_url(response_state)}/tmp/{cached_filename}"
                        generated_asset["upscaled_image"]["local_url"] = final_url
                        generated_asset["upscaled_image"]["url"] = final_url
                    except Exception as exc:
                        debug_logger.log_error(
                            f"[BATCH IMAGE] Failed to cache {resolution_name} image {prompt_index + 1}: {str(exc)}"
                        )
                        final_url = f"data:image/jpeg;base64,{encoded_image}"
                        generated_asset["upscaled_image"]["local_url"] = None
                        generated_asset["upscaled_image"]["url"] = image_url
                        generated_asset["upscaled_image"]["delivery_mode"] = "inline_base64_fallback"
                    break
                except Exception as exc:
                    error_str = str(exc)
                    debug_logger.log_error(
                        f"[BATCH IMAGE] 放大失败 ({prompt_index + 1}/{total_prompts}, "
                        f"尝试 {retry_attempt + 1}/{config.flow_max_retries}): {error_str}"
                    )
                    retry_reason = self.flow_client._get_retry_reason(error_str)
                    if retry_reason and retry_attempt < config.flow_max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    break
            item_trace["upsample_ms"] = int((time.time() - upsample_started_at) * 1000)
        else:
            cache_started_at = time.time()
            if config.cache_enabled:
                await self._update_request_log_progress(
                    request_log_state,
                    token_id=token.id,
                    status_text="caching_batch_images",
                    progress=99,
                )
                try:
                    cached_filename = await self.file_cache.download_and_cache(image_url, "image")
                    final_url = f"{self._get_base_url(response_state)}/tmp/{cached_filename}"
                except Exception as exc:
                    debug_logger.log_error(
                        f"[BATCH IMAGE] Failed to cache image {prompt_index + 1}: {str(exc)}"
                    )
                    final_url = image_url
            item_trace["cache_image_ms"] = int((time.time() - cache_started_at) * 1000)

        generated_asset["final_image_url"] = final_url
        item_trace["status"] = "succeeded"
        item_trace["final_image_url"] = final_url
        return {
            "index": prompt_index,
            "prompt": prompt,
            "status": "succeeded",
            "url": final_url,
            "origin_image_url": image_url,
            "media_id": media_id,
        }, generated_asset, item_trace

    def _resolve_video_model_key_for_tier(self, model_config: Dict[str, Any], user_tier: str) -> tuple[str, Optional[str]]:
        """根据账号层级调整视频模型 key。"""
        model_key = model_config["model_key"]
        allow_tier_upgrade = bool(model_config.get("allow_tier_upgrade", True))

        if user_tier == "PAYGATE_TIER_TWO":
            if allow_tier_upgrade and "ultra" not in model_key:
                upgraded_model_key = _resolve_tier_two_model_key(model_key)
                if upgraded_model_key != model_key:
                    return upgraded_model_key, f"TIER_TWO 账号自动切换到 ultra 模型: {upgraded_model_key}"
            return model_key, None

        if user_tier == "PAYGATE_TIER_ONE" and "ultra" in model_key:
            model_key = model_key.replace("_ultra_fl", "_fl").replace("_ultra", "")
            return model_key, f"TIER_ONE 账号自动切换到标准模型: {model_key}"

        return model_key, None

    async def _fail_video_task(self, operations: Optional[List[Dict[str, Any]]], error_message: str):
        """将视频任务收口到失败态，避免残留 processing。"""
        if not operations:
            return

        operation = operations[0] if operations else {}
        task_id = (operation.get("operation") or {}).get("name")
        if not task_id:
            return

        try:
            await self.db.update_task(
                task_id,
                status="failed",
                error_message=self._normalize_error_message(error_message),
                completed_at=time.time()
            )
        except Exception as exc:
            debug_logger.log_error(f"[VIDEO] 更新任务失败状态失败: {exc}")

    async def check_token_availability(self, is_image: bool, is_video: bool) -> bool:
        """检查Token可用性

        Args:
            is_image: 是否检查图片生成Token
            is_video: 是否检查视频生成Token

        Returns:
            True表示有可用Token, False表示无可用Token
        """
        token_obj = await self.load_balancer.select_token(
            for_image_generation=is_image,
            for_video_generation=is_video
        )
        return token_obj is not None

    async def handle_generation(
        self,
        model: str,
        prompt: str,
        images: Optional[List[bytes]] = None,
        stream: bool = False,
        base_url_override: Optional[str] = None,
        video_media_id: Optional[str] = None,
        batch_prompts: Optional[List[str]] = None,
        debug_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator:
        """统一生成入口

        Args:
            model: 模型名称
            prompt: 提示词
            images: 图片列表 (bytes格式)
            stream: 是否流式输出
        """
        start_time = time.time()
        token = None
        generation_type = None
        pending_token_state = {"active": False}
        request_id = f"gen-{int(start_time * 1000)}-{id(asyncio.current_task())}"
        perf_trace: Dict[str, Any] = {
            "request_id": request_id,
            "model": model,
            "status": "processing",
        }
        generation_result = self._create_generation_result()
        response_state = self._create_response_state()
        response_state["base_url"] = (base_url_override or "").strip().rstrip("/") or None
        debug_enabled = bool(isinstance(debug_options, dict) and debug_options.get("enabled"))
        response_state["debug_enabled"] = debug_enabled
        if isinstance(debug_options, dict) and "includeMediaCheck" in debug_options:
            response_state["debug_include_media_check"] = bool(debug_options.get("includeMediaCheck"))
        request_log_state: Dict[str, Any] = {"id": None, "progress": 0}

        async def _flow_debug_hook(event: str, payload: Dict[str, Any]):
            title_map = {
                "selected_token": "已选账号",
                "image_upload_prepare": "上传图片准备",
                "image_upload_request": "上传图片请求",
                "image_upload_response": "上传图片响应",
                "image_upload_error": "上传图片失败",
                "image_upload_fallback": "上传图片接口回退",
                "captcha_start": "开始打码",
                "captcha_result": "打码结果",
                "captcha_error": "打码失败",
                "http_request_context": "实际请求上下文",
                "image_submit_request": "提交生图请求",
                "image_submit_response": "生图请求响应",
                "image_submit_error": "生图请求失败",
                "image_media_check_request": "查询图片媒体状态",
                "image_media_check_response": "图片媒体状态响应",
                "image_media_check_error": "图片媒体状态查询失败",
            }
            chunk = await self._emit_debug_progress(
                response_state=response_state,
                stream=stream,
                request_log_state=request_log_state,
                token_id=token.id if token else None,
                event=event,
                title=title_map.get(event, event),
                payload=payload,
            )
            if chunk:
                yield_queue.append(chunk)

        yield_queue: List[str] = []
        if hasattr(self.flow_client, "set_debug_hook"):
            self.flow_client.set_debug_hook(_flow_debug_hook if debug_enabled else None)

        # 防止并发链路复用到上一次请求的指纹上下文
        if hasattr(self.flow_client, "clear_request_fingerprint"):
            self.flow_client.clear_request_fingerprint()

        # 1. 验证模型
        if model not in MODEL_CONFIG:
            error_msg = f"不支持的模型: {model}"
            debug_logger.log_error(error_msg)
            record_generation_result("unknown", "invalid", time.time() - start_time)
            yield self._create_error_response(error_msg, status_code=400)
            return

        model_config = MODEL_CONFIG[model]
        generation_type = model_config["type"]
        video_type_for_op = model_config.get("video_type", "")
        request_operation = "extend_video" if video_type_for_op == "extend" else f"generate_{generation_type}"
        prompt_log_source = prompt or (batch_prompts[0] if batch_prompts else "")
        prompt_for_log = (
            prompt_log_source
            if len(prompt_log_source) <= 2000
            else f"{prompt_log_source[:2000]}...(truncated)"
        )
        request_payload = {
            "model": model,
            "prompt": prompt_for_log,
            "has_images": images is not None and len(images) > 0,
        }
        if batch_prompts:
            request_payload["batch_prompt_count"] = len(batch_prompts)
        debug_logger.log_info(
            f"[GENERATION] 开始生成 - 模型: {model}, 类型: {generation_type}, "
            f"Prompt: {prompt_log_source[:50]}..."
        )

        if batch_prompts and generation_type != "image":
            error_msg = "batchPrompts 仅支持图片生成模型"
            debug_logger.log_error(f"[GENERATION] {error_msg}")
            record_generation_result(generation_type, "invalid", time.time() - start_time)
            yield self._create_error_response(error_msg, status_code=400)
            return

        if batch_prompts and stream:
            error_msg = "batchPrompts 暂不支持流式输出"
            debug_logger.log_error(f"[GENERATION] {error_msg}")
            record_generation_result(generation_type, "invalid", time.time() - start_time)
            yield self._create_error_response(error_msg, status_code=400)
            return

        # 向用户展示开始信息
        if stream:
            yield self._create_stream_chunk(
                f"✨ {'视频' if generation_type == 'video' else '图片'}生成任务已启动\n",
                role="assistant"
            )
            while yield_queue:
                yield yield_queue.pop(0)
            request_log_state["id"] = await self._log_request(
                token_id=None,
                operation=request_operation,
                request_data=request_payload,
                response_data={"status": "processing", "status_text": "started", "progress": 0, "request_id": request_id},
                status_code=102,
                duration=0,
                status_text="started",
                progress=0,
            )

        # 2. 选择Token
        debug_logger.log_info(f"[GENERATION] 正在选择可用Token...")
        token_select_started_at = time.time()

        if generation_type == "image":
            token = await self.load_balancer.select_token(
                for_image_generation=True,
                model=model,
                reserve=False,
                enforce_concurrency_filter=False,
                track_pending=True,
            )
        else:
            token = await self.load_balancer.select_token(
                for_video_generation=True,
                model=model,
                reserve=False,
                enforce_concurrency_filter=False,
                track_pending=True,
            )
        perf_trace["token_select_ms"] = int((time.time() - token_select_started_at) * 1000)

        if not token:
            error_msg = None
            if self.load_balancer and hasattr(self.load_balancer, "get_unavailable_reason"):
                error_msg = await self.load_balancer.get_unavailable_reason(
                    for_image_generation=(generation_type == "image"),
                    for_video_generation=(generation_type == "video"),
                    model=model,
                )
            if not error_msg:
                error_msg = self._get_no_token_error_message(generation_type)
            debug_logger.log_error(f"[GENERATION] {error_msg}")
            record_generation_result(generation_type, "no_token", time.time() - start_time)
            await self._log_request(
                token_id=None,
                operation=request_operation,
                request_data=request_payload,
                response_data={"error": error_msg, "performance": perf_trace},
                status_code=503,
                duration=time.time() - start_time,
                log_id=request_log_state.get("id"),
                status_text="failed",
                progress=request_log_state.get("progress", 0),
            )
            if stream:
                yield self._create_stream_chunk(f"❌ {error_msg}\n")
            yield self._create_error_response(error_msg, status_code=503)
            return

        debug_logger.log_info(f"[GENERATION] 已选择Token: {token.id} ({token.email})")
        pending_token_state["active"] = True
        selected_token_chunk = await self._emit_debug_progress(
            response_state=response_state,
            stream=stream,
            request_log_state=request_log_state,
            token_id=token.id,
            event="selected_token",
            title="已选账号",
            payload={
                "token_id": token.id,
                "email": token.email,
                "current_project_id": getattr(token, "current_project_id", None),
                "image_enabled": getattr(token, "image_enabled", None),
                "video_enabled": getattr(token, "video_enabled", None),
                "image_concurrency": getattr(token, "image_concurrency", None),
                "video_concurrency": getattr(token, "video_concurrency", None),
                "captcha_proxy_url": getattr(token, "captcha_proxy_url", None),
            },
        )
        if selected_token_chunk:
            yield_queue.append(selected_token_chunk)
        await self._update_request_log_progress(
            request_log_state,
            token_id=token.id,
            status_text="token_selected",
            progress=8,
            response_extra={"token_email": token.email},
        )

        try:
            # 3. 确保AT有效
            debug_logger.log_info(f"[GENERATION] 检查Token AT有效性...")
            if stream:
                yield self._create_stream_chunk("初始化生成环境...\n")
                while yield_queue:
                    yield yield_queue.pop(0)

            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="token_ready",
                progress=15,
            )
            ensure_at_started_at = time.time()
            token = await self.token_manager.ensure_valid_token(token)
            perf_trace["ensure_at_ms"] = int((time.time() - ensure_at_started_at) * 1000)
            if not token:
                error_msg = "Token AT无效或刷新失败"
                debug_logger.log_error(f"[GENERATION] {error_msg}")
                record_generation_result(generation_type, "failed", time.time() - start_time)
                if stream:
                    yield self._create_stream_chunk(f"❌ {error_msg}\n")
                yield self._create_error_response(error_msg, status_code=503)
                return

            # 4. 确保Project存在
            debug_logger.log_info(f"[GENERATION] 检查/创建Project...")

            if not supports_model_for_tier(model, token.user_paygate_tier):
                required_tier = get_required_paygate_tier_for_model(model)
                error_msg = "当前模型需要 " + get_paygate_tier_label(required_tier) + " 账号: " + model
                debug_logger.log_error(f"[GENERATION] {error_msg}")
                record_generation_result(generation_type, "failed", time.time() - start_time)
                if stream:
                    yield self._create_stream_chunk(f"❌ {error_msg}\n")
                yield self._create_error_response(error_msg, status_code=403)
                return

            ensure_project_started_at = time.time()
            project_id = await self.token_manager.ensure_project_exists(token.id)
            perf_trace["ensure_project_ms"] = int((time.time() - ensure_project_started_at) * 1000)
            debug_logger.log_info(f"[GENERATION] Project ID: {project_id}")
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="project_ready",
                progress=22,
                response_extra={"project_id": project_id},
            )
            prefill_action = "IMAGE_GENERATION" if generation_type == "image" else "VIDEO_GENERATION"
            prefill_proxy_url = str(getattr(token, "captcha_proxy_url", "") or "").strip() or None
            if prefill_proxy_url is None and self.proxy_manager:
                try:
                    prefill_proxy_url = await self.proxy_manager.get_request_proxy_url()
                except Exception as e:
                    debug_logger.log_warning(f"[GENERATION] 获取 remote_browser prefill 代理失败: {e}")
            await self.flow_client.prefill_remote_browser_pool(
                project_id=project_id,
                action=prefill_action,
                token_id=token.id,
                proxy_url=prefill_proxy_url,
            )

            # 5. 根据类型处理
            generation_pipeline_started_at = time.time()
            if generation_type == "image":
                debug_logger.log_info(f"[GENERATION] 开始图片生成流程...")
                if batch_prompts:
                    async for chunk in self._handle_batch_image_generation(
                        token, project_id, model_config, batch_prompts, images, stream,
                        perf_trace=perf_trace,
                        generation_result=generation_result,
                        response_state=response_state,
                        request_log_state=request_log_state,
                        pending_token_state=pending_token_state,
                    ):
                        yield chunk
                        while yield_queue:
                            yield yield_queue.pop(0)
                else:
                    async for chunk in self._handle_image_generation(
                        token, project_id, model_config, prompt, images, stream,
                        perf_trace=perf_trace,
                        generation_result=generation_result,
                        response_state=response_state,
                        request_log_state=request_log_state,
                        pending_token_state=pending_token_state
                    ):
                        yield chunk
                        while yield_queue:
                            yield yield_queue.pop(0)
            else:  # video
                debug_logger.log_info(f"[GENERATION] 开始视频生成流程...")
                async for chunk in self._handle_video_generation(
                    token, project_id, model_config, prompt, images, stream,
                    perf_trace=perf_trace,
                    generation_result=generation_result,
                    response_state=response_state,
                    request_log_state=request_log_state,
                    pending_token_state=pending_token_state,
                    video_media_id=video_media_id,
                ):
                    yield chunk
                    while yield_queue:
                        yield yield_queue.pop(0)
            perf_trace["generation_pipeline_ms"] = int((time.time() - generation_pipeline_started_at) * 1000)
            while yield_queue:
                yield yield_queue.pop(0)

            # 6. 记录使用
            if not generation_result.get("success"):
                error_msg = generation_result.get("error_message") or "生成未成功完成"
                debug_logger.log_warning(f"[GENERATION] 生成未成功，不扣次数: {error_msg}")
                if token:
                    await self.token_manager.record_error(token.id)
                duration = time.time() - start_time
                record_generation_result(generation_type, "failed", duration)
                perf_trace["status"] = "failed"
                perf_trace["total_ms"] = int(duration * 1000)
                perf_trace["error"] = error_msg
                error_extra = generation_result.get("error_extra") if isinstance(generation_result, dict) else None
                response_data = {"error": error_msg, "performance": perf_trace}
                if isinstance(error_extra, dict):
                    response_data.update(error_extra)
                await self._log_request(
                    token.id if token else None,
                    request_operation,
                    request_payload,
                    response_data,
                    500,
                    duration,
                    log_id=request_log_state.get("id"),
                    status_text="failed",
                    progress=request_log_state.get("progress", 0),
                )
                while yield_queue:
                    yield yield_queue.pop(0)
                if not generation_result.get("error_emitted"):
                    if stream:
                        yield self._create_stream_chunk(f"❌ {error_msg}\n")
                    yield self._create_error_response(error_msg, status_code=500, extra=error_extra)
                return

            result_status = str(generation_result.get("result_status") or "success")
            usage_success = bool(generation_result.get("usage_success", True))

            if usage_success:
                is_video = (generation_type == "video")
                await self.token_manager.record_usage(token.id, is_video=is_video)

                # 重置错误计数 (请求成功时清空连续错误计数)
                await self.token_manager.record_success(token.id)
            else:
                await self.token_manager.record_error(token.id)

            debug_logger.log_info(f"[GENERATION] ✅ 生成响应完成 (status={result_status})")

            # 7. 记录成功日志
            duration = time.time() - start_time
            record_generation_result(generation_type, "success" if usage_success else "failed", duration)
            perf_trace["status"] = result_status
            perf_trace["total_ms"] = int(duration * 1000)
            # 日志中保留更完整的 prompt，避免管理页只看到过短内容

            # 构建响应数据，包含生成的URL
            response_data = {
                "status": result_status,
                "model": model,
                "prompt": prompt_for_log,
                "performance": perf_trace
            }

            # 添加生成的URL（如果有）
            if response_state.get("url"):
                response_data["url"] = response_state["url"]
            if response_state.get("generated_assets"):
                response_data["generated_assets"] = response_state["generated_assets"]
            image_perf = perf_trace.get("image_generation", {}) if isinstance(perf_trace, dict) else {}
            video_perf = perf_trace.get("video_generation", {}) if isinstance(perf_trace, dict) else {}
            debug_logger.log_info(
                f"[PERF] [{request_id}] total={perf_trace.get('total_ms', 0)}ms, "
                f"select={perf_trace.get('token_select_ms', 0)}ms, "
                f"ensure_at={perf_trace.get('ensure_at_ms', 0)}ms, "
                f"project={perf_trace.get('ensure_project_ms', 0)}ms, "
                f"pipeline={perf_trace.get('generation_pipeline_ms', 0)}ms, "
                f"slot_wait={image_perf.get('slot_wait_ms', 0)}ms, "
                f"launch_queue={image_perf.get('launch_queue_wait_ms', 0)}ms, "
                f"launch_stagger={image_perf.get('launch_stagger_wait_ms', 0)}ms, "
                f"video_slot_wait={video_perf.get('slot_wait_ms', 0)}ms"
            )

            await self._log_request(
                token.id,
                request_operation,
                request_payload,
                response_data,
                200,
                duration,
                log_id=request_log_state.get("id"),
                status_text="completed",
                progress=100,
            )

        except asyncio.CancelledError:
            error_msg = "生成已取消: 客户端连接已断开"
            debug_logger.log_warning(f"[GENERATION] ⚠️ {error_msg}")
            duration = time.time() - start_time
            record_generation_result(generation_type or "unknown", "cancelled", duration)
            perf_trace["status"] = "failed"
            perf_trace["total_ms"] = int(duration * 1000)
            perf_trace["error"] = error_msg
            await self._log_request(
                token.id if token else None,
                request_operation if generation_type else "generate_unknown",
                request_payload if 'request_payload' in locals() else {"model": model},
                {"error": error_msg, "performance": perf_trace},
                499,
                duration,
                log_id=request_log_state.get("id"),
                status_text="failed",
                progress=request_log_state.get("progress", 0),
            )
            while yield_queue:
                yield yield_queue.pop(0)
            raise
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            debug_logger.log_error(f"[GENERATION] ❌ {error_msg}")
            if token:
                # 记录错误（所有错误统一处理，不再特殊处理429）
                await self.token_manager.record_error(token.id)

            # 先将最终失败状态落库，再返回错误响应，避免日志停在 102。
            duration = time.time() - start_time
            record_generation_result(generation_type or "unknown", "failed", duration)
            perf_trace["status"] = "failed"
            perf_trace["total_ms"] = int(duration * 1000)
            perf_trace["error"] = error_msg
            await self._log_request(
                token.id if token else None,
                request_operation if generation_type else "generate_unknown",
                request_payload if 'request_payload' in locals() else {"model": model},
                {"error": error_msg, "performance": perf_trace},
                500,
                duration,
                log_id=request_log_state.get("id"),
                status_text="failed",
                progress=request_log_state.get("progress", 0),
            )
            while yield_queue:
                yield yield_queue.pop(0)
            if stream:
                yield self._create_stream_chunk(f"❌ {error_msg}\n")
            yield self._create_error_response(error_msg, status_code=500)
        finally:
            if hasattr(self.flow_client, "set_debug_hook"):
                self.flow_client.set_debug_hook(None)
            if pending_token_state.get("active") and token and self.load_balancer:
                await self.load_balancer.release_pending(
                    token.id,
                    for_image_generation=(generation_type == "image"),
                    for_video_generation=(generation_type == "video"),
                )
                pending_token_state["active"] = False


    def _get_no_token_error_message(self, generation_type: str) -> str:
        """获取无可用Token时的详细错误信息"""
        if generation_type == "image":
            return "没有可用的Token进行图片生成。所有Token都处于禁用、冷却、锁定或已过期状态。"
        else:
            return "没有可用的Token进行视频生成。所有Token都处于禁用、冷却、配额耗尽或已过期状态。"

    def _extract_generated_image_media(self, media_entry: Dict[str, Any]) -> tuple[str, Optional[str]]:
        """从上游 media 条目中提取图片 URL 和 mediaId。"""
        image = media_entry.get("image", {}) if isinstance(media_entry, dict) else {}
        generated_image = image.get("generatedImage", {}) if isinstance(image, dict) else {}
        image_url = str(generated_image.get("fifeUrl") or "").strip()
        if not image_url:
            raise ValueError("生成结果缺少图片链接")
        media_id = media_entry.get("name") if isinstance(media_entry, dict) else None
        return image_url, media_id

    async def _handle_batch_image_generation(
        self,
        token,
        project_id: str,
        model_config: dict,
        prompts: List[str],
        images: Optional[List[bytes]],
        stream: bool,
        perf_trace: Optional[Dict[str, Any]] = None,
        generation_result: Optional[Dict[str, Any]] = None,
        response_state: Optional[Dict[str, Any]] = None,
        request_log_state: Optional[Dict[str, Any]] = None,
        pending_token_state: Optional[Dict[str, bool]] = None,
    ) -> AsyncGenerator:
        """处理批量图片生成，共享同一组参考图。"""

        if response_state is None:
            response_state = self._create_response_state()

        image_trace: Optional[Dict[str, Any]] = None
        if isinstance(perf_trace, dict):
            image_trace = perf_trace.setdefault("image_generation", {})
            image_trace["input_image_count"] = len(images) if images else 0
            image_trace["batch_prompt_count"] = len(prompts)

        normalized_tier = normalize_user_paygate_tier(token.user_paygate_tier)

        if image_trace is not None:
            image_trace["slot_wait_ms"] = 0

        if images and len(images) > 0:
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="uploading_images",
                progress=28,
            )
        else:
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="submitting_batch_images",
                progress=28,
            )

        try:
            upload_started_at = time.time()
            image_inputs = []
            upload_error: Optional[str] = None
            if images and len(images) > 0:
                try:
                    for image_bytes in images:
                        media_id = await self.flow_client.upload_image(
                            token.at,
                            image_bytes,
                            model_config["aspect_ratio"],
                            project_id=project_id
                        )
                        image_inputs.append({
                            "name": media_id,
                            "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"
                        })
                except Exception as exc:
                    upload_error = self._normalize_error_message(exc, max_length=240)
            if image_trace is not None:
                image_trace["upload_images_ms"] = int((time.time() - upload_started_at) * 1000)
            batch_items: List[Dict[str, Any]] = []
            batch_assets: List[Dict[str, Any]] = []
            item_traces: List[Dict[str, Any]] = []

            if upload_error:
                batch_items = [
                    self._build_failed_batch_item(idx, prompt_text, upload_error)
                    for idx, prompt_text in enumerate(prompts)
                ]
                batch_assets = [
                    {
                        "type": "image",
                        "prompt": prompt_text,
                        "status": "failed",
                        "error": upload_error,
                    }
                    for prompt_text in prompts
                ]
                item_traces = [
                    {
                        "index": idx,
                        "prompt": prompt_text,
                        "status": "failed",
                        "error": upload_error,
                    }
                    for idx, prompt_text in enumerate(prompts)
                ]
            else:
                async def _run_batch_item(prompt_index: int, prompt_text: str):
                    try:
                        return await self._process_single_batch_image_item(
                            token=token,
                            project_id=project_id,
                            model_config=model_config,
                            prompt=prompt_text,
                            prompt_index=prompt_index,
                            total_prompts=len(prompts),
                            image_inputs=image_inputs,
                            response_state=response_state,
                            request_log_state=request_log_state,
                            normalized_tier=normalized_tier,
                        )
                    except Exception as exc:
                        error_message = self._normalize_error_message(exc, max_length=240)
                        debug_logger.log_error(
                            f"[BATCH IMAGE] 并发子任务异常 ({prompt_index + 1}/{len(prompts)}): {error_message}"
                        )
                        return (
                            self._build_failed_batch_item(prompt_index, prompt_text, error_message),
                            {
                                "type": "image",
                                "prompt": prompt_text,
                                "status": "failed",
                                "error": error_message,
                            },
                            {
                                "index": prompt_index,
                                "prompt": prompt_text,
                                "status": "failed",
                                "error": error_message,
                            },
                        )

                tasks = [
                    asyncio.create_task(_run_batch_item(idx, prompt_text))
                    for idx, prompt_text in enumerate(prompts)
                ]
                task_results = await asyncio.gather(*tasks)

                total_generate_ms = 0
                total_upsample_ms = 0
                total_cache_ms = 0
                for item_result, generated_asset, item_trace in task_results:
                    batch_items.append(item_result)
                    batch_assets.append(generated_asset)
                    item_traces.append(item_trace)
                    total_generate_ms += int(item_trace.get("generate_api_ms") or 0)
                    total_upsample_ms += int(item_trace.get("upsample_ms") or 0)
                    total_cache_ms += int(item_trace.get("cache_image_ms") or 0)

                if image_trace is not None:
                    image_trace["generate_api_ms"] = total_generate_ms
                    if total_upsample_ms:
                        image_trace["upsample_ms"] = total_upsample_ms
                    image_trace["cache_image_ms"] = total_cache_ms

            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="batch_images_generated",
                progress=100,
            )

            summary = self._build_batch_summary(batch_items)
            response_state["url"] = self._get_first_successful_batch_url(batch_items)
            response_state["generated_assets"] = {
                "type": "image_batch",
                "items": batch_assets,
                "summary": summary,
            }
            if image_trace is not None:
                image_trace["batch_items"] = item_traces
                image_trace["batch_summary"] = summary

            if summary["succeeded"] > 0:
                result_status = "partial_success" if summary["failed"] else "success"
                summary_message = (
                    f"{summary['failed']} of {summary['total']} batch item(s) failed"
                    if summary["failed"]
                    else None
                )
                self._mark_generation_succeeded(
                    generation_result,
                    usage_success=True,
                    result_status=result_status,
                    message=summary_message,
                )
            else:
                self._mark_generation_succeeded(
                    generation_result,
                    usage_success=False,
                    result_status="all_failed",
                    message="All batch items failed",
                )

            if stream:
                yield self._create_stream_chunk("batchPrompts 暂不支持流式输出\n", finish_reason="stop")
            else:
                yield self._create_batch_image_completion_response(batch_items)

        finally:
            pass

    async def _handle_image_generation(
        self,
        token,
        project_id: str,
        model_config: dict,
        prompt: str,
        images: Optional[List[bytes]],
        stream: bool,
        perf_trace: Optional[Dict[str, Any]] = None,
        generation_result: Optional[Dict[str, Any]] = None,
        response_state: Optional[Dict[str, Any]] = None,
        request_log_state: Optional[Dict[str, Any]] = None,
        pending_token_state: Optional[Dict[str, bool]] = None
    ) -> AsyncGenerator:
        """处理图片生成 (同步返回)"""

        if response_state is None:
            response_state = self._create_response_state()

        image_trace: Optional[Dict[str, Any]] = None
        if isinstance(perf_trace, dict):
            image_trace = perf_trace.setdefault("image_generation", {})
            image_trace["input_image_count"] = len(images) if images else 0

        # 不在本地等待图片硬并发槽位；请求一到就直接向上游提交。
        normalized_tier = normalize_user_paygate_tier(token.user_paygate_tier)

        if image_trace is not None:
            image_trace["slot_wait_ms"] = 0

        if images and len(images) > 0:
            await self._update_request_log_progress(request_log_state, token_id=token.id, status_text="uploading_images", progress=28)
        else:
            await self._update_request_log_progress(request_log_state, token_id=token.id, status_text="submitting_image", progress=28)

        try:
            # 上传图片 (如果有)
            upload_started_at = time.time()
            image_inputs = []
            if images and len(images) > 0:
                if stream:
                    yield self._create_stream_chunk(f"上传 {len(images)} 张参考图片...\n")

                # 支持多图输入
                for idx, image_bytes in enumerate(images):
                    media_id = await self.flow_client.upload_image(
                        token.at,
                        image_bytes,
                        model_config["aspect_ratio"],
                        project_id=project_id
                    )
                    image_inputs.append({
                        "name": media_id,
                        "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"
                    })
                    if stream:
                        yield self._create_stream_chunk(f"已上传第 {idx + 1}/{len(images)} 张图片\n")
            if image_trace is not None:
                image_trace["upload_images_ms"] = int((time.time() - upload_started_at) * 1000)

            # 调用生成API
            if stream:
                if images and len(images) > 0:
                    yield self._create_stream_chunk("参考图片上传完成，正在进行打码验证...\n")
                else:
                    yield self._create_stream_chunk("正在进行打码验证并提交图片生成请求...\n")

            async def _image_progress_callback(status_text: str, progress: int):
                await self._update_request_log_progress(
                    request_log_state,
                    token_id=token.id,
                    status_text=status_text,
                    progress=progress,
                )

            generate_started_at = time.time()
            result, generation_session_id, upstream_trace = await self.flow_client.generate_image(
                at=token.at,
                project_id=project_id,
                prompt=prompt,
                model_name=model_config["model_name"],
                aspect_ratio=model_config["aspect_ratio"],
                image_inputs=image_inputs,
                token_id=token.id,
                token_image_concurrency=token.image_concurrency,
                progress_callback=_image_progress_callback,
            )
            if image_trace is not None:
                image_trace["generate_api_ms"] = int((time.time() - generate_started_at) * 1000)
                image_trace["upstream_trace"] = upstream_trace
                attempts = upstream_trace.get("generation_attempts") if isinstance(upstream_trace, dict) else None
                if isinstance(attempts, list) and attempts:
                    first_attempt = attempts[0] if isinstance(attempts[0], dict) else {}
                    image_trace["launch_queue_wait_ms"] = int(first_attempt.get("launch_queue_ms") or 0)
                    image_trace["launch_stagger_wait_ms"] = int(first_attempt.get("launch_stagger_ms") or 0)
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="image_generated",
                progress=72,
            )

            # 提取URL和mediaId
            media = result.get("media", [])
            if not media:
                self._mark_generation_failed(generation_result, "\u751f\u6210\u7ed3\u679c\u4e3a\u7a7a")
                yield self._create_error_response("生成结果为空", status_code=502)
                return

            image_url = media[0]["image"]["generatedImage"]["fifeUrl"]
            media_id = media[0].get("name")  # 用于 upsample
            response_state["generated_assets"] = {
                "type": "image",
                "origin_image_url": image_url
            }

            if response_state.get("debug_enabled") and response_state.get("debug_include_media_check") and media_id:
                try:
                    media_result = await self.flow_client.get_media(token.at, media_id)
                    refreshed_url = (
                        media_result.get("image", {})
                        .get("generatedImage", {})
                        .get("fifeUrl")
                    )
                    if refreshed_url:
                        image_url = refreshed_url
                        response_state["generated_assets"]["origin_image_url"] = image_url
                except Exception as exc:
                    debug_logger.log_warning(f"[IMAGE DEBUG] get_media failed: {exc}")

            # 检查是否需要 upsample
            upsample_resolution = model_config.get("upsample")
            if upsample_resolution and media_id:
                upsample_started_at = time.time()
                resolution_name = "4K" if "4K" in upsample_resolution else "2K"
                await self._update_request_log_progress(request_log_state, token_id=token.id, status_text=f"upsampling_{resolution_name.lower()}", progress=82)
                if stream:
                    yield self._create_stream_chunk(f"正在放大图片到 {resolution_name}...\n")

                # 4K/2K 图片重试逻辑 - 使用配置的最大重试次数
                max_retries = config.flow_max_retries
                for retry_attempt in range(max_retries):
                    try:
                        # 调用 upsample API
                        encoded_image = await self.flow_client.upsample_image(
                            at=token.at,
                            project_id=project_id,
                            media_id=media_id,
                            target_resolution=upsample_resolution,
                            user_paygate_tier=normalized_tier,
                            session_id=generation_session_id,
                            token_id=token.id
                        )

                        if encoded_image:
                            debug_logger.log_info(f"[UPSAMPLE] 图片已放大到 {resolution_name}")

                            if stream:
                                yield self._create_stream_chunk(f"✅ 图片已放大到 {resolution_name}\n")

                            # 2K/4K 图片统一落盘为真实文件，日志里只保留链接。
                            response_state["generated_assets"] = {
                                "type": "image",
                                "origin_image_url": image_url,
                                "upscaled_image": {
                                    "resolution": resolution_name
                                }
                            }

                            try:
                                await self._update_request_log_progress(
                                    request_log_state,
                                    token_id=token.id,
                                    status_text="caching_image",
                                    progress=90,
                                )
                                if stream:
                                    yield self._create_stream_chunk(f"缓存 {resolution_name} 图片中...\n")
                                cached_filename = await self.file_cache.cache_base64_image(encoded_image, resolution_name)
                                local_url = f"{self._get_base_url(response_state)}/tmp/{cached_filename}"
                                response_state["url"] = local_url
                                response_state["generated_assets"]["upscaled_image"]["local_url"] = local_url
                                response_state["generated_assets"]["upscaled_image"]["url"] = local_url
                                self._mark_generation_succeeded(generation_result)
                                if stream:
                                    yield self._create_stream_chunk(f"✅ {resolution_name} 图片缓存成功\n")
                                    yield self._create_stream_chunk(
                                        f"![Generated Image]({local_url})",
                                        finish_reason="stop"
                                    )
                                else:
                                    yield self._create_completion_response(
                                        local_url,
                                        media_type="image"
                                    )
                                if image_trace is not None:
                                    image_trace["upsample_ms"] = int((time.time() - upsample_started_at) * 1000)
                                return
                            except Exception as e:
                                debug_logger.log_error(f"Failed to cache {resolution_name} image: {str(e)}")
                                response_state["url"] = image_url
                                response_state["generated_assets"]["upscaled_image"]["local_url"] = None
                                response_state["generated_assets"]["upscaled_image"]["url"] = image_url
                                response_state["generated_assets"]["upscaled_image"]["delivery_mode"] = "inline_base64_fallback"
                                self._mark_generation_succeeded(generation_result)
                                base64_url = f"data:image/jpeg;base64,{encoded_image}"
                                if stream:
                                    cache_error = self._normalize_error_message(e, max_length=120)
                                    yield self._create_stream_chunk(f"⚠️ 缓存失败: {cache_error}，返回内联图片...\n")
                                    yield self._create_stream_chunk(
                                        f"![Generated Image]({base64_url})",
                                        finish_reason="stop"
                                    )
                                else:
                                    yield self._create_completion_response(
                                        base64_url,
                                        media_type="image"
                                    )
                                if image_trace is not None:
                                    image_trace["upsample_ms"] = int((time.time() - upsample_started_at) * 1000)
                                return
                        else:
                            debug_logger.log_warning("[UPSAMPLE] 返回结果为空")
                            if stream:
                                yield self._create_stream_chunk(f"⚠️ 放大失败，返回原图...\n")
                            break  # 空结果不重试

                    except Exception as e:
                        error_str = str(e)
                        debug_logger.log_error(f"[UPSAMPLE] 放大失败 (尝试 {retry_attempt + 1}/{max_retries}): {error_str}")
                        
                        # 检查是否是可重试错误（403、reCAPTCHA、超时等）
                        retry_reason = self.flow_client._get_retry_reason(error_str)
                        if retry_reason and retry_attempt < max_retries - 1:
                            if stream:
                                yield self._create_stream_chunk(f"⚠️ 放大遇到{retry_reason}，正在重试 ({retry_attempt + 2}/{max_retries})...\n")
                            # 等待一小段时间后重试
                            await asyncio.sleep(1)
                            continue
                        else:
                            if stream:
                                yield self._create_stream_chunk(f"⚠️ 放大失败: {error_str}，返回原图...\n")
                            break
                if image_trace is not None:
                    image_trace["upsample_ms"] = int((time.time() - upsample_started_at) * 1000)

            local_url = image_url
            cache_started_at = time.time()
            if config.cache_enabled:
                await self._update_request_log_progress(
                    request_log_state,
                    token_id=token.id,
                    status_text="caching_image",
                    progress=90,
                )
                if stream:
                    yield self._create_stream_chunk("正在缓存 1K 图片文件...\n")
                try:
                    cached_filename = await self.file_cache.download_and_cache(image_url, "image")
                    local_url = f"{self._get_base_url(response_state)}/tmp/{cached_filename}"
                    if stream:
                        yield self._create_stream_chunk("✅ 1K 图片缓存成功,准备返回缓存地址...\n")
                except Exception as e:
                    debug_logger.log_error(f"Failed to cache 1K image: {str(e)}")
                    local_url = image_url
                    if stream:
                        cache_error = self._normalize_error_message(e, max_length=120)
                        yield self._create_stream_chunk(f"⚠️ 缓存失败: {cache_error}\n正在返回源链接...\n")
            elif stream:
                yield self._create_stream_chunk("缓存已关闭,正在返回官方图片链接...\n")
            if image_trace is not None:
                image_trace["cache_image_ms"] = int((time.time() - cache_started_at) * 1000)

            # 返回结果
            # 存储URL用于日志记录
            response_state["url"] = local_url
            response_state["generated_assets"] = {
                "type": "image",
                "origin_image_url": image_url,
                "final_image_url": local_url
            }
            self._mark_generation_succeeded(generation_result)

            if stream:
                yield self._create_stream_chunk(
                    f"![Generated Image]({local_url})",
                    finish_reason="stop"
                )
            else:
                yield self._create_completion_response(
                    local_url,  # 直接传URL,让方法内部格式化
                    media_type="image"
                )

        finally:
            pass

    async def _handle_video_generation(
        self,
        token,
        project_id: str,
        model_config: dict,
        prompt: str,
        images: Optional[List[bytes]],
        stream: bool,
        perf_trace: Optional[Dict[str, Any]] = None,
        generation_result: Optional[Dict[str, Any]] = None,
        response_state: Optional[Dict[str, Any]] = None,
        request_log_state: Optional[Dict[str, Any]] = None,
        pending_token_state: Optional[Dict[str, bool]] = None,
        video_media_id: Optional[str] = None,
    ) -> AsyncGenerator:
        """处理视频生成 (异步轮询)"""

        if response_state is None:
            response_state = self._create_response_state()

        video_trace: Optional[Dict[str, Any]] = None
        if isinstance(perf_trace, dict):
            video_trace = perf_trace.setdefault("video_generation", {})
            video_trace["input_image_count"] = len(images) if images else 0

        # 不在本地等待视频硬并发槽位；请求一到就直接向上游提交。
        normalized_tier = normalize_user_paygate_tier(token.user_paygate_tier)

        if video_trace is not None:
            video_trace["slot_wait_ms"] = 0

        await self._update_request_log_progress(request_log_state, token_id=token.id, status_text="preparing_video", progress=24)

        try:
            # 获取模型类型和配置
            video_type = model_config.get("video_type")
            supports_images = model_config.get("supports_images", False)
            min_images = model_config.get("min_images", 0)
            max_images = model_config.get("max_images", 0)
            use_v2_model_config = bool(model_config.get("use_v2_model_config", False))

            # 根据账号tier自动调整模型 key
            user_tier = normalized_tier

            original_model_key = model_config["model_key"]
            model_key, tier_message = self._resolve_video_model_key_for_tier(model_config, user_tier)
            if tier_message:
                if stream:
                    yield self._create_stream_chunk(f"{tier_message}\n")
                debug_logger.log_info(f"[VIDEO] 账号层级模型调整: {original_model_key} -> {model_key}")
            elif user_tier == "PAYGATE_TIER_TWO" and original_model_key == model_key:
                debug_logger.log_info(f"[VIDEO] TIER_TWO 账号，未找到有效 ultra 变体，保持模型: {model_key}")

            # 更新 model_config 中的 model_key
            model_config = dict(model_config)  # 创建副本避免修改原配置
            model_config["model_key"] = model_key

            # 图片数量
            image_count = len(images) if images else 0

            # ========== 验证和处理图片 ==========

            # T2V: 文生视频 - 不支持图片
            if video_type == "t2v":
                if image_count > 0:
                    if stream:
                        yield self._create_stream_chunk("⚠️ 文生视频模型不支持上传图片,将忽略图片仅使用文本提示词生成\n")
                    debug_logger.log_warning(f"[T2V] 模型 {model_config['model_key']} 不支持图片,已忽略 {image_count} 张图片")
                images = None  # 清空图片
                image_count = 0

            # I2V: 首尾帧模型 - 需要1-2张图片
            elif video_type == "i2v":
                if image_count < min_images or image_count > max_images:
                    error_msg = f"❌ 首尾帧模型需要 {min_images}-{max_images} 张图片,当前提供了 {image_count} 张"
                    if stream:
                        yield self._create_stream_chunk(f"{error_msg}\n")
                    self._mark_generation_failed(generation_result, error_msg)
                    yield self._create_error_response(error_msg, status_code=400)
                    return

            # R2V: 多图生成 - 当前上游协议最多 3 张参考图
            elif video_type == "r2v":
                if max_images is not None and image_count > max_images:
                    error_msg = f"❌ 多图视频模型最多支持 {max_images} 张参考图,当前提供了 {image_count} 张"
                    if stream:
                        yield self._create_stream_chunk(f"{error_msg}\n")
                    self._mark_generation_failed(generation_result, error_msg)
                    yield self._create_error_response(error_msg, status_code=400)
                    return

            # ========== 上传图片 ==========
            start_media_id = None
            end_media_id = None
            reference_images = []

            # I2V: 首尾帧处理
            if video_type == "i2v" and images:
                if image_count == 1:
                    # 只有1张图: 仅作为首帧
                    if stream:
                        yield self._create_stream_chunk("上传首帧图片...\n")
                    start_media_id = await self.flow_client.upload_image(
                        token.at, images[0], model_config["aspect_ratio"], project_id=project_id
                    )
                    debug_logger.log_info(f"[I2V] 仅上传首帧: {start_media_id}")

                elif image_count == 2:
                    # 2张图: 首帧+尾帧
                    if stream:
                        yield self._create_stream_chunk("上传首帧和尾帧图片...\n")
                    start_media_id = await self.flow_client.upload_image(
                        token.at, images[0], model_config["aspect_ratio"], project_id=project_id
                    )
                    end_media_id = await self.flow_client.upload_image(
                        token.at, images[1], model_config["aspect_ratio"], project_id=project_id
                    )
                    debug_logger.log_info(f"[I2V] 上传首尾帧: {start_media_id}, {end_media_id}")

            # R2V: 多图处理
            elif video_type == "r2v" and images:
                if stream:
                    yield self._create_stream_chunk(f"上传 {image_count} 张参考图片...\n")

                for img in images:
                    media_id = await self.flow_client.upload_image(
                        token.at, img, model_config["aspect_ratio"], project_id=project_id
                    )
                    reference_images.append({
                        "imageUsageType": "IMAGE_USAGE_TYPE_ASSET",
                        "mediaId": media_id
                    })
                debug_logger.log_info(f"[R2V] 上传了 {len(reference_images)} 张参考图片")

            # ========== 调用生成API ==========
            if stream:
                yield self._create_stream_chunk("提交视频生成任务...\n")
            submit_started_at = time.time()

            # I2V: 首尾帧生成
            if video_type == "i2v" and start_media_id:
                if end_media_id:
                    # 有首尾帧
                    result = await self.flow_client.generate_video_start_end(
                        at=token.at,
                        project_id=project_id,
                        prompt=prompt,
                        model_key=model_config["model_key"],
                        aspect_ratio=model_config["aspect_ratio"],
                        start_media_id=start_media_id,
                        end_media_id=end_media_id,
                        use_v2_model_config=use_v2_model_config,
                        user_paygate_tier=normalized_tier,
                        token_id=token.id,
                        token_video_concurrency=token.video_concurrency,
                    )
                else:
                    # 只有首帧 - 需要去掉 model_key 中的 _fl
                    # 情况1: _fl_ 在中间 (如 veo_3_1_i2v_s_fast_fl_ultra_relaxed -> veo_3_1_i2v_s_fast_ultra_relaxed)
                    # 情况2: _fl 在结尾 (如 veo_3_1_i2v_s_fast_ultra_fl -> veo_3_1_i2v_s_fast_ultra)
                    actual_model_key = model_config["model_key"].replace("_fl_", "_")
                    if actual_model_key.endswith("_fl"):
                        actual_model_key = actual_model_key[:-3]
                    debug_logger.log_info(f"[I2V] 单帧模式，model_key: {model_config['model_key']} -> {actual_model_key}")
                    result = await self.flow_client.generate_video_start_image(
                        at=token.at,
                        project_id=project_id,
                        prompt=prompt,
                        model_key=actual_model_key,
                        aspect_ratio=model_config["aspect_ratio"],
                        start_media_id=start_media_id,
                        use_v2_model_config=use_v2_model_config,
                        user_paygate_tier=normalized_tier,
                        token_id=token.id,
                        token_video_concurrency=token.video_concurrency,
                    )

            # R2V: 多图生成
            elif video_type == "r2v" and reference_images:
                result = await self.flow_client.generate_video_reference_images(
                    at=token.at,
                    project_id=project_id,
                    prompt=prompt,
                    model_key=model_config["model_key"],
                    aspect_ratio=model_config["aspect_ratio"],
                    reference_images=reference_images,
                    user_paygate_tier=normalized_tier,
                    token_id=token.id,
                    token_video_concurrency=token.video_concurrency,
                )

            # Extend: 视频续写
            elif video_type == "extend":
                if not video_media_id:
                    error_msg = "❌ 视频续写需要提供源视频的 mediaGenerationId，请在 image_url 中传入 extend://VIDEO_MEDIA_ID"
                    if stream:
                        yield self._create_stream_chunk(f"{error_msg}\n")
                    self._mark_generation_failed(generation_result, error_msg)
                    yield self._create_error_response(error_msg, status_code=400)
                    return

                debug_logger.log_info(f"[EXTEND] 续写视频: {video_media_id}")
                if stream:
                    yield self._create_stream_chunk(f"视频续写任务提交中，源视频: {video_media_id[:8]}...\n")
                result = await self.flow_client.generate_video_extend(
                    at=token.at,
                    project_id=project_id,
                    prompt=prompt,
                    video_media_id=video_media_id,
                    model_key=model_config["model_key"],
                    aspect_ratio=model_config["aspect_ratio"],
                    user_paygate_tier=normalized_tier,
                    token_id=token.id,
                    token_video_concurrency=token.video_concurrency,
                )

            # T2V 或 R2V无图: 纯文本生成
            else:
                result = await self.flow_client.generate_video_text(
                    at=token.at,
                    project_id=project_id,
                    prompt=prompt,
                    model_key=model_config["model_key"],
                    aspect_ratio=model_config["aspect_ratio"],
                    use_v2_model_config=use_v2_model_config,
                    user_paygate_tier=normalized_tier,
                    token_id=token.id,
                    token_video_concurrency=token.video_concurrency,
                )
            if video_trace is not None:
                video_trace["submit_generation_ms"] = int((time.time() - submit_started_at) * 1000)

            # 获取task_id和operations
            operations = result.get("operations", [])
            if not operations:
                self._mark_generation_failed(generation_result, "\u751f\u6210\u4efb\u52a1\u521b\u5efa\u5931\u8d25")
                yield self._create_error_response("生成任务创建失败", status_code=502)
                return

            operation = operations[0]
            task_id = operation["operation"]["name"]
            scene_id = operation.get("sceneId")

            # 保存Task到数据库
            task = Task(
                task_id=task_id,
                token_id=token.id,
                model=model_config["model_key"],
                prompt=prompt,
                status="processing",
                scene_id=scene_id
            )
            await self.db.create_task(task)
            await self._update_request_log_progress(
                request_log_state,
                token_id=token.id,
                status_text="video_submitted",
                progress=45,
                response_extra={"task_id": task_id, "scene_id": scene_id},
            )

            # 轮询结果
            if stream:
                yield self._create_stream_chunk(f"视频生成中...\n")

            # 检查是否需要放大
            upsample_config = model_config.get("upsample")

            # 如果是 extend，传入源视频 media_id 用于后续拼接
            extend_source_id = video_media_id if video_type == "extend" else None
            async for chunk in self._poll_video_result(
                token,
                project_id,
                operations,
                stream,
                upsample_config,
                generation_result,
                response_state,
                request_log_state,
                extend_source_media_id=extend_source_id,
            ):
                yield chunk

        finally:
            pass

    async def _poll_video_result(
        self,
        token,
        project_id: str,
        operations: List[Dict],
        stream: bool,
        upsample_config: Optional[Dict] = None,
        generation_result: Optional[Dict[str, Any]] = None,
        response_state: Optional[Dict[str, Any]] = None,
        request_log_state: Optional[Dict[str, Any]] = None,
        extend_source_media_id: Optional[str] = None,
    ) -> AsyncGenerator:
        """轮询视频生成结果
        
        Args:
            upsample_config: 放大配置 {"resolution": "VIDEO_RESOLUTION_4K", "model_key": "veo_3_1_upsampler_4k"}
        """

        if response_state is None:
            response_state = self._create_response_state()

        max_attempts = config.max_poll_attempts
        poll_interval = config.poll_interval
        
        # 如果需要放大，轮询次数加倍（放大可能需要 30 分钟）
        if upsample_config:
            max_attempts = max_attempts * 3  # 放大需要更长时间

        consecutive_poll_errors = 0
        last_poll_error: Optional[Exception] = None
        max_consecutive_poll_errors = 3

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)

            try:
                result = await self.flow_client.check_video_status(token.at, operations)
                checked_operations = result.get("operations", [])
                consecutive_poll_errors = 0
                last_poll_error = None

                if not checked_operations:
                    continue

                operation = checked_operations[0]
                status = operation.get("status")

                # 状态更新 - 每20秒报告一次 (poll_interval=3秒, 20秒约7次轮询)
                progress_update_interval = 7  # 每7次轮询 = 21秒
                if stream and attempt % progress_update_interval == 0:  # 每20秒报告一次
                    progress = min(int((attempt / max_attempts) * 100), 95)
                    await self._update_request_log_progress(request_log_state, token_id=token.id, status_text="video_polling", progress=max(45, progress), response_extra={"upstream_status": status})
                    yield self._create_stream_chunk(f"生成进度: {progress}%\n")

                # 检查状态
                if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
                    # 成功
                    metadata, video_info, video_url = self._extract_video_details_from_operation(operation)
                    if not video_url:
                        if stream:
                            yield self._create_stream_chunk("视频已生成，正在等待下载链接...\n")
                        operation, metadata, video_info, video_url = await self._wait_for_video_url_after_success(
                            token,
                            checked_operations,
                            operation,
                            poll_interval,
                            request_log_state,
                            response_state,
                        )
                    # Extract short UUID from Google Storage URL (e.g., /video/UUID?)
                    # Both extend API and concat API need this short UUID format,
                    # NOT the CAUS base64 mediaGenerationId from video_info
                    import re as _re
                    _uuid_match = _re.search(r'/video/([0-9a-f-]{36})', video_url or '')
                    video_media_id = _uuid_match.group(1) if _uuid_match else video_info.get("mediaGenerationId", "")
                    aspect_ratio = video_info.get("aspectRatio", "VIDEO_ASPECT_RATIO_LANDSCAPE")

                    if not video_url:
                        error_msg = "视频生成失败: 视频URL为空"
                        error_extra = {
                            "upstream_metadata": metadata,
                            "upstream_operation": operation,
                        }
                        await self._fail_video_task([operation], error_msg)
                        self._mark_generation_failed(generation_result, error_msg, error_extra=error_extra)
                        yield self._create_error_response(
                            error_msg,
                            status_code=502,
                            extra=error_extra,
                        )
                        return

                    # ========== 视频放大处理 ==========
                    if upsample_config and video_media_id:
                        if stream:
                            resolution_name = "4K" if "4K" in upsample_config["resolution"] else "1080P"
                            yield self._create_stream_chunk(f"\n视频生成完成，开始 {resolution_name} 放大处理...（可能需要 30 分钟）\n")
                        
                        try:
                            # 提交放大任务
                            upsample_result = await self.flow_client.upsample_video(
                                at=token.at,
                                project_id=project_id,
                                video_media_id=video_media_id,
                                aspect_ratio=aspect_ratio,
                                resolution=upsample_config["resolution"],
                                model_key=upsample_config["model_key"],
                                token_id=token.id,
                                token_video_concurrency=token.video_concurrency,
                            )
                            
                            upsample_operations = upsample_result.get("operations", [])
                            if upsample_operations:
                                if stream:
                                    yield self._create_stream_chunk("放大任务已提交，继续轮询...\n")
                                
                                # 递归轮询放大结果（不再放大）
                                async for chunk in self._poll_video_result(
                                    token,
                                    project_id,
                                    upsample_operations,
                                    stream,
                                    None,
                                    generation_result,
                                    response_state,
                                    request_log_state,
                                ):
                                    yield chunk
                                return
                            else:
                                if stream:
                                    yield self._create_stream_chunk("⚠️ 放大任务创建失败，返回原始视频\n")
                        except Exception as e:
                            debug_logger.log_error(f"Video upsample failed: {str(e)}")
                            if stream:
                                yield self._create_stream_chunk(f"⚠️ 放大失败: {str(e)}，返回原始视频\n")

                    # ========== Extend 视频拼接 ==========
                    if extend_source_media_id and video_media_id:
                        try:
                            if stream:
                                yield self._create_stream_chunk("\n视频续写完成，正在拼接完整视频...\n")
                            debug_logger.log_info(f"[CONCAT] 开始拼接: original={extend_source_media_id[:12]}..., extend={video_media_id[:12]}...")
                            
                            # 提交拼接任务
                            concat_result = await self.flow_client.run_concatenation(
                                at=token.at,
                                original_media_id=extend_source_media_id,
                                extend_media_id=video_media_id,
                            )
                            
                            # 获取 operation name
                            concat_op = concat_result.get("operation", {}).get("operation", {}).get("name", "")
                            if concat_op:
                                if stream:
                                    yield self._create_stream_chunk("拼接任务已提交，等待完成...\n")
                                
                                # 轮询拼接状态
                                concat_status = await self.flow_client.poll_concatenation_status(
                                    at=token.at,
                                    operation_name=concat_op,
                                    timeout=300,
                                    poll_interval=3,
                                )
                                
                                concat_url = concat_status.get("outputUri", "")
                                if concat_url:
                                    # 如果是本地路径（/tmp/xxx.mp4），构造完整 URL
                                    if concat_url.startswith("/tmp/"):
                                        server_host = config.server_host or "0.0.0.0"
                                        server_port = config.server_port or 8000
                                        # 对外使用 localhost
                                        host = "localhost" if server_host == "0.0.0.0" else server_host
                                        concat_url = f"http://{host}:{server_port}{concat_url}"
                                    video_url = concat_url  # 替换为拼接后的完整视频 URL
                                    if stream:
                                        yield self._create_stream_chunk("✅ 视频拼接完成！返回 16s 完整视频\n")
                                    debug_logger.log_info(f"[CONCAT] 拼接成功: {concat_url[:80]}...")
                                else:
                                    if stream:
                                        yield self._create_stream_chunk("⚠️ 拼接完成但无 URL，返回续写片段\n")
                            else:
                                debug_logger.log_warning("[CONCAT] 拼接任务创建失败，返回续写片段")
                                if stream:
                                    yield self._create_stream_chunk("⚠️ 拼接任务创建失败，返回续写片段\n")
                        except Exception as e:
                            import traceback
                            debug_logger.log_error(f"[CONCAT] 拼接失败: {str(e)}")
                            debug_logger.log_error(f"[CONCAT] traceback: {traceback.format_exc()}")
                            if stream:
                                yield self._create_stream_chunk(f"⚠️ 拼接失败: {str(e)}，返回续写片段\n")
                            # 拼接失败不影响返回，继续使用 extend 片段的 URL

                    # 缓存视频 (如果启用)
                    local_url = video_url
                    is_local_encoded_video = "/tmp/encoded_video_" in (video_url or "")
                    if config.cache_enabled and not is_local_encoded_video:
                        await self._update_request_log_progress(request_log_state, token_id=token.id, status_text="caching_video", progress=92)
                        try:
                            if stream:
                                yield self._create_stream_chunk("正在缓存视频文件...\n")
                            cached_filename = await self.file_cache.download_and_cache(video_url, "video")
                            local_url = f"{self._get_base_url(response_state)}/tmp/{cached_filename}"
                            if stream:
                                yield self._create_stream_chunk("✅ 视频缓存成功,准备返回缓存地址...\n")
                        except Exception as e:
                            debug_logger.log_error(f"Failed to cache video: {str(e)}")
                            # 缓存失败不影响结果返回,使用原始URL
                            local_url = video_url
                            if stream:
                                cache_error = self._normalize_error_message(e, max_length=120)
                                yield self._create_stream_chunk(f"⚠️ 缓存失败: {cache_error}\n正在返回源链接...\n")
                    else:
                        if stream and not is_local_encoded_video:
                            yield self._create_stream_chunk("缓存已关闭,正在返回源链接...\n")

                    # 更新数据库
                    task_id = operation["operation"]["name"]
                    await self.db.update_task(
                        task_id,
                        status="completed",
                        progress=100,
                        result_urls=[local_url],
                        completed_at=time.time()
                    )

                    # 存储URL用于日志记录
                    response_state["url"] = local_url
                    response_state["generated_assets"] = {
                        "type": "video",
                        "final_video_url": local_url,
                        "mediaGenerationId": video_media_id,
                    }

                    # 返回结果
                    self._mark_generation_succeeded(generation_result)

                    if stream:
                        yield self._create_stream_chunk(
                            f"<video src='{local_url}' data-media-id='{video_media_id}' controls style='max-width:100%'></video>",
                            finish_reason="stop"
                        )

                    else:
                        yield self._create_completion_response(
                            local_url,  # 直接传URL,让方法内部格式化
                            media_type="video"
                        )
                    return

                elif status == "MEDIA_GENERATION_STATUS_FAILED":
                    # 生成失败 - 提取错误信息
                    error_info = operation.get("operation", {}).get("error", {})
                    error_code = error_info.get("code", "unknown")
                    error_message = error_info.get("message", "未知错误")
                    
                    # 更新数据库任务状态
                    await self._fail_video_task(
                        checked_operations,
                        f"{error_message} (code: {error_code})"
                    )
                    
                    # 返回友好的错误消息，提示用户重试
                    friendly_error = f"视频生成失败: {error_message}，请重试"
                    self._mark_generation_failed(generation_result, friendly_error)
                    if stream:
                        yield self._create_stream_chunk(f"❌ {friendly_error}\n")
                    yield self._create_error_response(friendly_error, status_code=502)
                    return

                elif status.startswith("MEDIA_GENERATION_STATUS_ERROR"):
                    # ??????
                    error_msg = f"视频生成失败: {status}"
                    await self._fail_video_task(checked_operations, error_msg)
                    self._mark_generation_failed(generation_result, error_msg)
                    yield self._create_error_response(error_msg, status_code=502)
                    return
                    
                elif status == "MEDIA_GENERATION_STATUS_ACTIVE" and attempt > 80:
                    # 如果持续4分钟（80次 * 3秒 = 240秒）依然是 ACTIVE 状态，则判定为卡死
                    error_msg = "视频生成超时 (上游卡顿超过4分钟，已自动取消)"
                    await self._fail_video_task(checked_operations, error_msg)
                    self._mark_generation_failed(generation_result, error_msg)
                    if stream:
                        yield self._create_stream_chunk(f"❌ {error_msg}\n")
                    yield self._create_error_response(error_msg, status_code=504)
                    return

            except Exception as e:
                last_poll_error = e
                consecutive_poll_errors += 1
                debug_logger.log_error(f"Poll error: {str(e)}")
                if consecutive_poll_errors >= max_consecutive_poll_errors:
                    error_msg = f"视频状态查询失败: {self._normalize_error_message(e)}"
                    await self._fail_video_task(operations, error_msg)
                    self._mark_generation_failed(generation_result, error_msg)
                    if stream:
                        yield self._create_stream_chunk(f"❌ {error_msg}\n")
                    yield self._create_error_response(error_msg, status_code=502)
                    return
                continue

        # 超时
        if last_poll_error is not None:
            error_msg = f"视频状态查询持续失败: {self._normalize_error_message(last_poll_error)}"
        else:
            error_msg = f"视频生成超时 (已轮询 {max_attempts} 次)"
        await self._fail_video_task(operations, error_msg)
        self._mark_generation_failed(generation_result, error_msg)
        yield self._create_error_response(error_msg, status_code=504)

    # ========== 响应格式化 ==========

    def _create_stream_chunk(self, content: str, role: str = None, finish_reason: str = None) -> str:
        """创建流式响应chunk"""
        import json
        import time

        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "flow2api",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        }

        if role:
            chunk["choices"][0]["delta"]["role"] = role

        if finish_reason:
            chunk["choices"][0]["delta"]["content"] = content
        else:
            chunk["choices"][0]["delta"]["reasoning_content"] = content

        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    def _create_completion_response(self, content: str, media_type: str = "image", is_availability_check: bool = False) -> str:
        """创建非流式响应

        Args:
            content: 媒体URL或纯文本消息
            media_type: 媒体类型 ("image" 或 "video")
            is_availability_check: 是否为可用性检查响应 (纯文本消息)

        Returns:
            JSON格式的响应
        """
        import json
        import time

        # 可用性检查: 返回纯文本消息
        if is_availability_check:
            formatted_content = content
        else:
            # 媒体生成: 根据媒体类型格式化内容为Markdown
            if media_type == "video":
                formatted_content = f"```html\n<video src='{content}' controls></video>\n```"
            else:  # image
                formatted_content = f"![Generated Image]({content})"

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "flow2api",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": formatted_content
                },
                "finish_reason": "stop"
            }]
        }

        return json.dumps(response, ensure_ascii=False)

    def _create_batch_image_completion_response(self, items: List[Dict[str, Any]]) -> str:
        """创建批量图片的非流式响应。"""
        import json
        import time

        content_blocks: List[str] = []
        for idx, item in enumerate(items):
            prompt = str(item.get("prompt") or "").strip()
            prefix = f"[{idx + 1}]"
            if prompt:
                prefix = f"{prefix} {prompt}"

            if item.get("status") == "succeeded" and item.get("url"):
                content_blocks.append(f"{prefix}\n![Generated Image]({item['url']})")
                continue

            error_message = self._normalize_error_message(item.get("error"), max_length=240)
            content_blocks.append(f"{prefix}\nGeneration failed: {error_message}")

        formatted_content = "\n\n".join(content_blocks)
        summary = self._build_batch_summary(items)
        first_url = self._get_first_successful_batch_url(items)
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "flow2api",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": formatted_content
                },
                "finish_reason": "stop"
            }],
            "images": items,
            "batch_summary": summary,
        }
        if first_url:
            response["url"] = first_url

        return json.dumps(response, ensure_ascii=False)

    def _create_error_response(
        self,
        error_message: str,
        status_code: int = 500,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """创建错误响应"""
        import json

        error = {
            "error": {
                "message": error_message,
                "type": "server_error" if status_code >= 500 else "invalid_request_error",
                "code": "generation_failed",
                "status_code": status_code,
            }
        }
        if extra:
            error.update(extra)

        return json.dumps(error, ensure_ascii=False)

    def _get_base_url(self, response_state: Optional[Dict[str, Any]] = None) -> str:
        """获取基础URL用于缓存文件访问"""
        # 已配置缓存访问域名时，始终优先使用它，避免被请求 Host/IP 覆盖。
        if config.cache_base_url:
            return config.cache_base_url.rstrip("/")

        request_base_url = ""
        if isinstance(response_state, dict):
            request_base_url = (response_state.get("base_url") or "").strip().rstrip("/")
        if request_base_url:
            return request_base_url

        # 回退到服务地址，避免把监听地址 0.0.0.0 / :: 直接返回给客户端
        server_host = (config.server_host or "").strip()
        if server_host in {"", "0.0.0.0", "::", "[::]"}:
            server_host = "127.0.0.1"

        return f"http://{server_host}:{config.server_port}"

    async def _update_request_log_progress(
        self,
        request_log_state: Optional[Dict[str, Any]],
        *,
        token_id: Optional[int] = None,
        status_text: str,
        progress: int,
        response_extra: Optional[Dict[str, Any]] = None,
    ):
        """?????????????"""
        if not isinstance(request_log_state, dict):
            return
        log_id = request_log_state.get("id")
        if not log_id:
            return

        incoming_progress = max(0, min(100, int(progress)))
        safe_progress = max(int(request_log_state.get("progress") or 0), incoming_progress)
        now = time.time()
        last_status_text = str(request_log_state.get("last_status_text") or "").strip()
        last_progress = int(request_log_state.get("last_progress") or 0)
        last_updated_at = float(request_log_state.get("last_progress_update_at") or 0)

        request_log_state["progress"] = safe_progress
        request_log_state["last_status_text"] = status_text
        request_log_state["last_progress"] = safe_progress
        payload = {
            "status": "processing",
            "status_text": status_text,
            "progress": safe_progress,
        }
        if isinstance(response_extra, dict):
            payload.update(response_extra)

        should_write = (
            safe_progress in (0, 100)
            or status_text != last_status_text
            or safe_progress >= last_progress + 5
            or (now - last_updated_at) >= 1.0
        )
        if not should_write:
            return

        request_log_state["last_progress_update_at"] = now

        try:
            await self.db.update_request_log(
                log_id,
                token_id=token_id,
                response_body=json.dumps(payload, ensure_ascii=False),
                status_code=102,
                duration=0,
                status_text=status_text,
                progress=safe_progress,
            )
        except Exception as e:
            debug_logger.log_error(f"Failed to update request log progress: {e}")

    async def _log_request(
        self,
        token_id: Optional[int],
        operation: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        status_code: int,
        duration: float,
        log_id: Optional[int] = None,
        status_text: Optional[str] = None,
        progress: Optional[int] = None,
    ):
        """???????????? log_id ????????"""
        try:
            effective_status_text = status_text or (
                "completed" if status_code == 200 else "failed" if status_code >= 400 else "processing"
            )
            effective_progress = progress
            if effective_progress is None:
                effective_progress = 100 if status_code == 200 else 0 if status_code >= 400 else 0
            effective_progress = max(0, min(100, int(effective_progress)))

            request_body = json.dumps(request_data, ensure_ascii=False)
            response_body = json.dumps(response_data, ensure_ascii=False)

            if log_id:
                await self.db.update_request_log(
                    log_id,
                    token_id=token_id,
                    operation=operation,
                    request_body=request_body,
                    response_body=response_body,
                    status_code=status_code,
                    duration=duration,
                    status_text=effective_status_text,
                    progress=effective_progress,
                )
                return log_id

            log = RequestLog(
                token_id=token_id,
                operation=operation,
                request_body=request_body,
                response_body=response_body,
                status_code=status_code,
                duration=duration,
                status_text=effective_status_text,
                progress=effective_progress,
            )
            return await self.db.add_request_log(log)
        except Exception as e:
            debug_logger.log_error(f"Failed to log request: {e}")
            return None
