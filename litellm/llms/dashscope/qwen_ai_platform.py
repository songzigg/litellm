from collections.abc import Mapping
import json
from typing import Final

import httpx
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from .chat.transformation import DashScopeChatConfig
from .common_utils import resolve_dashscope_family_rerank_api_base
from .embed.transformation import DashScopeEmbeddingConfig
from .image_generation.transformation import DashScopeImageGenerationConfig
from .rerank.transformation import DashScopeRerankConfig
from .videos.transformation import DashScopeVideoConfig

QWEN_AI_PLATFORM_API_BASE: Final = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_AI_PLATFORM_RERANK_API_BASE: Final = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
QWEN_AI_PLATFORM_IMAGE_API_BASE: Final = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
QWEN_AI_PLATFORM_VIDEO_API_BASE: Final = "https://dashscope.aliyuncs.com"
QWEN_AI_PLATFORM_TTS_API_BASE: Final = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
)


def _resolve_qwen_ai_platform_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("QWEN_AI_PLATFORM_API_KEY") or get_secret_str("DASHSCOPE_API_KEY")


def _require_qwen_ai_platform_api_key(api_key: str | None) -> str:
    resolved: Final = _resolve_qwen_ai_platform_api_key(api_key)
    if resolved is None:
        raise ValueError(
            "Qianwen AI Platform API key is required. Set 'QWEN_AI_PLATFORM_API_KEY' or 'DASHSCOPE_API_KEY' env var "
            "or pass api_key explicitly."
        )
    return resolved


class QwenAIPlatformChatConfig(DashScopeChatConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self._resolve_chat_api_base(api_base), _resolve_qwen_ai_platform_api_key(api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE") or QWEN_AI_PLATFORM_API_BASE


class QwenAIPlatformEmbeddingConfig(DashScopeEmbeddingConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_embedding_api_base(self, api_base: str | None) -> str:
        return api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE") or QWEN_AI_PLATFORM_API_BASE


class QwenAIPlatformRerankConfig(DashScopeRerankConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_rerank_api_base(self, api_base: str | None) -> str:
        return resolve_dashscope_family_rerank_api_base(
            api_base, "QWEN_AI_PLATFORM_API_BASE_RERANK", QWEN_AI_PLATFORM_RERANK_API_BASE
        )


class QwenAIPlatformImageGenerationConfig(DashScopeImageGenerationConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_image_api_base(self, image_api_base: str | None) -> str:
        return image_api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE_IMAGE") or QWEN_AI_PLATFORM_IMAGE_API_BASE


class QwenAIPlatformVideoConfig(DashScopeVideoConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return _require_qwen_ai_platform_api_key(api_key)

    def _resolve_video_api_base(self, video_api_base: str | None) -> str:
        return video_api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE_VIDEO") or QWEN_AI_PLATFORM_VIDEO_API_BASE


class QwenAIPlatformTextToSpeechConfig(BaseTextToSpeechConfig):
    """Token Plan Qwen TTS adapter for LiteLLM's standard /audio/speech API."""

    DEFAULT_VOICE: Final[str] = "longanhuan_v3.6"

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "sample_rate"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict]:
        selected_voice = voice if isinstance(voice, str) else None
        if selected_voice is None and isinstance(voice, dict):
            selected_voice = next(
                (voice.get(key) for key in ("voice_id", "id", "name") if isinstance(voice.get(key), str)),
                None,
            )
        # LiteLLM's dashboard connection check uses OpenAI's generic "alloy"
        # voice. Substitute the Token Plan documented default for that probe.
        if selected_voice == "alloy":
            selected_voice = self.DEFAULT_VOICE
        return selected_voice, {
            key: value
            for key, value in optional_params.items()
            if key in {"response_format", "sample_rate"}
        }

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key = _resolve_qwen_ai_platform_api_key(api_key)
        if not resolved_key:
            raise BaseLLMException(
                status_code=401,
                message="Qianwen AI Platform API key is required. Set QWEN_AI_PLATFORM_API_KEY or pass api_key.",
            )
        return {**headers, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        configured = api_base or get_secret_str("QWEN_AI_PLATFORM_API_BASE_TTS") or QWEN_AI_PLATFORM_TTS_API_BASE
        configured = configured.rstrip("/")
        # Provider resolution supplies the chat-completions base by default. TTS
        # models use the Token Plan SpeechSynthesizer endpoint unless overridden.
        if configured == QWEN_AI_PLATFORM_API_BASE:
            configured = QWEN_AI_PLATFORM_TTS_API_BASE
        if configured.endswith("/SpeechSynthesizer"):
            return configured
        return f"{configured}/api/v1/services/audio/tts/SpeechSynthesizer"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        if not voice:
            raise ValueError("voice is required for Qianwen AI Platform TTS")
        if voice == "alloy":
            voice = self.DEFAULT_VOICE
        fmt = optional_params.get("response_format", "mp3")
        request_input: dict[str, object] = {"text": input, "voice": voice, "format": fmt}
        sample_rate = optional_params.get("sample_rate")
        if sample_rate is not None:
            request_input["sample_rate"] = sample_rate
        return {
            "dict_body": {"model": model, "input": request_input},
            "headers": {"Content-Type": "application/json"},
        }

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> object:
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            response_data = raw_response.json()
        except (json.JSONDecodeError, ValueError):
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"Invalid JSON from Qianwen AI Platform TTS: {raw_response.text[:500]}",
                headers=dict(raw_response.headers),
            )
        audio_url = response_data.get("output", {}).get("audio", {}).get("url")
        if not isinstance(audio_url, str) or not audio_url:
            raise BaseLLMException(
                status_code=502,
                message=f"Qianwen AI Platform TTS response did not contain output.audio.url: {response_data}",
                headers=dict(raw_response.headers),
            )
        try:
            audio_response = httpx.get(audio_url, timeout=60.0)
            audio_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BaseLLMException(
                status_code=502,
                message=f"Failed to download temporary Qianwen TTS audio URL: {exc}",
                headers=dict(raw_response.headers),
            ) from exc
        audio_headers = {
            key: value
            for key, value in audio_response.headers.items()
            if key.lower() in {"content-type", "content-length"}
        }
        audio_headers.setdefault("content-type", "audio/mpeg")
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers=audio_headers,
                content=audio_response.content,
                request=raw_response.request,
            )
        )
