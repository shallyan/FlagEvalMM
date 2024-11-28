import os
from PIL import Image
from typing import Optional, List, Any
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel

logger = get_logger(__name__)
try:
    import google.generativeai as genai
    from google.ai import generativelanguage as glm
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
except ImportError:
    logger.warning(
        "google-generativeai is not installed, please install it by `pip install google-generativeai`"
    )


class Gemini(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: str | None = None,
        max_tokens: int = 1024,
        api_key: str | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        use_cache=False,
        max_image_size: int = 4 * 1024 * 1024,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            use_cache=use_cache,
            max_image_size=max_image_size,
        )
        self.model_type = "gemini"
        self.chat_args = {"temperature": self.temperature, "max_tokens": max_tokens}
        api_key = api_key if api_key else os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def _chat(self, chat_messages: Any, **kwargs):
        config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens, temperature=self.temperature
        )
        response = self.client.generate_content(
            chat_messages,
            generation_config=config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        if (
            response.prompt_feedback.block_reason
            == glm.GenerateContentResponse.PromptFeedback.BlockReason.OTHER
        ):
            yield "Can not answer because of blocked."
            return

        finish_reason = response.candidates[0].finish_reason
        if (
            finish_reason == glm.Candidate.FinishReason.SAFETY
            or finish_reason == glm.Candidate.FinishReason.OTHER
        ):
            yield "Can not answer because of safety reasons."
            return
        yield response.text

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append(system_prompt)
        messages.append(query)
        for img_path in image_paths:
            im = Image.open(img_path).convert("RGB")
            messages.append(im)
        return messages
