import anthropic
from typing import Optional, List, Any
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image
from anthropic.resources.messages import NOT_GIVEN

logger = get_logger(__name__)


class Claude(BaseApiModel):
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
        self.model_type = "claude"
        self.chat_args = {"temperature": self.temperature, "max_tokens": max_tokens}

        self.client = anthropic.Anthropic(api_key=api_key)

    def _chat(self, chat_messages: Any, **kwargs):
        system_prompt = (
            chat_messages.pop(0)["content"]
            if chat_messages[0]["role"] == "system"
            else NOT_GIVEN
        )
        chat_args = self.chat_args.copy()
        chat_args.update(kwargs)
        if self.stream:
            with self.client.messages.stream(
                system=system_prompt,
                messages=chat_messages,
                model=self.model_name,
                **chat_args,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            response = self.client.messages.create(
                system=system_prompt,
                messages=chat_messages,
                model=self.model_name,
                **chat_args,
            )
            yield response.content[0].text

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        )
        for img_path in image_paths:
            base64_image = encode_image(
                img_path, max_size=self.max_image_size, max_long_side=8000
            )

            media_type = "image/jpeg"
            messages[-1]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
            )
        return messages
