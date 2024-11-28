from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from typing import Optional, List

logger = get_logger(__name__)


class Qwen(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stream: bool = False,
        use_cache=False,
    ) -> None:
        assert stream is False, "Qwen does not support streaming"
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            use_cache=use_cache,
        )
        self.model_type = "qwen"

    def _chat(self, chat_messages, **kwargs):
        import dashscope

        response = dashscope.MultiModalConversation.call(
            model=self.model_name, messages=chat_messages, stream=self.stream
        )
        if response.status_code == 400:
            yield response.message
            return
        if self.stream:
            for chunk in response:
                yield chunk.output.choices[0].message.content[0]["text"]
        else:
            message = response.output.choices[0].message
            if hasattr(message, "content"):
                for data in message.content:
                    if "text" in data:
                        yield data["text"]
                yield ""
            else:
                yield ""

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append({"role": "system", "content": [{"text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"text": query},
                ],
            },
        )
        for img_path in image_paths:
            messages[-1]["content"].append({"image": "file://" + img_path})
        return messages


if __name__ == "__main__":
    model = Qwen(
        model_name="qwen-vl-plus", temperature=0.5, use_cache=False, stream=False
    )
    query = "tell me a joke about mars, it should be very funny"
    system_prompt = "Your task is to generate a joke that is very funny."
    messages = model.build_message(query, system_prompt=system_prompt)
    answer = model.infer(messages)

    query = "根据这张图片的内容，写一个笑话"
    messages = model.build_message(
        query, system_prompt=system_prompt, image_paths=["assets/test_1.jpg"]
    )
    answer = model.infer(messages)
