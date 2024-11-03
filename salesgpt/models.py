from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema.messages import AIMessageChunk
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.output import ChatGenerationChunk  # Đây là đường dẫn mới

from langchain_groq import ChatGroq

from litellm import completion, acompletion


class GroqCustomModel(ChatGroq):
    """A custom chat model using Groq's API.
    
    Example:
        .. code-block:: python
            model = GroqCustomModel(model="groq/llama3-8b-8192")
            result = model.invoke([HumanMessage(content="hello")])
    """

    model: str = "groq/llama3-8b-8192"
    system_prompt: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using Groq API.

        Args:
            messages: List of messages in the conversation
            stop: Optional stop sequences
            run_manager: Optional callback manager
        """
        last_message = messages[-1]

        print(messages)
        response = completion(
            model=self.model,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)
        content = response.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            raise NotImplementedError("Streaming not implemented")
        
        last_message = messages[-1]

        print(messages)
        response = await acompletion(
            model=self.model,
            messages=[{"content": last_message.content, "role": "user"}],
            max_tokens=1000,
        )
        print("output", response)
        content = response.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])