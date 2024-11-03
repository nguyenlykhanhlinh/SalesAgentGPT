from copy import deepcopy
from typing import Any, Callable, Dict, List, Union
from tenacity import retry, stop_after_attempt, wait_fixed  # Dùng tenacity cho retry logic

# LangChain v3 imports
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from litellm import completion, acompletion
from pydantic import Field

# Local imports
from salesgpt.chains import SalesConversationChain, StageAnalyzerChain
from salesgpt.custom_invoke import CustomAgentExecutor
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base

# Tạo một retry decorator tùy chỉnh bằng tenacity
def _create_retry_decorator(max_retries: int) -> Callable:
    """
    Tạo một decorator để xử lý việc retry cho các lỗi API

    Args:
        max_retries (int): Số lần retry tối đa.

    Returns:
        Callable: Một decorator để retry.
    """
    return retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))  # retry với mỗi giây đợi

class SalesGPT(Chain):
    """Controller model cho Sales Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_agent_executor: Union[CustomAgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    model_name: str = "groq/llama3-8b-8192"  # Sử dụng mô hình Groq

    use_tools: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        """
        Lấy stage cuộc hội thoại dựa trên key.
        """
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @time_logger
    def seed_agent(self):
        """Thiết lập trạng thái cuộc hội thoại ban đầu."""
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        """Xác định stage hội thoại dựa trên lịch sử."""
        print(f"Stage trước khi phân tích: {self.conversation_stage_id}")
        print("Lịch sử hội thoại:")
        print(self.conversation_history)
        stage_analyzer_output = self.stage_analyzer_chain.invoke(
            input={
                "conversation_history": "\n".join(self.conversation_history).rstrip("\n"),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join([f"{key}: {value}" for key, value in CONVERSATION_STAGES.items()]),
            },
            return_only_outputs=False,
        )
        print("Stage sau phân tích")
        print(stage_analyzer_output)
        self.conversation_stage_id = stage_analyzer_output.get("text")
        self.current_conversation_stage = self.retrieve_conversation_stage(self.conversation_stage_id)

        print(f"Stage hiện tại: {self.current_conversation_stage}")

    @time_logger
    async def adetermine_conversation_stage(self):
        """Xác định stage hội thoại (async)."""
        print(f"Stage trước khi phân tích: {self.conversation_stage_id}")
        print("Lịch sử hội thoại:")
        print(self.conversation_history)
        stage_analyzer_output = await self.stage_analyzer_chain.ainvoke(
            input={
                "conversation_history": "\n".join(self.conversation_history).rstrip("\n"),
                "conversation_stage_id": self.conversation_stage_id,
                "conversation_stages": "\n".join([f"{key}: {value}" for key, value in CONVERSATION_STAGES.items()]),
            },
            return_only_outputs=False,
        )
        print("Stage sau phân tích")
        print(stage_analyzer_output)
        self.conversation_stage_id = stage_analyzer_output.get("text")
        self.current_conversation_stage = self.retrieve_conversation_stage(self.conversation_stage_id)

        print(f"Stage hiện tại: {self.current_conversation_stage}")

    # Thêm các method cần thiết khác tại đây.
