from typing import Callable, List
from langchain.prompts import BasePromptTemplate


class CustomPromptTemplateForTools(BasePromptTemplate):
    # Định nghĩa template và tools_getter như các thuộc tính lớp
    template: str
    tools_getter: Callable  # Callable nhận đầu vào và trả về danh sách các tool

    def format(self, **kwargs) -> str:
        # Lấy các bước trung gian (AgentAction, Observation tuples) và định dạng chúng
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Đặt agent_scratchpad với nội dung đã xử lý
        kwargs["agent_scratchpad"] = thoughts

        # Lấy danh sách tools thông qua tools_getter
        tools = self.tools_getter(kwargs.get("input", ""))
        
        # Tạo biến tools và tool_names dựa trên danh sách tools đã lấy được
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        
        # Trả về nội dung template đã định dạng với các biến thay thế
        return self.template.format(**kwargs)
