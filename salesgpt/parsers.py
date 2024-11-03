import re
import os
from typing import Union

# Cập nhật import từ langchain
from langchain_core.output_parsers import BaseOutputParser  # Sử dụng BaseOutputParser thay vì AgentOutputParser
from langchain.prompts import PromptTemplate 
from langchain.schema import AgentAction, AgentFinish
from litellm import completion 

# Cập nhật class SalesConvoOutputParser
class SalesConvoOutputParser(BaseOutputParser):  # Thay AgentOutputParser bằng BaseOutputParser
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False
    model: str = "groq/llama3-8b-8192"  # Use Groq model

    def get_format_instructions(self) -> str:
        return "Your format instructions here"  # Cập nhật FORMAT_INSTRUCTIONS nếu cần

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(text)
            print("-------")
        
        # Cập nhật regex để bắt đầu chuỗi "Action" và "Action Input"
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        
        if not match:
            # Nếu không tìm thấy khớp, sử dụng Groq model cho quá trình parsing
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": text}]
            )
            parsed_text = response["choices"][0]["message"]["content"].strip()
            return AgentFinish(
                {"output": parsed_text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "sales-agent"