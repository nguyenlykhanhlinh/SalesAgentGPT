import os
from dotenv import load_dotenv
from litellm import completion  # Sử dụng litellm để tương tác với mô hình Groq
from salesgpt.agents import SalesGPT

# Tải biến môi trường từ file .env để lấy GROQ_API_KEY
load_dotenv()

# Đảm bảo có biến môi trường chứa API key của Groq
api_key = os.getenv('GROQ_API_KEY')
if api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Thay đổi cấu hình của model để dùng mô hình của Groq
model_name = "groq/llama3-8b-8192"

# Định nghĩa lại SalesGPT để sử dụng Groq model
class GroqSalesAgent(SalesGPT):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    async def async_step(self, stream=False):
        # Sử dụng litellm completion với stream để tương tác theo thời gian thực
        response = await completion(
            model=self.model_name,
            messages=[{"role": "user", "content": "Gợi ý câu hỏi cho mô hình tại đây"}],
            stream=stream
        )
        return response

# Khởi tạo đối tượng SalesAgent với mô hình Groq
sales_agent = GroqSalesAgent(
    model_name=model_name,
    salesperson_name="Ted Lasso",
    salesperson_role="Sales Representative",
    company_name="Sleep Haven",
    company_business="""Sleep Haven 
                            is a premium mattress company that provides
                            customers with the most comfortable and
                            supportive sleeping experience possible. 
                            We offer a range of high-quality mattresses,
                            pillows, and bedding accessories 
                            that are designed to meet the unique 
                            needs of our customers."""
)

# Khởi tạo trạng thái agent
sales_agent.seed_agent()

# Hàm để thực hiện và in từng chunk của output
async def run_streaming():
    async for chunk in sales_agent.async_step(stream=True):
        print(chunk)

# Thực hiện hàm
import asyncio
asyncio.run(run_streaming())
