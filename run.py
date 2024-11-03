import argparse
import json
import logging
import os
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings and set logging level
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

from litellm import completion
from salesgpt.agents import SalesGPT

# Optional LangSmith settings for tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = ""  # insert your project name here

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="SalesGPT configuration")

    # Add arguments
    parser.add_argument(
        "--config", type=str, help="Path to agent config file", default=""
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbosity for debugging", default=False
    )
    parser.add_argument(
        "--max_num_turns",
        type=int,
        help="Maximum number of turns in the sales conversation",
        default=10,
    )

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns

    # Set Groq API key
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

    # Initialize Groq model with Litellm
    llm = completion(
        model="groq/llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful sales assistant."}],
        temperature=0.2
    )

    # Load or set agent configuration
    if not config_path:
        print("No agent config specified, using a standard config")
        USE_TOOLS = True
        sales_agent_kwargs = {
            "verbose": verbose,
            "use_tools": USE_TOOLS,
        }

        if USE_TOOLS:
            sales_agent_kwargs.update(
                {
                    "product_catalog": "examples/sample_product_catalog.txt",
                    "salesperson_name": "Ted Lasso",
                }
            )

        sales_agent = SalesGPT.from_llm(llm, **sales_agent_kwargs)
    else:
        try:
            with open(config_path, "r", encoding="UTF-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the config file {config_path}.")
            exit(1)

        print(f"Agent config loaded: {config}")
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)

    # Seed agent
    sales_agent.seed_agent()
    print("=" * 10)
    
    # Start conversation loop
    cnt = 0
    while cnt < max_num_turns:
        cnt += 1
        if cnt == max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            break

        # Take a step in the conversation
        sales_agent.step()

        # Check if the conversation should end
        if "<END_OF_CALL>" in sales_agent.conversation_history[-1]:
            print("Sales Agent determined it is time to end the conversation.")
            break

        # Get user input and continue the conversation
        human_input = input("Your response: ")
        sales_agent.human_step(human_input)
        print("=" * 10)
