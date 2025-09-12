from langchain_ollama import ChatOllama
from langchain.agents import AgentType, tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from typing import Dict, Optional
import json
import requests
from bs4 import BeautifulSoup
from langchain.output_parsers import StructuredOutputParser
from langchain_core.runnables import Runnable

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.model_schema import NewsModel

class SafeConversationMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        safe_outputs = {}
        for k, v in outputs.items():
            if not isinstance(v, str):
                try:
                    safe_outputs[k] = json.dumps(v, indent = 2, ensure_ascii = False)
                except Exception:
                    safe_outputs[k] = str(v)
            else:
                safe_outputs[k] = v
        super().save_context(inputs, safe_outputs)


class NewsScraperAgent:
    def __init__(self, model : str):
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOllama(model = model, temperature = 0, streaming = True, callbacks = [self.callback_handler])
        self.memory = SafeConversationMemory(
            memory_key = "chat_history",
            return_messages = True
        )
        self.parser = StructuredOutputParser.from_response_schemas(NewsModel.response_schema)
        self.agent = self.build_agent()

    def build_agent(self):
        """Builds a REACT Agent"""
        format_instructions = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        system_template = f"""
            You are a senior investment analyst expert in news scraping and summarizing/analzying.
            Given the text of a full news article, analyze with extensive financial knowledge. 

            Then provide:
            1. An in-depth qualitative summary which includes import text details about the company
            2. An in-depth quantitative summary which includes figures, metrics, or numerical values relevant to the company.
            3. An un-biased and objective outlook or insights into the company based on your analysis. 

            Here are some guidelines:
            * Qualitative Summary: **Only textual and descriptive details** about the company. Do NOT include any numbers, figures, percentages, or metrics here. Focus only on narrative information such as growth drivers, business impacts, product strategies, or qualitative statements.
            * Quantiative Summary: **Only numerical facts**. List all metrics, values, percentages, figures, or amounts. No narrative, no explanations.
            * Insight and Outlook: An objective and balanced analysis or outlook of the company’s situation, based on the qualitative and quantitative details.

            Always output your final answer strictly as JSON in this format:
            {format_instructions}

            Use the tools provided to accomplish your tasks.
            """
        return initialize_agent(
            tools = self.get_tools(),
            llm = self.llm,
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory = self.memory,
            verbose = True,
            agent_kwargs = {"system_message" : system_template},
            handle_parsing_errors = True        
        )

    def get_tools(self):
        @tool
        def fetch_url_content(url: str) -> str:
            """Fetches raw text content of a news article from the given url"""
            headers = {
                "User-Agent" : (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
                )
            }
            try:
                response = requests.get(url, timeout = 10, headers = headers)
                soup = BeautifulSoup(response.text, "html.parser")
                text = [p.get_text() for p in soup.find_all("p")]
                return "\n".join(text)
            except Exception as e:
                return f"Error fetching content: {str(e)}"
            
        return [fetch_url_content]
    
    def run(self, state: dict) -> dict:
        """Agent scrapes/analyzes article"""
        result = self.agent.invoke({"input" : state['url']})
        output = result['output']

        if isinstance(output, dict):
            parsed_result = output
        else:
            parsed_result = self.parser.parse(output)
        return parsed_result

if __name__ == "__main__":
    agent = NewsScraperAgent(model = "llama3.1")
    state = {'url' : "https://www.cnbc.com/2020/08/04/square-sq-earnings-q2-2020.html"}
    results = agent.run(state)
    print('Qualitative Summary: ', results['qualitative_summary'])
    print("\n")
    print('Quantitative Summary: ', results['quantitative_summary'])
    print("\n")
    print('Insights and Outlook', results['insight_outlook']) 