from langchain_ollama import ChatOllama
from langchain.agents import AgentType, tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Optional
import json
import requests
from bs4 import BeautifulSoup
from utils.callback_handler import PrintCallbackHandler

class NewsScraperAgent:
    def __init__(self, model):
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOllama(model = model, temperature = 0, streaming = True, callbacks = [self.callback_handler])
        self.memory = ConversationBufferMemory(
            memory_key = "chat_history",
            return_messages = True
        )
        self.agent = self.build_agent()
        
    def build_agent(self):
        """Builds a REACT Agent"""
        system_template = """
        You are a senior investment analyst expert in news scraping and summarizing/analzying.
        First, when given a news link, read the news article and analyze in-depth.
        Second, provide a qualitative and quantitative summary of the news. 

        Here are some guidelines to write a summary.
        * Qualitative Summary should include important text details about the company
        * Quantiative Summary should include relevant figures, metrics, or numerical values that are important about the company. 

        Use the tools provided to accomplish your tasks. 
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}")
        ])

        return initialize_agent(
            tools = self.get_tools(),
            llm = self.llm,
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory = self.memory,
            verbose = True,
            agent_kwargs = {"prompt" : prompt}
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
        """Run the agent and update state"""
        result = self.agent.invoke({"input" : {state['url']}})
        streamed_text = self.callback_handler.get_output()

        print(streamed_text)
        return result

if __name__ == "__main__":
    agent = NewsScraperAgent(model = "llama3.1")
    state = {'url' : "https://www.cnbc.com/2020/08/04/square-sq-earnings-q2-2020.html"}
    results = agent.run(state)
    print(results)