import os
import re, json
from dotenv import load_dotenv

# Agent Setup and Structuring Output
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, tool, initialize_agent, AgentExecutor
from langchain.tools import BaseTool

# Import utils function and models
from utils.callback_handler import PrintCallbackHandler
from utils.conversation_buffer_safe import SafeConversationMemory

load_dotenv()

class PreprocessAgent:
    def __init__(self, model: str) -> None:
        self.callback_handler = PrintCallbackHandler()
        self.llm = ChatOpenAI(
            model = model, 
            temperature = 0, 
            streaming = True, 
            callbacks = [self.callback_handler],
            api_key = os.getenv("OPENAI_API_KEY")
        )
        self.memory = SafeConversationMemory(
            memory_key = "chat_history",
            return_messages = True
        )
        self.agent = self.build_agent()
    
    def build_agent(self) -> AgentExecutor:
        """Builds a REACT Agent"""
        system_template : str = f"""
        You are a preprocessing agent specialized in finance and research.
        Your task is to classify a user's query into exactly one of the following categories:
        - "finance-company": Query is micro-level, specifically about one company or a set of companies 
        (e.g., earnings, stock prices, financial reports, competitor comparisons).
        - "finance-market": Query is macro-level, about the general economy, sectors, or overall market trends 
        (e.g., interest rates, inflation, market indices).
        - "finance-education": Query is asking for explanations or definitions of financial terms or concepts 
        (e.g., "What is P/E ratio?", "Explain short selling").
        - "irrelevant": Query has nothing to do with finance, or is nonsense. 

        ### Guidelines
        - Do not provide an answer to the query itself — only categorize it.
        - Choose exactly one category for each query.
        - If the query is unclear but still finance-related, assign the best matching category.
        - If the query mixes finance and non-finance topics, prioritize finance.

        ### Examples
        - Query: "Show me Apple's Q2 2023 earnings" → finance-company  
        - Query: "Create me a table comparing Apple and its competitors" → finance-market
        - Query: "What is a P/E ratio?" → finance-education  
        - Query: "Tell me a simple joke" → irrelevant  

        ### Output Format
        Always output your final answer **strictly as JSON** by returning it as a category. 

        Use the available tools to accomplish your tasks. 
        """

        return initialize_agent(
            llm = self.llm,
            tools = self.get_tools(),
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory = self.memory,
            verbose = True,
            agent_kwargs = {"system_message" : system_template},
            handle_parsing_errors = True
        )
    
    def get_tools(self) -> list[BaseTool]:
        @tool
        def do_nothing(query: str) -> str:
            """Don't use this tool for now. It's just for testing"""
            return ""
        
        return [do_nothing]

    def run(self, state: dict[str, str]) -> dict[str, str]:
        """Agent categorizes the query"""
        result = self.agent.invoke({"input" : state['query']})
        return {'query' : state['query'], 'category' : result['output']}

    
if __name__ == "__main__":
    agent = PreprocessAgent(model = "gpt-4o-mini")
    state = {"query" : "Should I invest in Tesla or Apple?"}
    results = agent.run(state)
    print(results)



    

