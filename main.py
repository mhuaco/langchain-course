from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

#tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tools that searches over internet
#     Args:
#         quey: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent """

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the response with answer and sources"""

    answer:str = Field(description="The answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 remote job postings for support engineer with experience ai agents")})
    print(result)

if __name__ == "__main__":
    main()
