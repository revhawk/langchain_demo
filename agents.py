import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import OpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#os.environ["OPENAI_API_KEY"] = apikey

llm = OpenAI(temperature=0)

tools = load_tools(['wikipedia','llm-math'], llm)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#get the answer from commandline input
answer = input('Input Wikipedia Research Task\n')

agent_executor.invoke({'input': answer})

