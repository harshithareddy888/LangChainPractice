from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
#agent imports
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import langchain.agents as agents

# from langchain_core.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color): #whenever ypu add a parameter, you must add that in input_variables inside prompt_template_name as well
    llm = OpenAI(temperature=0.7)
    # prompt_template_name= PromptTemplate(input_variables=['animal_type', 'pet_color'], template="Suggest a creative and unique name for my {pet_color} {animal_type} pet.")
    # name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    
    # The new LangChain uses "Runnable" syntax instead of LLMChain
    prompt = PromptTemplate.from_template(
        "Suggest a creative and unique name for my {pet_color} {animal_type} pet."
    )
    chain = prompt | llm | StrOutputParser()  #pipe operator to connect prompt to llm
    # Wrap it to produce a named output like "pet_name"
    chain_with_key = RunnableLambda(lambda x: {"pet_name": chain.invoke(x)})
    response = chain_with_key.invoke({ #invoke method is used to call the chain with the input variables, (run this chain once with these inputs and gives me the result.)
            "animal_type": animal_type,
            "pet_color": pet_color
        })
    
    # response is a string (for OpenAI) or an object with `.content` if ChatOpenAI is used
    return response

 #older version code

# def langchain_agent():
#     llm=OpenAI(temperature=0.5)
#     tools=load_tools(['wikipedia','llm-math'], llm=llm) #can use multiplr types of tools which are available
#     #initalizing agent
#     agent = initialize_agent(
#         tools,
#         llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #this agent uses react framework to decide which tool to use based on the tools description
#         verbose=True
#     )
#     #specify the task you want the agent to perform
#     result = agent.run("Who is the president of the United States? What is his current age raised to the power of 0.23?")
#     return result

#newer version code  (langchain 0.1.0+ version)
def langchain_agent():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Define tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [wikipedia]

    # Load standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Create agent and executor (using the safer import)
    agent = agents.create_react_agent(llm, tools, prompt)
    executor = agents.AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the agent
    result = executor.invoke({
        "input": "Who is the president of the United States? What is his current age raised to the power of 0.23?"
    })

    return result

if __name__ == "__main__":
    # print(generate_pet_name("dog", "brown"))
    print(langchain_agent())
