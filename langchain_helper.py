from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
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
    response = chain_with_key.invoke({
            "animal_type": animal_type,
            "pet_color": pet_color
        })
    
    # response is a string (for OpenAI) or an object with `.content` if ChatOpenAI is used
    return response
  #invoke method is used to call the chain with the input variables, (run this chain once with these inputs and gives me the result.)

if __name__ == "__main__":
    print(generate_pet_name("dog", "brown"))
