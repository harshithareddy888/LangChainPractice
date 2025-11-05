def langchain_agent():
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain import hub
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    # Initialize model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Define tools
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [wikipedia]

    # Load prompt from LangChain Hub
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Create agent and executor (new syntax for 1.0+)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run it
    result = executor.invoke({
        "input": "Who is the president of the United States? What is his current age raised to the power of 0.23?"
    })

    return result
