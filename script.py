from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import os

@tool
def meeting_planner(meeting_date: str , meeting_time: str, meeting_title: str):
    """
    This function is to be used to schedule a meeting.
    Following aguments are must to successfully execute the function:
    meeting_date: date as string in format "YYYY-MM-DD"
    meeting_time: time as string in format "HH:MM" in 24 hour format
    meting_title: title of the meeting as string
    """
    return (f"Meeting {meeting_title} scheduled on {meeting_date} at {meeting_time}")

def main(meeting_planner):

    load_dotenv()

    llm_summary = ChatOpenAI(model="YOUR MODEL",
                             api_key=os.getenv("API_KEY"),
                             verbose=1)

    prompt_summary = ChatPromptTemplate.from_template(
        """Summarize the following meeting transcipt in MINIMUM words.
        Provide summary in readable markdown format. Use pointers for important points.
        DO NOT MISS ANY IMPORTANT POINTS. Provide next actions if any.
        Transcript: {transcript}""")

    with open("transcript.txt", "r") as f:
        input_transcript = f.read()


    summary_chain = prompt_summary | llm_summary | StrOutputParser()

    # meeting_summary = summary_chain.invoke({"transcript":input_transcript})

    tools = [meeting_planner]

    llm_scheduler = ChatOpenAI(model="YOUR MODEL",
                            api_key=os.getenv("API_KEY")).bind_tools(tools,tool_choice="any")

    prompt_meeting = ChatPromptTemplate.from_template(
                    """You are a meeting scheduling agent.

                    Here is the summary of a meeting transcript:
                    {meeting_summary}

                    Identify meetings that are to be scheduled as next steps and use
                    the meeting_planner tool to schedule them.

                    {agent_scratchpad}
                    """)
    
    meeting_planner_agent = create_tool_calling_agent(
        llm = llm_scheduler,
        tools=tools,
        prompt = prompt_meeting
    )

    agent_executer = AgentExecutor(agent=meeting_planner_agent,
                                   tools = tools,
                                   verbose=True)
    
    meeting_planner = (
        {"meeting_summary": summary_chain}
        | agent_executer
        )
    
    response = meeting_planner.invoke({"transcript":input_transcript})

    print(response["output"])

if __name__ == "__main__":

    main(meeting_planner)
