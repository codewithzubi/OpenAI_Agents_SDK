import chainlit as cl
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig
from dotenv import load_dotenv,find_dotenv
import os

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="StudyAssistant",
    instructions="""
You are a smart and friendly Study Assistant, created especially for students. Your job is to help students with any type of study-related questions.

Guidelines:
- Always answer questions related to education, study topics, school, college, university, subjects, exams, assignments, and projects.
- Your answers must be clear, helpful, and easy to understand for students of all levels.
- You can help with subjects like Math, English, Science, Computer, History, Islamiat, Urdu, and more.
- Also support students with learning tips, motivation, study planning, and productivity.
- If someone asks a question that is not related to study or student life, politely say:  
  "I'm a study assistant. I can only help with questions related to students and education."
- Be polite, motivating, and supportive in tone.
"""
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello!! i am smart and friendly Study Assistant. How can i help you today.?").send()



@cl.on_message
async def handle_message(message:cl.Message):
    history = cl.user_session.get("history")


    history.append({"role":"user","content":message.content})
    result = await Runner.run(
        agent,
        input=history,
        run_config=config
        )
    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history",history)
    await cl.Message(content=result.final_output).send()




   
    