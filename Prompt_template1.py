from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

draft_email_prompt_template=PromptTemplate(
    template="""
    You are a helpful email assistant. Your task is to write a draft email for the folloing email.
    Topic: {email_topic}
    Recipient: {recipient}
    Name: {name}
    """
)

model=ChatOpenAI(model="gpt-4o-mini")
chain=draft_email_prompt_template|model


response=chain.invoke({
    "email_topic": "New Product Launch",
    "recipient": "John Doe",
    "name": "Sanij",
})

print(response.content)
