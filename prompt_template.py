from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

email_prompt_template=PromptTemplate(
    input_variables=["tone"],
    template="""
    You are a helpful email assistant. Your task is to write email in the {tone} tone
    to tge {recipient} about the {subject}
    """
)

model=ChatOpenAI(model="gpt-4o-mini")
chain=email_prompt_template|model


response=chain.invoke({
    "tone": "formal",
    "recipient": "John Doe",
    "subject": "Meeting",
})

print(response.content)