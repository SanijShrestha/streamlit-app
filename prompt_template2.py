from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

draft_email_prompt_template=PromptTemplate(
    template="""
    You are a helpful email assistant. Your task is to write a draft email for the folloing email.
    Topic: {email_topic}
    Recipient: {recipient}
    Name: {name}

    Give ma the draft email in the following json format:
    {{
    "draft_email": "This is draft email"
    }}
    """
)


model=ChatOpenAI(model="gpt-4o-mini")
chain=draft_email_prompt_template|model



grammar_chain_prompt_template=PromptTemplate(
template="""
    You are a helpful email assistant. Check and validate grammar for the following email.
    Email: {draft_email}
    Also humanize the email to make it more natural and readable.
    Give me the grammar in the folloeing json format:
    {{
        "grammar_check_complete": "This is the grammar check",
        "final_email": "This is the humanized email"
    }}

"""
)

grammar_chain=grammar_chain_prompt_template|model|JsonOutputParser()

combined_chain=(chain|grammar_chain)

subject_line_prompt=PromptTemplate(
    template="""
    You are a helpful email assistant.Generate a subject line for the following email.
    question: {question}

    Give me the subject line in the following json format:
    {{
        "subject_line:This is the subject line"
    }}
"""
)

subject_line_chain=subject_line_prompt|model|JsonOutputParser()

parallel_chain=RunnableParallel(
    {
        "combined_chain": combined_chain,
        "subject_line": subject_line_chain,
    }
)

