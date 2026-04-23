import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Humanizer Tool", page_icon="✍️")
st.title("AI Humanizer Rephraser")

# Logic to handle API Key (maintains logic from cell 1)
# Checks environment variable first, provides a fallback input if missing.
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

if not os.environ.get("GROQ_API_KEY"):
    st.warning("Please provide a GROQ_API_KEY in your .env file or the sidebar to proceed.")
    st.stop()

# Initialize LLM (Logic from cell 1)
try:
    llm = ChatGroq(model="openai/gpt-oss-120b")
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

# Define the Prompt Template (Logic from cell 4)
# Exact prompt string as defined in the notebook
template_string = """
You are an expert text editor and paraphrasing assistant.

Your task is to rewrite the given content so that it reads naturally, clearly, and like it was written by a human, while preserving the original meaning and technical accuracy.

STRICT REQUIREMENTS:

1. Preserve meaning:
- Do NOT change facts, technical terms, model names, or key concepts.
- Do NOT introduce new information.

2. Improve readability:
- Simplify complex or rigid phrasing.
- Replace overly formal or robotic language with natural phrasing.

3. Sentence variation:
- Vary sentence lengths (mix short and long sentences).
- Restructure sentences instead of only replacing words.
- Combine or split sentences where appropriate.

4. Natural flow:
- Use smooth transitions between sentences.
- Avoid repetitive sentence structures.

5. Human-like tone:
- Use clear, conversational but professional language.
- Avoid unnecessary jargon unless required.

6. Output constraints:
- Maintain approximately the same length as the original text.
- Do not add bullet points unless present in input.
- Do not explain your changes.
- Output only the rewritten content.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_template(template_string)

# Default context from the notebook (Logic from cell 4)
default_context = """While the current iteration serves as a functional and
robust prototype, specific architectural constraints have been
identified for resolution in future deployment phases.

1) Database Scalability: To address concurrency limitations
inherent in the current file-based SQLite configuration,
which restricts write operations via database locking,
the system will be migrated to a server-based Relational
Database Management System (RDBMS) such as Post-
greSQL or MySQL. This transition will enable row-level
locking to support simultaneous user interactions.

2) Asynchronous Processing: The current synchronous
view architecture blocks worker threads during I/O-
bound API calls, limiting throughput. Future work in-
volves refactoring critical views to utilize asynchronous
syntax and implementing background task queues (e.g.,
Celery with Redis). This will allow long-running pro-
cesses, such as comprehensive research report genera-
tion, to execute without degrading web server respon-
siveness."""

# User Input Section
input_text = st.text_area("Enter Context to Humanize:", height=300)

# Execution Logic (Logic from cell 4 & 5)
if st.button("Humanize Text"):
    if input_text:
        with st.spinner("Humanizing..."):
            try:
                # Create the chain
                chain = prompt | llm | StrOutputParser()
                
                # Invoke the chain
                response = chain.invoke({"context": input_text})
                
                # Display output (Logic from cell 5 - utilizing markdown)
                st.subheader("Humanized Output")
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
    else:
        st.warning("Please enter some text to humanize.")