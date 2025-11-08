
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
import urllib.parse
import re

st.set_page_config(
    page_title="Project Samarth 🌾",
    page_icon="🤖",
    layout="wide"
)
st.title("🌾 Project Samarth: Tamil Nadu Agri-Climate Q&A")
st.caption("Ask me complex questions about Tamil Nadu's agricultural production and climate data (2016-2019).")


try:
    DB_URL = "postgresql://postgres.qqxsquonbtkhlcfcytdf:Vaishnavi#25@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    API_KEY = "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"
except Exception as e:
    st.error(f"Error in key configuration: {e}")
    st.stop()



classifier_prompt_template = """
You are a classifier. Your job is to classify the user's input into one of three categories:
'greeting', 'data_query', or 'off_topic'.
- 'greeting': For simple hellos, his, how are you.
- 'data_query': For any question related to agriculture, crops, rainfall, districts, production, or climate in Tamil Nadu.
- 'off_topic': For any other question (e.g., "What is your name?", "What's the weather tomorrow?", "Who is the prime minister?").

Respond with only one word: 'greeting', 'data_query', or 'off_topic'.

User Input:
{question}

Classification:
"""


sql_prompt_template = """
Based on the table schema below, write a single PostgreSQL SQL query to answer the user's question.
Your query must be for PostgreSQL.

--- IMPORTANT RULES ---
1.  **You MUST return only the SQL query and nothing else.**
2.  **DO NOT add any explanation, text, or preamble** like "Here is the SQL query:".
3.  **DO NOT wrap the query in markdown** like ```sql ... ```.
4.  **CRITICAL: You MUST wrap ALL column names in double-quotes (")** because they contain spaces or capital letters.
    For example: SELECT "District", "Total Production" FROM agriculture_production WHERE "Crop Type" = 'Rice';
5.  **CRITICAL CASTING: The "Total Production" and "Actual" columns are TEXT.
    Anytime you sort, sum, or compare these columns, you MUST cast them to a number.
    Example: `ORDER BY CAST("Total Production" AS numeric) DESC`
    Example: `WHERE CAST("Actual" AS numeric) > 1000`
6.  **NEW CRITICAL RULE: The "Year" column is TEXT (e.g., '2016-17'). You MUST compare it as a string.**
    **DO NOT cast the "Year" column to a number.**
    Example: `WHERE "Year" = '2018-19'`
7.  All data is for districts within TAMIL NADU only.
8.  The *only* available years are '2016-17', '2017-18', and '2018-19'.
9.  If the user asks for "last year" or "most recent year", you MUST use '2018-19'.
10. Table 'agriculture_production' has columns: "District", "Crop Type", "Year", "Total Production".
11. Table 'rainfall_data' has columns: "District", "Year", "Actual", "Normal".
---

Schema:
{schema}

Question:
{question}

SQL Query:
"""


response_prompt_template = """
You are a helpful assistant answering a user's question.
The user's question was:
{question}

A SQL query was run and it returned the following data. This data is the *direct answer* to the question.
Data:
{data}

Your job is to present this data as a clear, simple, natural language answer.
- **DO NOT** try to re-analyze the data.
- **DO NOT** add any extra information, thoughts, or contradictions (like "I need to check...").
- Just state the answer from the data clearly.

For example, if the data is `[('Thanjavur', 1441111)]`, your answer should be:
"The district with the highest rice production in 2018-19 was Thanjavur, with 14,41,111 metric tons."

Answer:
"""



@st.cache_resource(ttl=3600)
def get_llm():
    """Initializes and caches the Groq LLM."""
    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=API_KEY,
            temperature=0
        )
        return llm
    except Exception as e:
        st.error(f"Failed to connect to Groq. Check your API key. Error: {e}")
        st.stop()

@st.cache_resource(ttl=3600)
def get_db():
    """Initializes and caches the Supabase DB connection."""
    try:
        db = SQLDatabase.from_uri(DB_URL)
        return db
    except Exception as e:
        st.error(f"Failed to connect to Supabase. Check your DB_URL. Error: {e}")
        st.stop()



def get_classifier_chain():
    """
    Creates the chain that classifies user input.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(classifier_prompt_template)
    classifier_chain = prompt | llm | StrOutputParser()
    return classifier_chain

def get_full_qa_chain():
    """
    Creates and returns the full LangChain Q&A chain.
    """
    llm = get_llm()
    db = get_db()
    

    sql_prompt = ChatPromptTemplate.from_template(sql_prompt_template)
    sql_query_chain = (
        RunnablePassthrough.assign(schema=lambda x: db.get_table_info())
        | sql_prompt
        | llm
        | StrOutputParser()
    )


    get_sql_query = RunnablePassthrough.assign(
        generated_query=sql_query_chain
    )
    get_data = RunnablePassthrough.assign(
        data=lambda x: db.run(x["generated_query"])
    )
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    generate_response = (
        response_prompt
        | llm
        | StrOutputParser()
    )

    full_qa_chain = (
        get_sql_query
        | get_data
        | (lambda x: {"query": x["generated_query"], "response": generate_response.invoke(x)})
    )
    return full_qa_chain


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me a question about Tamil Nadu's agriculture or climate data (2016-2019)."
    }]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question, or say hello!"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            year_match = re.search(r'\b(19\d{2}|20[0-2]\d|2030)\b', prompt)
            invalid_year_found = False
            if year_match:
                year_found = year_match.group(1)
                if year_found not in ["2016", "2017", "2018", "2019"]:
                    invalid_year_found = True
            
            if invalid_year_found:
                response_text = "I'm sorry, I only have data for the years 2016-17, 2017-18, and 2018-19. Please ask a question about those years."
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            else:
                try:
                    classifier_chain = get_classifier_chain()
                    classification = classifier_chain.invoke({"question": prompt}).lower()
                    
                    classification = re.sub(r'[^a-z_]', '', classification)

                    if "greeting" in classification:
                        response_text = "Hello! How can I help you with questions about Tamil Nadu's agriculture or climate?"
                        st.markdown(response_text)

                    elif "data_query" in classification:
                        qa_chain = get_full_qa_chain()
                        result = qa_chain.invoke({"question": prompt})
                        
                        response_text = result["response"]
                        sql_query = result["query"]

                        with st.expander("Generated SQL Query"):
                            st.code(sql_query, language="sql")
                        
                        st.markdown(response_text)

                    else:
                        response_text = "I'm sorry, I'm only an assistant for questions about agriculture and rainfall in Tamil Nadu (2016-2019). Could you please ask a question related to that?"
                        st.markdown(response_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    error_message = f"I'm sorry, I encountered an error trying to understand that. Could you please rephrase your question more clearly?\n\n**Error details:**\n```\n{e}\n```"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

