import streamlit as st
import os

# Modern LangChain core imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Third-party integrations
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Page Configuration
st.set_page_config(page_title="12th Physics AI Tutor", page_icon="⚛️", layout="centered")
st.title("⚛️ 12th Class Physics AI Tutor")
st.write("Ask any question from your FBISE 12th Class Physics Textbook!")

# 2. Access API Key securely from secrets.toml
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please add your GROQ_API_KEY to .streamlit/secrets.toml")
    st.stop()

# 3. Load Vector Database & Models
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Looks for the folder in the current directory (content/)
    vector_store = FAISS.load_local(
        "physics_faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    return retriever, llm

try:
    retriever, llm = load_rag_components()
except Exception as e:
    st.error(f"Failed to load FAISS Index: {e}")
    st.info("Ensure the folder 'physics_faiss_index' containing 'index.faiss' and 'index.pkl' is inside your project directory.")
    st.stop()

# 4. Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Construct Prompts
qa_system_prompt = (
    "You are an expert Physics Tutor specializing in the 12th-class textbook context.\n"
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 6. Modern Fixed LCEL Chain Pipeline
# Explicitly isolates the string query from the input dictionary to bypass the AttributeError
def route_retriever(input_dict):
    query_str = input_dict["input"]
    docs = retriever.invoke(query_str)
    return format_docs(docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(route_retriever)
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 7. Handle Chat History Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages on screen refresh
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 8. Chat Input & Response Generation
if user_query := st.chat_input("What is alternating current?"):
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing physics context..."):
            # Invoke using the dictionary structure
            answer = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(answer)
            
    # Track the chat history state
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))