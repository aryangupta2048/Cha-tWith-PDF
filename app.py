import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------------------------------


st.set_page_config(
    page_title="Chat with PDF",
    layout="wide"
)

st.title("📚 RAG Based Chat With PDF")

# -------------------------------------------------------


st.sidebar.title("API Configuration")

google_api_key = st.sidebar.text_input(
    "Google API Key",
    type="password"
)

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    st.sidebar.success("API Key Loaded ✅")
else:
    st.sidebar.info("Enter your Google API Key")

# Stop if API key not entered
if not google_api_key:
    st.stop()

# -------------------------------------------------------


uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a PDF file.")
    st.stop()

# Save uploaded PDF

save_dir = "pdf_files"
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, uploaded_file.name)

with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

st.success(f"Uploaded: {uploaded_file.name}")

# -------------------------------------------------------


@st.cache_data
def load_documents(path):
    loader = PyPDFLoader(path)
    return loader.load()

documents = load_documents(file_path)

# -------------------------------------------------------


@st.cache_data
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)

chunks = split_documents(documents)

# -------------------------------------------------------


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------------------------------------


@st.cache_resource
def create_vector_db(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore

vectorstore = create_vector_db(chunks, embeddings)

# -------------------------------------------------------


k_value = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=10,
    value=4
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": k_value}
)

# -------------------------------------------------------
# Gemini LLM
# -------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------------------------------------------
# Prompt
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
"""
Answer the question using ONLY the context below.

If the answer is not found in the context,
reply exactly:

"I don't know based on the document."

Context:
{context}

Question:
{question}
"""
)

# -------------------------------------------------------
# Format Documents
# -------------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------
# Build RAG Chain
# -------------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------------
# Ask Question
# -------------------------------------------------------

st.header("Ask Questions")

question = st.text_input("Enter your question")

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching document..."):

            response = rag_chain.invoke(question)

        st.subheader("Answer")

        st.write(response)
