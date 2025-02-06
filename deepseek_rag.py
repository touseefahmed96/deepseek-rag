import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROMPT_TEMPLATE = """
Your are an expert research assistant. Use the provided document to answer the query.
If unsure, state that you dont know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = "document_store/pdfs/"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(
    model="deepseek-r1:1.5b",
    streaming=True,
)


def save_uploaded_file(uploaded_file):
    """
    Saves an uploaded file to the specified storage path.

    Args:
        uploaded_file (UploadedFile): The file object that has been uploaded.

    Returns:
        str: The file path where the uploaded file has been saved.
    """
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_documents(file_path):
    """
    Loads PDF documents from the specified file path using PDFPlumberLoader.

    Args:
        file_path (str): The path to the PDF file to be loaded.

    Returns:
        list: A list of documents loaded from the PDF file.
    """
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()


def chunk_documents(raw_document):
    """
    Splits a raw document into smaller chunks using a RecursiveCharacterTextSplitter.

    Args:
        raw_document (str): The raw document text to be split into chunks.

    Returns:
        list: A list of document chunks, each represented as a dictionary with keys
              such as 'text' and 'start_index'.
    """
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_processor.split_documents(raw_document)


def index_documents(documents_chunks):
    """
    Indexes the provided document chunks into the document vector database.

    Args:
        documents_chunks (list): A list of document chunks to be indexed.

    Returns:
        None
    """
    DOCUMENT_VECTOR_DB.add_documents(documents_chunks)


def find_related_documents(query):
    """
    Find related documents based on the given query.

    This function searches for documents in the DOCUMENT_VECTOR_DB that are similar to the provided query.

    Args:
        query (str): The search query to find related documents.

    Returns:
        list: A list of documents that are similar to the query.
    """
    return DOCUMENT_VECTOR_DB.similarity_search(query)


def generate_answer(user_query, context_document):
    """
    Generates an answer to the user's query based on the provided context document.
    Args:
        user_query (str): The query provided by the user.
        context_document (list): A list of document objects containing the context information.
    Yields:
        str: Chunks of the generated response.
    """
    context_text = "\n\n".join([doc.page_content for doc in context_document])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    # Stream responses in chunks
    response_generator = response_chain.stream(
        {
            "user_query": user_query,
            "document_context": context_text,
        }
    )

    # Yield each response chunk
    for chunk in response_generator:
        yield chunk


# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Sidebar for document upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_pdf = st.file_uploader(
        "Upload a Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False,
    )

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("🔎 Analyzing document..."):
            relevant_docs = find_related_documents(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()

            # Start streaming response
            response_stream = generate_answer(user_input, relevant_docs)
            response_placeholder.write_stream(response_stream)
