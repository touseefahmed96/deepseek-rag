# DocuMind AI

DocuMind AI is an intelligent document assistant that helps you analyze and extract information from research documents. This application uses Streamlit for the user interface and LangChain for document processing and language modeling.

## Features

- Upload and process PDF documents
- Split documents into smaller chunks for efficient processing
- Index documents for quick retrieval
- Generate answers to user queries based on the document content
- Customizable UI with dark mode and styled chat components

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/touseefahmed96/deepseek-rag.git
    cd deepseek-rag
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run deepseek_rag.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF document using the sidebar.

4. Ask questions about the document using the chat input.

## Code Overview

### Main Components

- **Document Upload and Processing**: Handles file uploads and processes PDF documents into smaller chunks.
- **Document Indexing**: Indexes document chunks for efficient retrieval.
- **Query Handling**: Finds related documents based on user queries and generates answers using a language model.

### Key Functions

- `save_uploaded_file(uploaded_file)`: Saves the uploaded file to the specified storage path.
- `load_pdf_documents(file_path)`: Loads PDF documents using `PDFPlumberLoader`.
- `chunk_documents(raw_document)`: Splits a raw document into smaller chunks.
- `index_documents(documents_chunks)`: Indexes document chunks into the document vector database.
- `find_related_documents(query)`: Finds related documents based on the given query.
- `generate_answer(user_query, context_document)`: Generates an answer to the user's query based on the provided context document.

### UI Configuration

- **Title and Description**: Sets the title and description of the application.
- **Sidebar**: Provides an interface for uploading PDF documents.
- **Chat Interface**: Allows users to input queries and displays responses from the assistant.

## Styling

The application includes custom CSS for a dark-themed UI, styled chat input, and message components.

## Models

- **Ollama**: Utilized for advanced language understanding and processing.
- **DeepSeek-R1 1.5B**: Employed for robust document analysis and query response generation.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [PDFPlumber](https://github.com/jsvine/pdfplumber)
- [Ollama](https://ollama.com/)
- [DeepSeek-R1 1.5B](https://deepseek.com/)
