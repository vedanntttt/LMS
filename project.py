import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback

# Function to load Gemini API key
def load_gemini_api_key():
    gemini_api_key = "AIzaSyCn6a211kLSvhzA2BWJ3BklZlVymwVBUSU"
    return gemini_api_key

# Function to process text into chunks and create a knowledge base
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

# Main function that ties everything together
def main():
    # Add some custom CSS for a better UI design
    st.markdown("""
    <style>
        body {
            background-color: #f4f7fc;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>label {
            font-size: 14px;
            color: #444;
        }
        .stFileUploader>label {
            font-size: 14px;
            color: #444;
        }
        .stTextArea>label {
            font-size: 14px;
            color: #444;
        }
        .stMarkdown {
            font-size: 14px;
            color: #444;
        }
        .stSubheader {
            font-size: 20px;
            font-weight: 600;
            color: #2d2d2d;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("📄 PDF Summarizer")
    st.markdown("""
    ## Welcome to the PDF Summarizer App!
    This app extracts the text from your PDF, splits it into smaller chunks, and uses **Gemini AI** to summarize the content for you in a few sentences.
    """)
    st.divider()

    try:
        # Set the API key for Gemini
        os.environ["GOOGLE_API_KEY"] = load_gemini_api_key()
    except ValueError as e:
        st.error(str(e))
        return

    # File upload input
    pdf = st.file_uploader('Upload your PDF Document', type='pdf', label_visibility='collapsed')

    # Number of sentences input
    n_sentences = st.number_input("Number of Sentences in Summary", value=8, min_value=3, max_value=100)

    if pdf is not None:
        # Show loader while processing
        with st.spinner("Processing PDF..."):
            # Reading the PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Create the knowledge base
            knowledgeBase = process_text(text)

            query = f"Summarize the content of the uploaded PDF file in approximately {n_sentences}-{n_sentences+1} sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

            if query:
                # Get the relevant documents for the query
                docs = knowledgeBase.similarity_search(query)
                
                # Use the Gemini model for generating the summary
                model = "gemini-1.5-pro"
                llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
                chain = load_qa_chain(llm, chain_type='stuff')

                # Run the chain to get the summary
                response = chain.run(input_documents=docs, question=query)

                # Display the results
                st.subheader('Summary Results:')
                st.markdown(f"<div style='background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);'>{response}</div>", unsafe_allow_html=True)

# Entry point for the Streamlit app
if __name__ == '__main__':
    main()

