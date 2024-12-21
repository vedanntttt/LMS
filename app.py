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

# def load_openai_api_key():
#     openai_api_key = "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"
#     if not openai_api_key:
#         raise ValueError(f"Unable to retrieve OPENAI_KEY")
#     return openai_api_key

def load_gemini_api_key():
    gemini_api_key = "AIzaSyCn6a211kLSvhzA2BWJ3BklZlVymwVBUSU"
    return gemini_api_key

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

def main():
    # st.title("📄PDF Summarizer")
    st.image(r"C:\Users\Vedant\Desktop\python\PDF-Summarizer\banner.png")
    st.write("*Created by Vedant Yadav | Rohan Bhadkumbhe | Vaibhav Wadekar | Varad Bhagvat*")
    st.divider()

    try:
        # os.environ["OPENAI_API_KEY"] = load_openai_api_key()
        os.environ["GOOGLE_API_KEY"] = load_gemini_api_key()
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    n_sentences = st.number_input("number of sentences", value=8, min_value=3, max_value=100)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = f"Summarize the content of the uploaded PDF file in approximately {n_sentences}-{n_sentences+1} sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledgeBase.similarity_search(query)
            model = "gemini-1.5-pro"
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
            # llm = ChatOpenAI(model=model, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            # with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
                # print(cost)

            st.subheader('Summary Results:')
            st.write(response)


if __name__ == '__main__':
    st.set_page_config(page_icon="🤖", page_title="PDF Summarizer")
    main()

# python -m streamlit run app.py