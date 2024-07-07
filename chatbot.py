# streamlit to create UI
import streamlit as st 

# using pypdf2 to read source PDF
from PyPDF2 import PdfReader

# using langchain to split data in smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# using langchain to generate embeddings
import openai
from langchain.embeddings.openai import OpenAIEmbeddings

# generatign vector store
from langchain.vectorstores.faiss import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


# OPENAI API KEY
OPEN_AI_API_KEY = "PERSONAL_OPENAI_API_KEY"

# Basic UI 
st.header("Abhir's Bot")
st.title("Abhir's Personal Chat Bot")


# UI to upload a PDF file
with st.sidebar:
    st.title("My Documents")
    file = st.file_uploader("Upload Your PDF to start Asking questions",type= "PDF")

text =""
# Extracting Data
if file:
    pdf_reader = PdfReader(file)
    # text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
    # st.write(text)


    # Breaking complete Data in smaller chunks
    text_Splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
    )

    chunks = text_Splitter.split_text(text)
    # st.write(chunks)


    #Generate Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPEN_AI_API_KEY)

    # creating vector store
    store = FAISS.from_texts(chunks,embeddings)

     # get user question
    user_question = st.text_input("Type Your question here")

    # do similarity search
    if user_question:
        match = store.similarity_search(user_question)
        #st.write(match)

        #defining the LLM
        llm = ChatOpenAI(
            openai_api_key = OPEN_AI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        #output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)



    




