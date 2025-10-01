import os
from dotenv import load_dotenv
import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

load_dotenv() # I don't think I need this as it's covere in line 3
st.title('Chat with Document') #title of our web page
loader = TextLoader('./constitution.txt') # load the text doc
documents = loader.load()
#print(documents) # print to ensure document loaded correctly
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)
#st.write(chunks[0]) #used to test all is working so far
#st.write(chunks[1])
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)
#initalise OpenAI instance
llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature =0)
retriever = vector_store.as_retriever()

crc = ConversationalRetrievalChain.from_llm(llm, retriever)
#get question from user input
question = st.text_input('Input your question')

if question:
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    response = crc.invoke({'question':question, 'chat_history': st.session_state['history']})
    st.session_state['history'].append((question, response['answer']))
    st.write("Question :" + question + " Answer: " + response['answer'])
    for prompts in st.session_state ['history']:
        st.write("Question: " + prompts[0])
        st.write("Answer: " + prompts[1])
    



