import os
import streamlit as st
from dotenv import load_dotenv

# Import specific LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialization and Setup (Runs only once) ---

@st.cache_resource
def setup_rag_components(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Loads document, splits it, creates embeddings, and initializes the CRC chain.
    Uses st.cache_resource to ensure this expensive setup only runs once.
    """
    try:
        # Load, Split, and Embed
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)

        # Initialize LLM and Chain
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
        retriever = vector_store.as_retriever()
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        return crc
    except Exception as e:
        st.error(f"Error during RAG setup: {e}")
        return None

# Check if setup is done; if not, run it and initialize session state
if 'crc' not in st.session_state:
    st.session_state['crc'] = setup_rag_components('./constitution.txt')
    st.session_state['history'] = []
    st.session_state['question_key'] = 0  # Used to force input clearing

# Exit if setup failed
if st.session_state['crc'] is None:
    st.stop()


# --- 2. Callback Function for Submission ---

def submit_question():
    """
    Handles question submission, API call, history update, and input clearing.
    This function is called when the user presses Enter or clicks 'Submit'.
    """
    # Get the question using its unique key
    question = st.session_state.get(f'user_question_{st.session_state["question_key"]}')

    if question:
        with st.spinner("Thinking..."):
            # Invoke the Conversational Retrieval Chain
            response = st.session_state['crc'].invoke({
                'question': question,
                'chat_history': st.session_state['history']
            })

            # Update history with the new Q&A pair
            st.session_state['history'].append((question, response['answer']))

            # Increment the key to force Streamlit to render an empty input field next time
            st.session_state['question_key'] += 1


# --- 3. Streamlit UI Layout ---

st.title('Chat with Document')
st.markdown("Ask questions about the U.S. Constitution.")

# Input Field and Submission Button
st.text_input(
    'Input your question',
    key=f'user_question_{st.session_state["question_key"]}',
    on_change=submit_question,  # Submits when Enter is pressed
    placeholder="e.g., Who signed the constitution?"
)
st.button('Submit', on_click=submit_question) # Submits when button is clicked

st.markdown("---")
st.subheader("Chat History")

# Display History (Ensures no double display)
if st.session_state['history']:
    # Display in reverse order (newest at the top)
    for q, a in reversed(st.session_state['history']):
        st.markdown(f"**Q:** {q}")
        st.info(a)
        st.markdown("---")
else:
    st.markdown("Your conversation history will appear here.")