import streamlit as st 
# from dotenv import load_dotenv
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import speech_recognition as sr  # For voice input


# Accessing the OpenAI API key from secrets
openai_api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]

#function to read pdfs and extract text
def get_text_from_pdf(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfFileReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap= 200, #to not loose the meaning
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Key Components
# texts:

# The input texts for which embeddings are generated.
# OpenAIEmbeddings:

# Converts the input text into dense vector representations.
# FAISS.from_texts():

# A LangChain wrapper that:
# Computes embeddings for each text.
# Builds a FAISS index.
# Maps the original text to the embeddings for retrieval.

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chains = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory=memory      
    )
    return conversation_chains

def handle_userinput(user_question):
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    # max_history = 3
    # st.session_state.chat_history = st.session_state.chat_history[-(max_history * 2):]


    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
                    
                    
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please speak now...")
        audio = recognizer.listen(source)
        try:
            # Use Google Web Speech API to recognize speech
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the speech.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

    return None


def main():
    # load_dotenv()
    st.set_page_config(page_title='Voice-Powered PDF Chatbot', page_icon='📖🎤')
    st.write(css,unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

    
        
    st.header('Voice-Powered PDF Chatbot :books:')
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
# Add Record Voice button below the input field
    if st.button("Record Voice"):
        voice_input = get_voice_input()
        if voice_input:
            # Once the voice is converted to text, update the user input and process it
            st.session_state.voice_input = voice_input  # Store it in session state for reference
            st.text_input("Ask a question about your documents:", value=voice_input)  # Update input field with voice text
            handle_userinput(voice_input)  # Process the voice input just like typed input
    
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs= st.file_uploader("Upload your PDFs, then click on 'Process'!", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing:"):
                #get the pdf text
                raw_text = get_text_from_pdf(pdf_docs)
                # st.write(raw_text)
                         
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
            
                #create the vector store
                vector_store = get_vectorstore(text_chunks)
                
                
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

                

if __name__ == '__main__':
    main()
