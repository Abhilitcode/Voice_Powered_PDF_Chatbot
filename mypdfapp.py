import streamlit as st 
# from dotenv import load_dotenv
from st_audiorec import st_audiorec
import io
from pydub import AudioSegment
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
    # Record audio in the browser using st_audiorec
    audio_bytes = st_audiorec()
    if audio_bytes:
        try:
            # Convert audio to a file-like object
            audio_file = io.BytesIO(audio_bytes)
            
            # Convert to AudioSegment format (ensure it is wav format)
            audio = AudioSegment.from_file(audio_file, format="wav")
            
            # Use speech_recognition to process the audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio.export(format="wav").read())) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)
                
                # Use Google Speech Recognition to transcribe the audio
                text = recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    
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

     # Displaying the notes
    st.subheader("Important Notes:")
    st.write("""
    1. **Upload your PDF documents first, and then enter your question.**
    2. **Use either text or voice input, but not both at the same time.**
       - If you use the microphone for the first question, please refresh the page and then use voice input.
       - Use only audio or only text input—basically, one at a time.
    """)
    
    # Text Input
    user_question = st.text_input("Ask a question about your documents:", key="text_input")
    if user_question:
        handle_userinput(user_question)

    # Voice Input
    st.subheader("Or Record Your Question:")
    voice_input = get_voice_input()
    if voice_input:
        st.session_state.voice_input = voice_input
        st.success(f"Recognized voice input: {voice_input}")
        handle_userinput(voice_input)
    
    
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
