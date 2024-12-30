import streamlit as st 
from st_audiorec import st_audiorec
import io
from pydub import AudioSegment
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import speech_recognition as sr  # For voice input

# Accessing the OpenAI API key from secrets
openai_api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]

# Function to read PDFs and extract text
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
        chunk_size=1000,
        chunk_overlap=200,  # to not lose meaning
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chains = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chains

def handle_userinput(user_question):
    # Clear chat history after each question
    st.session_state.chat_history = []

    # Process the user input and display the response
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history (only the latest message)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"<div style='background-color:#f1f1f1; padding:10px'>{message.content}</div>", unsafe_allow_html=True)
        else:
            st.write(f"<div style='background-color:#d1e7ff; padding:10px'>{message.content}</div>", unsafe_allow_html=True)

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
    st.set_page_config(page_title='Voice-Powered PDF Chatbot', page_icon='📖🎤')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

    st.header('Voice-Powered PDF Chatbot :books:')

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
        pdf_docs = st.file_uploader("Upload your PDFs, then click on 'Process'!", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing:"):
                # Get the pdf text
                raw_text = get_text_from_pdf(pdf_docs)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create the vector store
                vector_store = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()
