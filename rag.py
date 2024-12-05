import streamlit as st
import openai
import tempfile
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from googletrans import Translator
from streamlit_chat import message

st.set_page_config(page_title="Legal Assistant Chatbot", page_icon="⚖️", layout="wide")

openai.api_key = 'Your api key' 


# Helper functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text


def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def get_embeddings(texts, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=texts, model=model)
        embeddings = np.array([data['embedding'] for data in response['data']], dtype=np.float32)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None


def build_sklearn_index(embeddings):
    try:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='cosine')
        nbrs.fit(embeddings)
        return nbrs
    except Exception as e:
        st.error(f"Error building sklearn index: {e}")
        return None


def find_most_relevant_chunk(query, nbrs, chunks, embeddings):
    query_embedding = get_embeddings([query])  
    if query_embedding is None:
        return None
    distances, indices = nbrs.kneighbors(query_embedding)
    return chunks[indices[0][0]]  


def generate_response(query, context, language="English"):
    try:
        system_message = {
            "role": "system",
            "content": f"You are a legal assistant specializing in constitutional law. Use the provided context to answer the query. Respond in {language}. Only provide answers related to the Indian Constitution and Indian Rights. Do not answer questions related to other countries' laws."
        }
        user_message = {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[system_message, user_message]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


def translate_text(text, target_language="en"):
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return text  # Return the original text if translation fails


def transcribe_audio(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name

        # Get language code from session state or default to 'en' for English
        language_code = st.session_state.get('language', 'en')

        response = openai.Audio.transcribe(
            model="whisper-1",
            file=open(temp_file_path, "rb"),
            language=language_code
        )
        return response['text']  # Return the transcribed text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None


def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "sklearn_index" not in st.session_state:
        st.session_state.sklearn_index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []


def refresh_chat():
    st.session_state.conversation = []
    st.session_state.sklearn_index = None
    st.session_state.chunks = []
    st.success("Chat refreshed! Start a new conversation.")


# Streamlit app main function
def main():
    st.markdown("<h1 style='text-align: center;'>⚖️ RAG Legal Assistant ⚖️</h1>", unsafe_allow_html=True)

    # Sidebar for language selection
    language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Tamil", "Telugu", "Malayalam"])

    # File upload (use local PDF path)
    LOCAL_PDF_PATH = r"C:\Users\HP\Downloads\Chatgptclone-main\law.pdf"
    try:
        with open(LOCAL_PDF_PATH, "rb") as file:
            pdf_text = extract_text_from_pdf(file)
            chunks = split_text_into_chunks(pdf_text)
            embeddings = get_embeddings(chunks)
            if embeddings is not None:
                sklearn_index = build_sklearn_index(embeddings)
                st.session_state.sklearn_index = sklearn_index
                st.session_state.chunks = chunks
                st.sidebar.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")

    # Audio transcription section
    st.header("Audio Input")
    audio_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if audio_file:
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(audio_file)
            if transcription:
                st.sidebar.success("Audio successfully transcribed, processing query...")

                # Process the transcribed query
                context = find_most_relevant_chunk(transcription, st.session_state.sklearn_index, st.session_state.chunks, None)
                response = generate_response(transcription, context, language=language)

                translated_response = translate_text(response, target_language=language)

                # Add the user and bot messages to the conversation history
                st.session_state.conversation.append({"role": "user", "content": transcription})
                st.session_state.conversation.append({"role": "bot", "content": translated_response})

    # Chat interface for text input
    st.subheader("💬 Ask a Legal Question")
    user_query = st.text_input("Enter your question here:")

    if st.button("Ask"):
        if not st.session_state.sklearn_index or not st.session_state.chunks:
            st.error("Please upload a PDF to enable question answering.")
        else:
            context = find_most_relevant_chunk(user_query, st.session_state.sklearn_index, st.session_state.chunks, None)
            response = generate_response(user_query, context, language=language)

            # Translate the response to the selected language
            translated_response = translate_text(response, target_language=language)

            # Add the user and bot messages to the conversation history
            st.session_state.conversation.append({"role": "user", "content": user_query})
            st.session_state.conversation.append({"role": "bot", "content": translated_response})

    # Display conversation history in an interactive chat format
    st.subheader("📜 Conversation History")
    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            message(chat['content'], is_user=True)
        else:
            message(chat['content'], is_user=False)

    st.markdown("<p style='font-size: 0.9em; text-align: center;'>Developed by Subhash Govindaraj</p>", unsafe_allow_html=True)

    # Button to refresh the chat
    if st.button("🔄 Refresh Chat"):
        refresh_chat()


if __name__ == "__main__":
    initialize_session_state()
    main()
