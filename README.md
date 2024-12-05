# ⚖️ RAG Legal Assistant Chatbot ⚖️

**RAG Legal Assistant Chatbot** is a legal assistant tool that provides accurate, context-driven answers to legal queries based on the Indian Constitution. It supports multiple languages and allows users to interact via text or audio input. This chatbot is developed using OpenAI APIs and integrates features like document-based context retrieval and multilingual support.

---

## Features
- 📄 **PDF-based Context Retrieval**: Extracts text from legal documents and processes it into manageable chunks for answering queries.
- 🗣️ **Audio Input Support**: Transcribes audio files into text and provides answers based on the transcribed query.
- 🌐 **Multilingual Support**: Supports English, Hindi, Tamil, Telugu, and Malayalam for both input and output.
- 🤖 **AI-Powered Chat**: Uses OpenAI's GPT-3.5 Turbo model for generating context-aware responses.
- 🔍 **Most Relevant Chunk Finder**: Implements a machine learning model to find the most relevant document chunk for a given query.
- 🛠 **Easy-to-Use Interface**: Built with Streamlit for an intuitive and interactive user experience.
- 🔄 **Conversation History**: Displays the chat history for seamless interaction tracking.
- 🚀 **Customizable**: Extendable to include more features as required.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- A valid OpenAI API key
- Required Python libraries (listed in `requirements.txt`)

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SubhashGovindharaj/rag_legalbot.git
   cd rag_legalbot

Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Application:

bash
Copy code
streamlit run app.py
Access the App: Open your browser and go to http://localhost:8501.

File Structure
bash
Copy code
real_time_face_recognition/
├── app.py                 # Main application script
├── requirements.txt       # Required Python libraries
├── README.md              # Documentation
└── assets/                # (Optional) Additional resources
