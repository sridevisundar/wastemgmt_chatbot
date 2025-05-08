# Waste Management Chatbot

An intelligent waste management system that integrates AI-driven image classification and a chatbot for efficient waste categorization and awareness.

## Features

- Waste Classification: Utilizes a fine-tuned MobileNetV2 model with 94% accuracy to classify waste images into categories.
- AI Chatbot: Powered by the Groq API using LLaMA 8B model with Retrieval-Augmented Generation (RAG) for providing informative responses related to waste management.
- Streamlit Interface: Clean, interactive frontend built using Streamlit for both chatbot interaction and image upload.
- Dockerized Deployment: Easily containerized and portable using Docker.

## Tech Stack

- Model: MobileNetV2 (for classification), LLaMA 8B via Groq API (for chatbot)
- LLM Retrieval: RAG (Retrieval-Augmented Generation)
- Frontend: Streamlit
- Containerization: Docker
- Frameworks: PyTorch, sentence-transformers, ChromaDB

## How It Works

1. User uploads an image of waste.
2. Image is classified into a waste category using the MobileNetV2 model.
3. User can chat with the bot to ask about how to dispose of the item or related queries.
4. The chatbot retrieves context from a knowledge base using RAG and answers using LLaMA 8B via Groq API.

## Getting Started

### Prerequisites

- Python 3.12+
- Docker
- Groq API Key
- ChromaDB or similar vector database

### Installation

1. *Clone the Repository*

   git clone https://github.com/yourusername/waste-management-chatbot.git
   cd waste-management-chatbot

2. Replace the your-key-here with groq api key in app.py:

   GROQ_API_KEY=your_key_here


3. Build and Run with Docker

   docker build -t waste-chatbot .
   docker run -p 8501:8501 waste-chatbot


4. Access the App Open your browser and go to: http://localhost:8501

###License

This project is licensed under the MIT License.

Acknowledgements

Groq API

Streamlit

PyTorch

HuggingFace Transformers
