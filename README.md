# Educational AI Model: Multi-Mode Intelligent Assistant

## Overview

This repository contains an educational AI model designed to provide advanced question-answering and content understanding capabilities. The model can:

- Retrieve and summarize PDF documents (Retrieval-Augmented Generation, RAG).
- Generate intelligent responses to direct questions using a generative AI approach.
- Generate detailed captions and descriptions for images when provided with a URL.

This model is built with state-of-the-art NLP and vision transformers, making it a versatile tool for students, educators, and researchers across various domains.

## Features

### 1. Document Question-Answering (RAG Mode)

Upload a PDF document, and the model will:

- Parse and clean the text.
- Split it into semantically meaningful chunks.
- Create a vector-based knowledge base.
- Retrieve relevant information to answer specific questions based on the document content.

### 2. Generative Question-Answering

Provides detailed and technical answers to user queries using a specialized prompt structure for:

- Science
- Engineering
- Mathematics
- And other technical disciplines.

### 3. Image Captioning

- Generate detailed descriptions of images based on a provided URL.
- Answer specific questions related to the content of the image.



## Technologies Used

- **NLP Models**: SentenceTransformers (e.g., all-MiniLM-L6-v2) for semantic embeddings.
- **Generative AI**: Pretrained transformer models (e.g., Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit) for text generation.
- **Image Processing**: Vision-text models to describe image content.
- **Vector Databases**: ChromaDB for semantic search and retrieval.

### Libraries:
- langchain
- sentence-transformers
- transformers
- Pillow
- PyMuPDF

# Installation

## Clone the repository:
git clone <repository_url>


Install the required dependencies:
pip install -r requirements.txt

## Instructions

- Place the generative model weights in the specified path.
- Place the tokenizer in the specified path.

## Usage

The main application script can handle three modes:

1. Generative
2. RAG (Document-based Retrieval)
3. Image Captioning

Example Usage
```bash

from bedo import bedo

# Initialize the model
model = bedo()

# Generative Mode Example
question = "What is quantum mechanics?"
result = model(question, url=None, pdf_path=None, model_type="generative")
print(result)

# RAG Mode Example
pdf_path = "path/to/your/document.pdf"
question = "Explain the first law of thermodynamics."
result = model(question, url=None, pdf_path=pdf_path, model_type="rag")
print(result)

# Image Captioning Example
url = "https://example.com/your-image.jpg"
question = "What is happening in this image?"
result = model(question, url=url, pdf_path=None, model_type="image_captioning")
print(result)

```

## Flask Integration

This repository includes a Flask web application for interacting with the model via a web interface.

# Running the Flask Application

To start the Flask server, run the following command:

```bash
python app.py
```
Then, access the web interface at http://localhost:5400.

### File Structure
```bash

├── bedo.py               # Core implementation of the AI model
├── app.py                # Flask web application
├── requirements.txt      # Dependencies
├── templates/            # HTML templates for Flask
├── uploads/              # Folder for uploaded PDF files
├── README.md             # Documentation
└── examples/             # Example inputs and outputs

```
### Key Functions

#### retrieve_documents
- Splits and processes text from PDF documents.
- Creates a vector database and performs semantic search.

#### get_genrative_answer
- Generates responses to technical and scientific queries using structured prompts.

#### get_Rag_answer
- Combines retrieved document content with generative AI for precise answers.

#### describe_image_from_url
- Uses vision-language models to describe and interpret images.
