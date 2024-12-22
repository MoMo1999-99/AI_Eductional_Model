from langchain_community.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText
from langchain.embeddings import SentenceTransformerEmbeddings
from PIL import Image
import requests
import re

class bedo(object):
    def __init__(self):
        self.model_name = "/llama3_MODEL"
        self.tokenizer_name = "/llama3_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.image_model = AutoModelForImageTextToText.from_pretrained("image_model")
        self.processor = AutoProcessor.from_pretrained("image_processor")
        self.embedding_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
        self.PERSIST_DIRECTORY="chromadb"

    def clean_text(self,text):
        # Remove hashtags and words that start with a hash (#)
        text = re.sub(r'#\w+', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)        
        # Remove special characters and punctuation (except for spaces and alphanumeric characters)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)       
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()        
        # Optional: Convert text to lowercase
        text = text.lower()       
        return text
    # Function for document retrieval
    def retrieve_documents(self,data, query, chunk_size=500, chunk_overlap=250, k=3):
        """
        Retrieves documents based on a user's query using semantic search.
        """
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Clean and prepare the documents
        texts = [self.clean_text(str(d.page_content)) for d in data]
        metadatas = [d.metadata for d in data]
        
        # Split the documents into chunks
        semantic_chunker = SemanticChunker(self.embedding_model, breakpoint_threshold_type="percentile")
        semantic_chunks = semantic_chunker.create_documents(texts, metadatas=metadatas)

        # Create a vector database using Chroma
        vectordb = Chroma.from_documents(documents=semantic_chunks, embedding=self.embedding_model, persist_directory=self.PERSIST_DIRECTORY)

        # Perform similarity search with the provided query
        similar_docs = vectordb.similarity_search(query, k=k)

        # Extract the content of the retrieved documents
        retrieved_documents = [result.page_content for result in similar_docs]

        # Join the retrieved documents into a single string
        return " ".join(retrieved_documents)

    # New function for generating answers with structured prompt
    def get_genrative_answer(self,question: str) -> str:
        prompt = f'''
    ### Character:
    You are an AI model specialized in answering questions related to the fields of Science and Engineering. You possess expert knowledge in various scientific domains including Physics, Chemistry, Biology, Mathematics, Mechanical Engineering, Electrical Engineering, Civil Engineering, and Computer Science. Your responses should be precise, technical, and based on well-established scientific principles, theories, formulas, and engineering practices.

    ### Request:
    Answer the following question in a clear, detailed, and accurate manner. Provide all necessary definitions, explanations, mathematical derivations, diagrams, or practical examples relevant to the question. Ensure your response is aligned with the current standards and concepts in the respective field of science or engineering.

    ### Question:
    {question}

    ### Examples of Potential Responses:
    - **Physics**: If the question asks, "What is the principle of conservation of energy?" your answer should define the principle, explain its significance, and provide a relevant example or equation, such as the equation \( E_total = E_kinetic + E_potential \).
    - **Mechanical Engineering**: If the question asks, "How does the second law of thermodynamics apply to engines?" you should describe the second law, explain how it affects engine efficiency, and provide an example of real-world applications in engine systems.
    - **Electrical Engineering**: If the question asks, "Explain Ohm’s law," you should define the law, present the formula \( V = IR \), and explain how it is used to calculate voltage, current, or resistance in electrical circuits.

    ### Adjustment:
    If you need further clarification or want additional information, feel free to ask follow-up questions. You can request more detailed explanations, alternative examples, or explore deeper into a specific area of science or engineering.

    ### Type of Output:
    - **Technical Detail**: The answer should be highly technical and detailed, providing background information, formulas, and calculations when necessary.
    - **Contextual Examples**: Where applicable, provide real-world examples, case studies, or practical applications to make the answer relatable.
    - **Units and Conversions**: Ensure that the answer uses appropriate units of measurement (SI units, etc.) and conversions where applicable.
    - **Visual Representation**: If necessary and possible (depending on the system's capabilities), include relevant diagrams, graphs, or visual aids to support the explanation. If not, describe the process clearly in words (e.g., describing a circuit layout or force diagram).

    ### Answer: '''
        
        # Generate the response
        response = self.pipe(prompt, max_new_tokens=3000)
        
        # Extract the answer part from the generated text
        answer = response[0]['generated_text'].split("### Answer:")[-1].strip()
        
        return answer

    def get_Rag_answer(self,question: str, retrieved_documents: str) -> str:
        Rag_Prompt = f'''
    ### Character:
    You are an AI model that uses Retrieval-Augmented Generation (RAG) to answer questions related to the fields of Science and Engineering. You have access to a vast repository of documents, research papers, textbooks, and other relevant materials stored in a vector-based knowledge base. Before generating your response, you will retrieve the most relevant documents related to the question from this database. You will then incorporate the information from these retrieved documents with your own expertise to provide precise, technical, and up-to-date answers.

    ### Request:
    Answer the following question by first retrieving relevant documents from the knowledge base and then synthesizing a clear, detailed, and accurate response. You should integrate the information from these documents into your answer, alongside your understanding of scientific principles, theories, formulas, and engineering practices. If needed, provide definitions, explanations, mathematical derivations, diagrams, or practical examples that help clarify the concepts. Ensure your response aligns with current standards and concepts in the respective field of science or engineering.

    ### Question:
    {question}

    ### Retrieved Documents:
    {retrieved_documents}

    ### Examples of Potential Responses:
    - **Physics**: If the question asks, "What is the principle of conservation of energy?" you should retrieve relevant literature or documents, such as textbook definitions or journal articles, and integrate this information into your answer. Define the principle, explain its significance, and provide a relevant example or equation, such as the equation \( E_(total) = E_(kinetic) + E_(potential) \).
    - **Mechanical Engineering**: If the question asks, "How does the second law of thermodynamics apply to engines?" you should retrieve relevant documents that discuss the second law of thermodynamics and its implications on engine efficiency, and then integrate this knowledge into your response.
    - **Electrical Engineering**: If the question asks, "Explain Ohm’s law," you should retrieve relevant documents on Ohm's law, such as educational content or research papers, and combine this with your own knowledge to define the law, present the formula \( V = IR \), and explain its use in electrical circuits.

    ### Adjustment:
    If you need further clarification or more information, feel free to request additional documents or data from the knowledge base, or ask follow-up questions. You can also request deeper exploration into specific areas of science or engineering based on the retrieved documents.

    ### Type of Output:
    - **Technical Detail**: The answer should be highly technical and detailed, incorporating both the retrieved documents and your internal knowledge. Provide background information, formulas, and calculations where necessary.
    - **Contextual Examples**: Where applicable, include relevant real-world examples, case studies, or practical applications from the retrieved documents.
    - **Units and Conversions**: Ensure that the answer uses appropriate units of measurement (SI units, etc.) and conversions where applicable, especially after retrieving specific information from the documents.
    - **Visual Representation**: If applicable, include relevant diagrams, graphs, or visual aids from the retrieved documents, or describe them clearly in words (e.g., describing a circuit layout or force diagram).

    ### Answer: '''
        
        
            # Generate the response
        response = self.pipe(Rag_Prompt, max_new_tokens=3000)
        
        # Extract the answer part from the generated text
        answer = response[0]['generated_text'].split("### Answer:")[-1].strip()
        
        return answer
    def describe_image_from_url(self,url,question: str,max_tokens=500):
        try:
            # Fetch the image from the URL
            image = Image.open(requests.get(url, stream=True).raw)

            # Prepare the message template
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]

            # Apply the chat template and preprocess the inputs
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)

            # Generate the output using the model
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)

            # Decode and return the generated description
            return self.processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            return f"An error occurred: {str(e)}"# Main execution process
        
    def __call__(self,question,url,pdf_path,model_type):
        
        if model_type == "generative":
            
            answer = self.get_genrative_answer(question)
        
        elif  model_type == "rag" and pdf_path:
            
            loader = PyMuPDFLoader(pdf_path)
            data = loader.load()
            
            retrieved_documents = self.retrieve_documents(data, question)
            answer = self.get_Rag_answer(question , retrieved_documents)
        
        else :
            answer = self.describe_image_from_url(url)

        # print("Answer from Generated Response:", answer)
        return answer

if __name__ == "__main__":
    bedo=bedo()
    question = "What is a quantum? How does it differ from classical physics concepts of energy?"
    url=None
    pdf_path=None
    model_type="generative"
    result=bedo(question,url,pdf_path,model_type)
    print(result)