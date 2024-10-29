import os
import gradio as gr
from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
import asyncio
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.staticfiles import StaticFiles
from torch.cuda import is_available as if_gpu
from math import exp, log, log10, sqrt, fabs
import torch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Add your API key here if not added already 
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'abcd123'
# os.system("pip install -r requirements.txt")

# Hyper Parameters. Edit these to change Models & Directories

model_llm = "google/flan-t5-large"
# model_llm = "sarvamai/sarvam-2b-v0.5"
model_classifier = "facebook/bart-large-mnli"
model_embeddings = 'sentence-transformers/all-mpnet-base-v2'
vectorDB_dir = 'ncert.index/'
source_directory = 'PDFs/'
labels = [
    "Greeting or identity question", 
    "Question about sound or waves", 
    "Physics or chemistry question (not sound/waves related)", 
    "Mathematical calculation"
]

classifier_threshold = 0.5

max_new_tokens = 512                # Default for Flan T5 is None

# Beam Search
repetition_penalty = 1.0            # Default for Flan T5 is 1.0
length_penalty = 2.0                # Default for Flan T5 is 1.0
num_beams = 3                       # Default for Flan T5 is 1.0

# Sampling Search
temperature = 0.7                   # Default for Flan T5 is 1.0

alpha_hybridsearch = 0.5
k = 5

chunk_size = 1000
chunk_overlap = 200

template_replies = [
    'Hey, This is Rosho! I can help you with any doubts from the NCERT chapter Sound. ',
    'I am only equipped to do simple math calculations. Please provide Python compatible mathematical expressions. ',
    'Sorry, as of now I am only trained on the Chapter Sound from Physics NCERT Class 10 Textbook. ',
    'Sorry, I do not understand your question / Don\'t know that topic. Can you clarify a little more?'
]

# Helper Functions. Edit these to improve Prompts or Functionalities

def format_prompt(results, query):
    user_prompt = results[0].page_content + ' \n ' + query
    user_prompt = user_prompt + ' Explain in detail.'
    return user_prompt

def calculator(query):
    query = str(query)
    query = query.strip()
    query = query.strip("?")
    query = query.replace("^", "**")
    return str(eval(query))

# Initializing

app = FastAPI()

device = 0 if torch.cuda.is_available() else -1
llm = pipeline(model = model_llm, device = device)
classifier = pipeline("zero-shot-classification", model=model_classifier, device=device)
embeddings = HuggingFaceEmbeddings(model_name = model_embeddings)

os.system('cls')

def create_vectorstore(directory, embeddings):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pdf = loader.load()
            documents.extend(pdf)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# VectorDB Loading

vectorstore = None
faiss_retriever = None
def initialize_vectorstore(k=5):
    global vectorstore, faiss_retriever
    try:
        vectorstore = FAISS.load_local(vectorDB_dir, embeddings, allow_dangerous_deserialization=True)
        print("Vector store initialized successfully")
    except Exception as e:
        print(f"{e}: Creating VectorStore from {source_directory}")
        vectorstore = create_vectorstore(source_directory, embeddings)
        vectorstore.save_local(vectorDB_dir)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k":k})

initialize_vectorstore(k)

def labelling_fun(query, threshold): 
    label_dict = classifier(query, labels, multi_label=True)
    print(label_dict)
    best_label, max_score = None, 0
    for label, score in zip(label_dict['labels'], label_dict['scores']):
        if score >= threshold:
            if score > max_score:
                max_score = score 
                best_label = label
    print(best_label)
    return best_label


def hybrid_search(query, vectorstore, faiss_retriever, k=5, alpha=0.5):
    # Initializing BM25 only on the top-k FAISS Results
    faiss_res = vectorstore.similarity_search_with_score(query, 2*k)
    faiss_docs = [res[0] for res in faiss_res]
    bm25_retreiver = BM25Retriever.from_documents(faiss_docs)
    bm25_retreiver.k = k
    # Initializing a Hybrid Retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers = [faiss_retriever, bm25_retreiver],
        weights = [alpha, 1-alpha]
    )
    return hybrid_retriever.invoke(query)

# Most happening function

@app.post("/api_calling")
async def api_calling(query: str):
    try:
        if vectorstore is None:
            return {"error": "Vector store not initialized"}

        label = labelling_fun(query, classifier_threshold)

        if labels[1] == label:
            results = hybrid_search(query, vectorstore, faiss_retriever, k, alpha_hybridsearch)
            # print(results)
            user_prompt = format_prompt(results, query)
            print(user_prompt)

            # Beam Search
            response = llm(user_prompt,
                repetition_penalty=repetition_penalty,
                max_new_tokens = max_new_tokens,
                num_beams = 3,
                length_penalty = length_penalty
            )[0]['generated_text']

            # Sampling Search
            # response = llm(user_prompt,
            #     repetition_penalty=repetition_penalty,
            #     max_new_tokens = max_new_tokens,
            #     temperature=temperature,
            #     do_sample=True
            # )[0]['generated_text']

        elif labels[0]  == label:
            response = template_replies[0]
        elif labels[3]  == label:
            try:
                response = calculator(query)
            except Exception as e:
                print(e)
                response = template_replies[1]
        elif labels[2]  == label:
            response = template_replies[2]
        else:
            response = template_replies[3]
        return response
    except Exception as e:
        return str(e)

async def fetch_results(message_textbox, history):
    history = history or []
    response = await api_calling(message_textbox)
    output = response if response else "No Answer Found"
    history.append((message_textbox, output))
    return history, history, gr.Textbox(value="", placeholder="Next Question?")

block = gr.Blocks(theme=gr.themes.Monochrome(), 
    title="Chat with NCERT!",
    css="footer {visibility: hidden}")

with block:
    gr.Markdown("""
        <html>
            <head>
                <title>Chat with NCERT!</title>
                <link rel = "icon" href="favicon.ico" type="image/x-icon"/>
            </head>
        </html>

        <h1><center>Chat with your NCERT PDF!</center></h1>
    """)
    chatbot_UI = gr.Chatbot(height = 620)
    message_textbox = gr.Textbox(placeholder="What is a wave?")
    history = gr.State()
    submit_button = gr.Button("SEND")
    submit_button.click(fetch_results,
                 inputs=[message_textbox, history],
                 outputs=[chatbot_UI, history, message_textbox])
    message_textbox.submit(fetch_results,
                 inputs=[message_textbox, history],
                 outputs=[chatbot_UI, history, message_textbox])

app = gr.mount_gradio_app(app, block, path="/", favicon_path="favicon.ico")

uvicorn.run(app, host="127.0.0.1", port=8000)
