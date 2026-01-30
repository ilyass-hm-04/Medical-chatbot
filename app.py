from flask import Flask, render_template, request
from src.helper import download_embeddings

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from src.prompt import *
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Must set OPENAI_API_KEY to your OpenRouter key
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embedding = download_embeddings()

# Pinecone vector store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Chat model
chat = ChatOpenAI(
    model_name="qwen/qwen3-next-80b-a3b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1", 
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# RAG chain
question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Test the chat model
msg = [HumanMessage(content="Say hi")]
resp = chat(msg)
print("Test response:", resp.content)

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat_route():
    msg = request.form["msg"]
    print("User input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
