from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

extracted_docs = load_pdf_file(data= "data/")
minimal_docs = filter_to_minimal_docs(extracted_docs)
texts_chunk = text_split(minimal_docs)
embedding = download_embeddings()


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric= "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding ,
    index_name=index_name
)