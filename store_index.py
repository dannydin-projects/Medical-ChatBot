from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import src.helper as helper 
from langchain_pinecone import PineconeVectorStore

print("Loading environment variables...")
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

extracted_data = helper.load_pdf_files("data/")
filtered_data = helper.filter_to_minimal_docs(extracted_data)
chunked_data = helper.chunking_docs(filtered_data)

embeddings = helper.download_embeddings()

pc = Pinecone(api_key = pinecone_api_key)

index_name = 'med-chatbot'

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        dimension=384, # Dimension of the embeddings
        metric='cosine' # Cosine similarity search
    )

index = pc.Index(index_name)

docSearch = PineconeVectorStore.from_documents(
    documents = chunked_data,
    embedding=embeddings,
    index_name=index_name,
    batch_size=5  # Process documents in batches of 10 to stay under Pinecone's 2MB request limit
)