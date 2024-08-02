from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from src.data_converter import text_chunks
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def data_ingestion(status):

    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name = "medical",
        api_endpoint = ASTRA_DB_API_ENDPOINT,
        token = ASTRA_DB_APPLICATION_TOKEN,
        namespace = ASTRA_DB_KEYSPACE)
    
    storage = status

    if storage == None:
        docs = text_chunks()
        insert_ids = vstore.add_documents(docs)
    
    else:
        return vstore
    return vstore, insert_ids

if __name__ == "__main__":

    # vstore, insert_ids = data_ingestion(None)
    # print(f"\n Inserted {len(insert_ids)} documents.")
    # results = vstore.similarity_search("What are heart acttack?")
    # for res in results:
    #     print(f"\n {res.page_content} [{res.metadata}]")
    pass