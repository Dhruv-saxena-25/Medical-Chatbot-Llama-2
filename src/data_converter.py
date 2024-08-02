from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def text_chunks():
    data_path = "./data"
    loader = DirectoryLoader(data_path,
                    glob= "*.pdf",
                    loader_cls= PyPDFLoader)
    extracted_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap= 20)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

if __name__ == "__main__":
    # chunks= text_chunks()
    # print(chunks)
    # print("length of my chunk:", len(chunks))
    pass

