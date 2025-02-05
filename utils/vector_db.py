import os
from dotenv import load_dotenv
#from llama_index.vector_stores import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
#from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.node_parser import SimpleNodeParser

#from llama_index import Document
   #from llama_index.schema import Document
from llama_index.core import Document

#from llama_index.readers.file.base import SimpleFileReader
   #from llama_index.readers.file import SimpleFileReader
from llama_index.core import SimpleDirectoryReader


#from llama_index.readers.schema.base import Document as LlamaDocument
from llama_index.core import Document as LlamaDocument


import chromadb
from typing import List

from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.storage import StorageContext


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIRECTORY = 'chroma_data' #"mevzuat_db" # Chroma veritabanının kaydedildiği dizin


class CustomFileLoader(SimpleDirectoryReader):
    def __init__(self, input_dir=None, input_files=None):
        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")
        super().__init__(input_dir=input_dir, input_files=input_files)

    """Load files of different types."""
    def load_data(self, file: str) -> List[Document]:
        """Load data from the file."""
        
        file_ext = os.path.splitext(file)[1].lower()
        
        if file_ext==".pdf":
           try:
                from llama_index.readers.file import PyPDFLoader
                loader = PyPDFLoader(file_path=file)
                documents = loader.load_data()
           except ImportError:
                  print("pypdf import hatası")
        elif file_ext==".txt":
            try:
                loader = SimpleDirectoryReader(input_files=[file])
                documents = loader.load_data()
            except ImportError:
                 print("txt loader import hatası")
        elif file_ext == ".html":
            try:
                from llama_index.readers.file import UnstructuredReader
                loader = UnstructuredReader(file_path=file,encoding="utf-8",file_extractor = {
                            ".html": "html",
                        })
                documents= loader.load_data()
            except ImportError:
                  print("html import hatası")
                
        else:
            try:
                from llama_index.readers.file import UnstructuredReader
                loader=UnstructuredReader(file_path=file,encoding="utf-8")
                documents = loader.load_data()
            except ImportError:
                 print("unstructured import hatası")


        return documents
    
    def load_data(self, *args, **kwargs):
        """Override load_data to gracefully ignore unexpected arguments."""
        kwargs.pop('extra_info', None)  # Remove 'extra_info' if present
        return super().load_data(*args, **kwargs)




def load_documents(data_dir="data/documents"):
    reader = SimpleDirectoryReader(data_dir, file_extractor={
        ".pdf": CustomFileLoader(input_dir=data_dir,),
        ".txt": CustomFileLoader(input_dir=data_dir),
        ".html": CustomFileLoader(input_dir=data_dir),
    })

    documents = reader.load_data()
    #documents = reader.load_data(extra_info="some metadata") 

    return documents

def create_vector_db(documents, persist_directory=PERSIST_DIRECTORY):
   
    #from llama_index.vector_stores.chroma import ChromaVectorStore 

    embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=2048, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)
    

    #db = Chroma.from_documents(texts, hf, collection_name='karpayi_bge_m3_1024', persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, collection_metadata={"hnsw:space": "cosine"})
    #db.persist()
    #db = None
    #print ('learning finished')


    # Chroma istemcisini başlat
    db = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = db.get_or_create_collection("mevzuat_2")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)#, collection_metadata = {"hnsw:space": "cosine"} )
    #vector_store.persist()
    #vector_store=None
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    print ('learning finished')
    return index



def load_vector_db(persist_directory=PERSIST_DIRECTORY):
    # Chroma istemcisini başlat
    db = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = db.get_or_create_collection("mevzuat_2")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,storage_context=storage_context)

    return index

def get_relevant_documents(query, vector_db, k=4):
    query_engine = vector_db.as_query_engine(similarity_top_k=k)
    response = query_engine.query(query)
    return response.source_nodes


def update_vector_db(data_dir="data/documents", persist_directory=PERSIST_DIRECTORY):
    documents = load_documents(data_dir)
    create_vector_db(documents, persist_directory)
    print("Vektör veritabanı güncellendi.")



def get_original_chunk(node,vector_db):
    return ""


if __name__ == "__main__":
    # Örnek kullanım:
    documents = load_documents()
    vectordb = create_vector_db(documents)
    #vectordb=load_vector_db()
    #query = "binaların tehlike sınıflandırması hakkında bilgi verirmisn ?"
    #relevant_docs = get_relevant_documents(query, vectordb)
    #print(relevant_docs)
 
    # binaların tehlike sınıflandırması hakkında bilgi verirmisn ?
