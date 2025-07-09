from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Upload & Load raw PDF(s)

pdfs_directory = 'pdfs/'  # Folder where PDFs are stored

# Function to upload a PDF file
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

# Function to load a PDF and extract documents
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)  # Likely uses a custom or external class
    documents = loader.load()
    return documents

# Specify the path and load the PDF
file_path = 'UHR.pdf'
documents = load_pdf(file_path)

# Output number of extracted documents/pages
#print(len(documents))

# Step 2: Create Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
#print("Chunks count: ", len(text_chunks))


# Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings


# Step 4: Index Documents – Store embeddings in FAISS (vector store)
FAISS_DB_PATH = "vectorstore/db_faiss"

faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)




