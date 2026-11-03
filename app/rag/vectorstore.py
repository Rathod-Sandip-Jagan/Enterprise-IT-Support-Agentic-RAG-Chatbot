
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# Embedding model dimensions
EMBEDDING_DIMENSIONS = {
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model: {name}. "
        "Please configure a supported embedding model."
    )


# --------------------------------
# Hugging Face Embedding Model
# --------------------------------

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        model_name = settings.embedding_model

        if not model_name:
            raise RuntimeError("EMBEDDING_MODEL is not configured")

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print(f"Hugging Face embedding model loaded: {model_name}")

    return _embeddings


# --------------------------------
# Pinecone Index Configuration
# --------------------------------

def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    index_name = settings.pinecone_index_name

    def get_index_names():
        return pc.list_indexes().names()

    # Check whether index exists
    if index_name in get_index_names():

        index_info = pc.describe_index(index_name)

        current_dimension = index_info.dimension

        # Never automatically delete existing index
        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone dimension mismatch! "
                f"Existing index: {current_dimension}, "
                f"Required dimension: {desired_dimension}. "
                "Create a new index with the correct dimension "
                "or manually recreate the existing index."
            )

        print(f"Pinecone index already exists: {index_name}")

    else:

        print(f"Creating Pinecone index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait until index is ready
        while True:
            index_info = pc.describe_index(index_name)

            if index_info.status.ready:
                break

            time.sleep(1)

        print("Pinecone index created successfully!")

    return pc.Index(index_name)


# --------------------------------
# Initialize Vector Store
# --------------------------------

def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

        print("Pinecone vector store initialized successfully!")

    return _vectorstore


# --------------------------------
# Retriever
# --------------------------------

def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# --------------------------------
# Add Documents
# --------------------------------

def add_documents(chunks):

    if not chunks:
        print("No document chunks available.")
        return []

    store = get_vectorstore()

    result = store.add_documents(chunks)

    print(f"Successfully added {len(chunks)} chunks to Pinecone!")

    return result
