from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader, PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
# TOGGLE BETWEEN EMBEDDINGS
USE_OPENAI = False  # Set to True for OpenAI, False for HuggingFace

if USE_OPENAI:
    print("🤖 Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    INDEX_NAME = "langchain-doc-index"  # 1536 dimensions
else:
    print("🤗 Using HuggingFace embeddings (FREE)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    INDEX_NAME = "langchain-doc-index-hf"  # 384 dimensions

print(f"Using index: {INDEX_NAME}")
print(f"Expected dimensions: {1536 if USE_OPENAI else 384}")

def ingest_docs():
    loader = PyPDFDirectoryLoader("sky-document-builder")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # PROCESS ALL DOCUMENTS
    print(f"Going to add ALL {len(documents)} chunks to Pinecone")
    print("This is FREE with HuggingFace - no limits!")

    try:
        # Process in batches to be safe (optional but recommended)
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        vectorstore = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            if vectorstore is None:
                # First batch - create vectorstore
                vectorstore = PineconeVectorStore.from_documents(
                    batch, 
                    embeddings, 
                    index_name=INDEX_NAME
                )
            else:
                # Add to existing vectorstore
                vectorstore.add_documents(batch)
            
            print(f"✅ Batch {batch_num} completed")
        
        print(f"🎉 Successfully processed ALL {len(documents)} chunks!")
        print("****Loading to vectorstore done ***")

        # Test search
        print("\n🔍 Testing search on full dataset...")
        results = vectorstore.similarity_search("configuration", k=3)
        print(f"Search results for 'configuration':")
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"{i}. Source: {source}")
            print(f"   Content: {doc.page_content[:150]}...")
            print()

    except Exception as e:
        print(f"❌ Error: {str(e)}")     

if __name__ == "__main__":
    ingest_docs()