from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub

load_dotenv()

INDEX_NAME = "langchain-doc-index-hf"

def run_llm(query: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    chat = ChatOpenAI(
    model="gpt-3.5-turbo",  # Cheaper than GPT-4
    verbose=True, 
    temperature=0
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query="What is SIP protocol?")
    print(res["answer"])
