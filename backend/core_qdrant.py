from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain import hub
import time

load_dotenv()

COLLECTION_NAME = "spark-incidents-openai"
QDRANT_URL = "http://localhost:6333"

class SparkIncidentAssistant:
    def __init__(self):
        print("🚀 Initializing Spark Incident Assistant (OpenAI powered)...")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = QdrantClient(url=QDRANT_URL)
        self.docsearch = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings
        )

        self.chat = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.stuff_documents_chain = create_stuff_documents_chain(
            self.chat, self.retrieval_qa_chat_prompt
        )

        self.qa = create_retrieval_chain(
            retriever=self.docsearch.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain=self.stuff_documents_chain
        )

        print("✅ Assistant ready!")

    def run_llm(self, query: str):
        start = time.time()
        result = self.qa.invoke({"input": query})
        result["response_time"] = round(time.time() - start, 2)
        return result

assistant = None

def get_assistant():
    global assistant
    if assistant is None:
        assistant = SparkIncidentAssistant()
    return assistant

def run_llm(query: str):
    return get_assistant().run_llm(query)
