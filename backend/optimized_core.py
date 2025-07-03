from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
import tiktoken
import time

load_dotenv()  # ✅ FIXED: Added parentheses

INDEX_NAME = "langchain-doc-index-hf"

# OPTIMIZATION 1: Create reusable components (initialize once)
class DocumentationAssistant:
    def __init__(self):
        """Initialize once, reuse many times"""
        print("🚀 Initializing Documentation Assistant...")
        
        # Initialize embeddings (FREE - local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Connect to Pinecone (FREE - just connection)
        self.docsearch = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=self.embeddings
        )
        
        # Initialize chat model (FREE - just setup)
        self.chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            verbose=False,  # Less verbose for production
            temperature=0
        )
        
        # OPTIMIZATION 2: Download prompt template once (not every query)
        print("📥 Loading prompt template...")
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        
        # Create reusable chains
        self.stuff_documents_chain = create_stuff_documents_chain(
            self.chat, 
            self.retrieval_qa_chat_prompt
        )
        
        # OPTIMIZATION 3: Limit document retrieval for cost control
        self.qa = create_retrieval_chain(
            retriever=self.docsearch.as_retriever(
                search_kwargs={"k": 2}  # Reduced from 4 to 2 docs (less tokens)
            ),
            combine_docs_chain=self.stuff_documents_chain
        )
        
        # Initialize cost tracking
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.total_cost = 0.0
        self.query_count = 0
        
        print("✅ Assistant ready!")
    
    def estimate_query_cost(self, query: str) -> dict:
        """Estimate cost before making the query"""
        
        # Get documents that would be retrieved
        docs = self.docsearch.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Estimate prompt size
        estimated_prompt = f"""Based on the following context documents, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        input_tokens = len(self.encoding.encode(estimated_prompt))
        estimated_output_tokens = 300  # Conservative estimate
        
        input_cost = (input_tokens / 1000) * 0.0005
        output_cost = (estimated_output_tokens / 1000) * 0.0015
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost": total_cost,
            "retrieved_docs": len(docs)
        }
    
    def run_llm(self, query: str, confirm_cost: bool = False) -> dict:
        """Run query with cost tracking"""
        
        start_time = time.time()
        
        # OPTIMIZATION 4: Cost estimation before query
        if confirm_cost:
            cost_info = self.estimate_query_cost(query)
            print(f"💰 Estimated cost: ${cost_info['estimated_cost']:.6f}")
            print(f"📄 Documents to retrieve: {cost_info['retrieved_docs']}")
            
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                return {"cancelled": True}
        
        # Execute query (this is where money is spent)
        try:
            result = self.qa.invoke({"input": query})
            
            # Track actual cost (approximate)
            cost_info = self.estimate_query_cost(query)
            actual_cost = cost_info["estimated_cost"]
            
            self.total_cost += actual_cost
            self.query_count += 1
            
            elapsed_time = time.time() - start_time
            
            # Add cost tracking to result
            result["cost_info"] = {
                "estimated_cost": actual_cost,
                "total_session_cost": self.total_cost,
                "query_number": self.query_count,
                "response_time": elapsed_time
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            if "quota" in str(e).lower():
                print("💳 OpenAI quota exceeded. Check billing at https://platform.openai.com/account/billing")
            return {"error": str(e)}
    
    def get_session_stats(self):
        """Get cost statistics for current session"""
        remaining_budget = 5.0 - self.total_cost
        queries_remaining = int(remaining_budget / (self.total_cost / max(1, self.query_count)))
        
        return {
            "queries_made": self.query_count,
            "total_cost": self.total_cost,
            "average_cost_per_query": self.total_cost / max(1, self.query_count),
            "remaining_budget": remaining_budget,
            "estimated_queries_remaining": queries_remaining
        }

# OPTIMIZATION 5: Singleton pattern - create once, use many times
assistant = None

def get_assistant():
    """Get or create assistant instance"""
    global assistant
    if assistant is None:
        assistant = DocumentationAssistant()
    return assistant

def run_llm(query: str, confirm_cost: bool = False):
    """Optimized function that reuses components"""
    return get_assistant().run_llm(query, confirm_cost)

if __name__ == "__main__":
    # Test the optimized version
    queries = [
        "Can you explain more about SIP REFER Messages?"
    ]
    
    print("Documentation Assistant")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        result = run_llm(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            break
        elif "cancelled" in result:
            print("⏹️  Query cancelled")
            continue
        
        print(f"✅ Answer: {result['answer'][:200]}...")
        
        cost_info = result.get("cost_info", {})
        print(f"💰 Cost: ${cost_info.get('estimated_cost', 0):.6f}")
        print(f"⏱️  Time: {cost_info.get('response_time', 0):.2f}s")
    
    # Session statistics
    stats = get_assistant().get_session_stats()
    print(f"\n📊 Session Statistics:")
    print(f"   Queries made: {stats['queries_made']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")
    print(f"   Avg cost/query: ${stats['average_cost_per_query']:.6f}")
    print(f"   Remaining budget: ${stats['remaining_budget']:.6f}")
    print(f"   Est. queries remaining: {stats['estimated_queries_remaining']}")