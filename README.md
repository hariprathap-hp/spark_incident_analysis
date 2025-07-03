# 📚 AI Documentation Assistant

An intelligent documentation search and Q&A system built with LangChain, OpenAI, and Pinecone. Ask questions about your technical documentation in natural language and get accurate, contextual answers with source citations.

## 🎯 Features

- **Semantic Document Search** - Find relevant information using natural language queries
- **Cost-Optimized** - Smart retrieval strategies to minimize OpenAI API costs
- **Multi-Format Support** - Process PDF documents from Confluence exports
- **Source Attribution** - Track which documents contain the answers
- **Real-time Cost Tracking** - Monitor API usage and remaining budget
- **Streamlit Web Interface** - User-friendly chat-based interaction
- **Production Ready** - Optimized for performance and scalability

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Documentation   │───▶│   OpenAI GPT    │
│                 │    │    Assistant     │    │   (Answer Gen)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Pinecone DB    │
                       │ (Vector Storage) │
                       └──────────────────┘
                              ▲
                              │
                       ┌──────────────────┐
                       │  HuggingFace     │
                       │  Embeddings      │
                       │     (FREE)       │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone account and API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-documentation-assistant.git
   cd ai-documentation-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # myenv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your API keys
   OPENAI_API_KEY=sk-proj-your-openai-key-here
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-environment
   PINECONE_INDEX_NAME=langchain-doc-index-hf
   ```

5. **Create Pinecone index**
   ```bash
   # Create index with 384 dimensions for HuggingFace embeddings
   python scripts/create_pinecone_index.py
   ```

6. **Ingest your documents**
   ```bash
   # Place PDF files in sky-document-builder/ directory
   python backend/ingestion.py
   ```

7. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 📁 Project Structure

```
ai-documentation-assistant/
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── main.py                  # Streamlit web interface
│
├── backend/                 # Core logic
│   ├── __init__.py
│   ├── core.py             # Basic LLM function
│   ├── optimized_core.py   # Optimized assistant class
│   └── ingestion.py        # Document processing pipeline
│
├── sky-document-builder/    # Your PDF documents
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
│
├── scripts/                # Utility scripts
│   ├── create_pinecone_index.py
│   ├── test_connection.py
│   └── cost_calculator.py
│
└── docs/                   # Additional documentation
    ├── SETUP.md
    ├── API_COSTS.md
    └── TROUBLESHOOTING.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT responses | ✅ Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector storage | ✅ Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., us-east-1-aws) | ✅ Yes |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index | No (default: langchain-doc-index-hf) |

### Pinecone Index Setup

**For HuggingFace Embeddings (Free):**
- **Dimensions**: 384
- **Metric**: cosine
- **Index Name**: langchain-doc-index-hf

**For OpenAI Embeddings (Paid):**
- **Dimensions**: 1536
- **Metric**: cosine
- **Index Name**: langchain-doc-index

## 💰 Cost Optimization

### Current Cost Structure
- **Average cost per query**: ~$0.0007
- **With $5 budget**: ~7,000 queries
- **Daily usage (50 queries)**: ~$0.035/day

### Optimization Features
- **Smart k-selection**: Dynamically choose 2-4 documents based on query complexity
- **HuggingFace embeddings**: Free local embeddings (vs paid OpenAI embeddings)
- **Batch processing**: Efficient document ingestion
- **Cost tracking**: Real-time budget monitoring

### Cost Breakdown
```
Input tokens:  ~1,100 tokens (k=2) | ~1,900 tokens (k=4)
Output tokens: ~300 tokens
Cost per query: $0.0006 (k=2) | $0.0007 (k=4)
```

## 📖 Usage Examples

### Basic Query
```python
# Ask about SIP protocol
"What is SIP INVITE message?"

# Result: Detailed explanation with source citations
```

### Complex Query
```python
# Compare protocols
"Compare SIP vs H.323 protocols"

# Result: Comprehensive comparison using multiple document sources
```

### Procedural Query
```python
# Configuration steps
"How to configure AGW step by step?"

# Result: Step-by-step instructions from documentation
```

## 🛠️ Development

### Running Tests
```bash
# Test document ingestion
python backend/test_ingestion.py

# Test API connections
python scripts/test_connection.py

# Calculate costs
python scripts/cost_calculator.py
```

### Adding New Documents
1. Place PDF files in `sky-document-builder/` directory
2. Run ingestion: `python backend/ingestion.py`
3. Documents are automatically processed and indexed

### Customizing the Assistant
```python
# Modify retrieval parameters in backend/optimized_core.py
class DocumentationAssistant:
    def __init__(self):
        self.k_value = 2  # Number of documents to retrieve
        self.chunk_size = 600  # Text chunk size
        self.chunk_overlap = 50  # Overlap between chunks
```

## 🔍 Troubleshooting

### Common Issues

**1. "Missing API keys" Error**
```bash
# Check .env file exists and contains keys
cat .env
```

**2. "No documents found" Error**
```bash
# Verify Pinecone index exists and has data
python scripts/check_pinecone_status.py
```

**3. "Import errors" in Streamlit**
```bash
# Ensure backend directory is in Python path
export PYTHONPATH="${PYTHONPATH}:./backend"
```

**4. High costs**
- Reduce k value (documents retrieved)
- Use shorter, more specific queries
- Enable cost confirmation prompts

### Performance Issues
- **Slow responses**: Check internet connection to OpenAI/Pinecone
- **High memory usage**: Reduce batch size in ingestion
- **Import errors**: Verify all dependencies installed

## 📊 Monitoring

### Built-in Metrics
- Total queries made
- Total cost incurred
- Average cost per query
- Remaining budget
- Response times

### Cost Alerts
The system automatically tracks costs and provides warnings when approaching budget limits.

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - LLM application framework
- **OpenAI** - GPT models for response generation
- **Pinecone** - Vector database for document storage
- **HuggingFace** - Free embedding models
- **Streamlit** - Web interface framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-documentation-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-documentation-assistant/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/ai-documentation-assistant/wiki)

## 🔮 Roadmap

- [ ] **Multi-language support** - Support for non-English documents
- [ ] **Advanced filters** - Filter by document type, date, author
- [ ] **Conversation memory** - Remember previous questions in chat
- [ ] **Document upload UI** - Upload PDFs through web interface
- [ ] **Analytics dashboard** - Detailed usage and cost analytics
- [ ] **API endpoints** - REST API for integration
- [ ] **Docker deployment** - Containerized deployment option

---
