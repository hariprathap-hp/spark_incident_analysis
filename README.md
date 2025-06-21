# Ice Breaker - LinkedIn Profile Analyzer

A FastAPI application that uses AI agents to find LinkedIn profiles and generate personalized summaries and interesting facts about people. Perfect for networking, sales outreach, or getting to know someone before a meeting!

## 🚀 Features

- **Smart LinkedIn Search**: Uses AI agents to automatically find LinkedIn profiles by name
- **Profile Analysis**: Generates concise summaries and interesting facts from LinkedIn data
- **Web Interface**: Clean, easy-to-use web interface
- **AI-Powered**: Leverages OpenAI GPT models and LangChain agents
- **Mock Mode**: Development-friendly mode that uses sample data to avoid API costs

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- Tavily API Key (for web search)
- Optional: LangSmith API Key (for debugging/monitoring)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hariprathap-hp/langchain_lin/tree/LinkedIn
   cd langchain_lin & git checkout LinkedIn
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env  # If example exists, or create manually
   ```
   
   Add your API keys to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LANGCHAIN_API_KEY=your_langsmith_api_key_here (optional)
   LANGSMITH_TRACING=true
   LANGCHAIN_PROJECT=Ice Breaker
   ```

## 🔑 Getting API Keys

### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Add billing information (pay-per-use)

### Tavily API Key (Required)
1. Visit [Tavily](https://tavily.com/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Tavily provides web search capabilities for finding LinkedIn profiles

### LangSmith API Key (Optional)
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account
3. Generate API key for debugging and monitoring (optional but recommended for development)

## 🚀 Running the Application

1. **Start the FastAPI server**
   ```bash
   python app.py
   ```
   
   Or alternatively:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the application**
   - Open your browser and go to: `http://localhost:8000`
   - Enter a person's name in the form
   - Click "Generate Ice Breaker" to get their profile summary

3. **API Documentation**
   - FastAPI automatically generates API docs at: `http://localhost:8000/docs`

## 📁 Project Structure

```
hari_ice_breaker/
├── app.py                          # FastAPI main application
├── ice_breaker.py                  # Core ice breaker logic
├── output_parsers.py               # Pydantic models for structured output
├── agents/
│   ├── __init__.py
│   └── linkedin_lookup_agent.py    # AI agent for LinkedIn profile search
├── tools/
│   ├── __init__.py
│   └── tools.py                    # Search tools using Tavily
├── third_parties/
│   └── linkedin.py                 # LinkedIn profile scraping utilities
├── templates/
│   └── index.html                  # Web interface template
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
└── README.md                      # This file
```

## 🔒 Security Notes

- **Never commit your `.env` file** - it contains sensitive API keys
- The `.env` file is already in `.gitignore` to prevent accidental commits
- API keys in this project are used for:
  - OpenAI: AI model access (charges apply per token)
  - Tavily: Web search functionality
  - LangSmith: Optional debugging/monitoring
- Mock mode is enabled by default to prevent unnecessary API charges during development

## 🧪 Development Mode

The application runs in **mock mode** by default, which:
- Uses sample LinkedIn data from a GitHub Gist instead of real scraping
- Prevents API charges for LinkedIn data scraping
- Still uses OpenAI and Tavily APIs for search and analysis

To disable mock mode, edit `ice_breaker.py` and change:
```python
linkedin_data = scrape_linkedin_profile(
    linkedin_profile_url=linkedin_username, mock=True  # Change to False
)
```

## 📝 Usage Examples

1. **Basic Usage**: Enter "Eden Marco" to see the sample profile analysis
2. **Real Profiles**: The AI agent will search for and analyze any public LinkedIn profile
3. **API Access**: Send POST requests to `/process` with form data containing the name

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Run from project root directory
   python -m agents.linkedin_lookup_agent
   ```

2. **Pydantic Version Conflicts**
   ```bash
   pip install pydantic --upgrade
   ```

3. **Missing Templates Directory**
   ```bash
   mkdir templates
   # Make sure index.html exists in templates/
   ```

4. **API Key Issues**
   - Ensure all required API keys are in `.env` file
   - Check API key format and validity
   - Verify billing setup for OpenAI

### Environment Variables Check
```bash
# Test if environment variables are loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OpenAI Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'MISSING')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

- This tool is for educational and professional networking purposes only
- Respect LinkedIn's terms of service and rate limits
- Be mindful of privacy when analyzing public profiles
- API usage charges apply for OpenAI and other services

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review FastAPI logs for error details
3. Ensure all dependencies are installed correctly
4. Verify API keys are valid and have sufficient credits

---

**Happy networking! 🌐**
