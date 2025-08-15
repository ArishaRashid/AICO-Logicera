# AICO Web Summarization Agent

A sophisticated AI agent that summarizes web content using advanced chain summarization techniques, built with LangChain and powered by OpenAI models. The agent can both summarize web pages and engage in conversational chat with memory of previous interactions.

## 🎯 Approach & Innovation

### **Intelligent Chain Summarization**
The system employs a **creative multi-technique approach** that intelligently selects and combines different summarization strategies based on content characteristics. Rather than using a single method, it dynamically chooses from 7 specialized techniques:

- **Adaptive Selection**: Automatically determines the best approach based on content length and complexity
- **Technique Fusion**: Combines multiple methods for optimal results
- **Context-Aware Processing**: Adjusts strategies based on content type and structure

### **Prompt Engineering Excellence**
Our prompts are **carefully crafted and continuously refined** to achieve:

- **Anti-Hallucination**: Structured output formats that prevent AI from making up information
- **Quality Consistency**: Prompts that maintain high standards across different content types
- **Structured Output**: Clear separation between main topic, key points, and summary
- **Context Preservation**: Maintains important details while eliminating noise

### **Memory-Enhanced Conversations**
The chat system goes beyond simple Q&A by implementing **conversational memory** that:
- Remembers the last 3 interactions for context continuity
- Provides intelligent follow-up responses
- Combines web content understanding with conversation history

This approach ensures that each interaction builds upon previous ones, creating a more natural and helpful user experience.

## 🚀 Features

### Part 1: LangChain Agent + Web Integration
- **Web Content Reading**: Uses `WebScraper` to fetch and read webpage content
- **Smart Summarization**: Automatically summarizes web content using advanced AI techniques
- **Polite Responses**: Provides helpful, concise summaries with follow-up question support

### Part 2: Conversation Memory & Chat
- **Memory Management**: Remembers the last 3 messages using `ConversationBufferWindowMemory`
- **Context Awareness**: Maintains conversation context for better follow-up responses
- **Session Persistence**: Keeps track of user interactions within each session
- **Chat Endpoint**: Interactive chat with the AI agent about web content or general topics

### Part 3: API Endpoint
- **RESTful API**: FastAPI-based endpoints for web summarization and chat
- **JSON Responses**: Structured responses with summary and main topic
- **Error Handling**: Comprehensive error handling for invalid inputs and edge cases
- **Async Support**: Background task processing for long-running operations

### Part 4: Advanced Summary Quality
- **Multiple Techniques**: 7 different summarization approaches
- **Adaptive Selection**: Automatically chooses the best technique based on content length
- **Chain Summarization**: Implements map-reduce, recursive, and hierarchical approaches
- **Quality Optimization**: Tuned prompts for better summary accuracy and relevance

## 🏗️ Architecture

The system uses a modular architecture with the following components:

```
AICO Agent/
├── aico_agent/
│   ├── agent_orchestrator.py  # Main agent implementation and orchestration
│   ├── chain_summarizer.py    # Advanced summarization techniques
│   └── web_scraper.py        # Web content extraction and scraping
├── app/
│   └── main.py               # FastAPI application
├── config.py                  # Configuration settings
└── requirements.txt           # Dependencies
```

### Component Responsibilities

- **`agent_orchestrator.py`**: Main orchestrator that handles chat, summarization, and coordinates between components
- **`chain_summarizer.py`**: Implements advanced summarization techniques (stuff, map-reduce, recursive, hierarchical, etc.)
- **`web_scraper.py`**: Handles web page fetching and content extraction using readability and BeautifulSoup
- **`main.py`**: FastAPI server with endpoints for summarization and chat

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection for web content fetching

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AICO
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
# Required: OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom configurations
LOG_LEVEL=INFO
API_PORT=8000
```

### 4. Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

**Note**: OpenAI API usage incurs costs based on token consumption. Monitor your usage to manage costs effectively.

## 🚀 Usage

### Starting the API Server
```bash
cd app
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints


Returns available summarization techniques and descriptions.

#### 1. Summarize Webpage
```bash
POST /summarize
```

**Request Body:**
```json
{
  "url": "https://example.com",
  "technique": "adaptive"
}
```

**Response:**
```json
{
  "summary": "The website discusses the impact of AI on modern education.",
  "main_topic": "AI in Education",
  "technique_used": "adaptive",
  "content_length": 5000,
  "success": true,
  "message": "Webpage summarized successfully using OpenAI models"
}
```

#### 4. Chat with Agent
```bash
POST /chat
```

**Request Body:**
```json
{
  "message": "What's this page about?",
  "url": "https://example.com"
}
```

**Note**: The `url` field is optional. If provided, the agent will summarize the webpage first and then answer your question based on that content. If no URL is provided, the agent will engage in general conversation.

**Response:**
```json
{
  "response": "This page discusses the impact of AI on modern education...",
  "chat_history": [
    {"role": "user", "content": "What's this page about?"},
    {"role": "assistant", "content": "This page discusses the impact of AI..."}
  ],
  "success": true,
  "message": "Chat response generated successfully"
}
```

#### 5. Async Summarization
```bash
POST /summarize/async
```

For long-running summarization tasks.

### Available Summarization Techniques

1. **`adaptive`** (default): Automatically chooses the best approach
2. **`stuff`**: Single-pass summarization for short texts
3. **`map_reduce`**: Advanced map-reduce with recursive collapsing
4. **`recursive`**: Recursively breaks down long texts
5. **`hierarchical`**: Multi-level summarization with increasing detail
6. **`chain_of_thought`**: Step-by-step reasoning approach
7. **`tree_of_thoughts`**: Explores multiple approaches before choosing

## 🔍 Example Usage

### Using the API
```python
import requests

# Summarize a webpage
response = requests.post("http://localhost:8000/summarize", json={
    "url": "https://example.com",
    "technique": "hierarchical"
})

if response.status_code == 200:
    result = response.json()
    print(f"Summary: {result['summary']}")
    print(f"Main Topic: {result['main_topic']}")
    print(f"Technique Used: {result['technique_used']}")

# Chat with the agent about a webpage
chat_response = requests.post("http://localhost:8000/chat", json={
    "message": "What are the main benefits mentioned?",
    "url": "https://example.com"
})

if chat_response.status_code == 200:
    chat_result = chat_response.json()
    print(f"Agent Response: {chat_result['response']}")
    print(f"Chat History: {chat_result['chat_history']}")

# General chat without webpage context
general_chat = requests.post("http://localhost:8000/chat", json={
    "message": "Hello, how are you?"
})

if general_chat.status_code == 200:
    result = general_chat.json()
    print(f"Agent Response: {result['response']}")
```

### Using the Agent Directly
```python
from aico_agent.agent_orchestrator import build_web_summarization_agent, chat_with_agent_simple

# Build the agent
llm, memory = build_web_summarization_agent()

# Chat with the agent
response = chat_with_agent_simple(
    message="What's this page about?",
    url="https://example.com",
    llm=llm,
    memory=memory
)

print(response)
```

## 🧪 Testing

### Test the API
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test summarization
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'

# Test chat with webpage context
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is this page about?", "url": "https://example.com"}'

# Test general chat
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}'
```

### Test Different Techniques
```bash
# Test hierarchical summarization
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com", "technique": "hierarchical"}'
```

## 📊 Configuration

The system is highly configurable through `config.py`:

- **Models**: Choose different OpenAI models for different tasks
- **Chunking**: Configure text splitting parameters
- **Prompts**: Customize summarization prompts
- **Logging**: Adjust logging levels and formats
- **API**: Configure timeouts and model parameters

### OpenAI Model Configuration
```python
MODELS = {
    "agent": "gpt-4o-mini",      # Fast and cost-effective for agent tasks
    "summarizer": "gpt-4o",      # High-quality summarization model
    "chunk_summarizer": "gpt-4o", # Same as summarizer for consistency
    "final_summarizer": "gpt-4o"  # Same as summarizer for consistency
}
```

## 🔒 Security Considerations

- **API Key Protection**: Never commit your `.env` file to version control
- **Input Validation**: All URLs are validated before processing
- **Rate Limiting**: Consider implementing rate limiting for production use
- **CORS**: Configure CORS appropriately for your deployment environment
- **Cost Management**: Monitor OpenAI API usage to manage costs

## 💰 Cost Considerations

OpenAI API usage incurs costs based on:
- **Input tokens**: Text sent to the model
- **Output tokens**: Text generated by the model
- **Model selection**: Different models have different pricing

**Cost Optimization Tips:**
- Use `gpt-4o-mini` for agent tasks (cheaper)
- Use `gpt-4o` for high-quality summarization
- Monitor token usage in your OpenAI dashboard
- Set appropriate `max_tokens` limits

## 🚀 Deployment

### Local Development
```bash
python app/main.py
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Using Docker (create Dockerfile)
docker build -t aico-summarizer .
docker run -p 8000:8000 aico-summarizer
```

## 📝 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set in your `.env` file
   - Verify the API key has proper permissions
   - Check your OpenAI account for any restrictions

2. **Web Content Fetching Issues**
   - Check internet connectivity
   - Verify the URL is accessible
   - Some websites may block automated access

3. **Summarization Quality Issues**
   - Try different summarization techniques
   - Check content length and complexity
   - Review logs for specific error messages

4. **Cost Management Issues**
   - Monitor token usage in OpenAI dashboard
   - Set appropriate `max_tokens` limits
   - Use cheaper models for less critical tasks


## 🙏 Acknowledgments

- **LangChain**: For the excellent framework and tools
- **OpenAI**: For providing powerful AI models and API
- **FastAPI**: For the modern, fast web framework
- **AICO**: For the project requirements and guidance
- **BeautifulSoup & Readability**: For web content extraction capabilities
- **LangGraph**: For advanced chain summarization workflows

---

**Happy Summarizing and Chatting! 🎉**
