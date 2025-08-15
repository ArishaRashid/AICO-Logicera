import logging
from typing import Dict, Any
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

# Import configuration
from config import MODELS, AGENT

from aico_agent.web_scraper import WebBrowserTool
from aico_agent.chain_summarizer import build_advanced_summarizer, get_available_techniques

logger = logging.getLogger(__name__)

def build_web_summarization_agent():
    """Build a simple web summarization agent that doesn't get stuck in loops."""
    logger.info("🔧 Building simple web summarization agent...")
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=MODELS["agent"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=1000
    )
    
    # Create conversation memory (last 3 messages as required)
    memory = ConversationBufferWindowMemory(
        k=AGENT["memory_window"],
        memory_key="chat_history",
        return_messages=AGENT["return_messages"],
    )
    
    logger.info("✅ Simple web summarization agent built successfully")
    return llm, memory

def chat_with_agent_simple(message: str, url: str, llm, memory) -> Dict[str, Any]:
    """Simple chat function that directly uses the tool and LLM."""
    try:
        # Get chat history first
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        if url:
            # Use the same summarization approach as the summarize endpoint
            logger.info(f"🔍 Using summarization for chat about URL: {url}")
            
            # Get web content and summarize it
            content = get_web_content(url)
            if not content:
                return {
                    "response": "Sorry, I couldn't read that web page. Please check the URL and try again.",
                    "success": False
                }
            
            # Build the advanced summarizer and get summary
            advanced_summarizer = build_advanced_summarizer()
            summary_result = advanced_summarizer.invoke({
                "text": content,
                "technique": "adaptive"
            })
            
            # Parse the structured output
            parsed_result = parse_structured_summary(summary_result)
            
            # Create a focused prompt for the LLM with the summary
            prompt = f"""You are a helpful AI assistant. A user asked: "{message}"

Here's a summary of the web page content:
Main Topic: {parsed_result['main_topic']}

Summary: {parsed_result['summary']}

Please provide a clear, helpful response to the user's question based on this summary. Be concise and accurate.

Response:"""
        else:
            # General chat without web content
            prompt = f"""You are a helpful AI assistant. A user asked: "{message}"

Please provide a clear, helpful response to the user's question. Be concise and accurate.

Response:"""
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Save to memory
        memory.save_context(
            {"input": message},
            {"output": response.content}
        )
        
        # Get updated chat history
        updated_history = memory.load_memory_variables({})["chat_history"]
        
        return {
            "response": response.content,
            "chat_history": [{"role": "user", "content": msg.content} if hasattr(msg, 'content') else {"role": "user", "content": str(msg)} for msg in updated_history],
            "success": True,
            "message": "Chat response generated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in simple chat: {e}")
        return {
            "response": f"Sorry, I encountered an error: {str(e)}",
            "chat_history": [],
            "success": False,
            "message": "Error occurred"
        }

# =============================================================================
# ADVANCED SUMMARIZATION FUNCTIONS
# =============================================================================

def get_summarization_techniques() -> Dict[str, str]:
    """Get available summarization techniques."""
    return {
        "stuff": "Simple document stuffing",
        "map_reduce": "Map-reduce with recursive collapsing",
        "recursive": "Recursive summarization",
        "hierarchical": "Hierarchical summarization",
        "chain_of_thought": "Chain of thought reasoning",
        "tree_of_thoughts": "Tree of thoughts exploration",
        "adaptive": "Adaptive technique selection"
    }

def summarize_web_content(url: str, technique: str = "adaptive") -> Dict[str, str]:
    """Summarize web content using the specified technique."""
    logger.info(f"🔍 Starting web content summarization for: {url}")
    
    try:
        # Get web content
        content = get_web_content(url)
        if not content:
            raise Exception("Failed to retrieve web content")
        
        logger.info(f"📄 Retrieved {len(content)} characters from web page")
        
        # Build the advanced summarizer
        advanced_summarizer = build_advanced_summarizer()
        
        # Get summary using the specified technique
        summary_result = advanced_summarizer.invoke({
            "text": content,
            "technique": technique
        })
        
        # Parse the structured output to extract main_topic and summary
        parsed_result = parse_structured_summary(summary_result)
        
        return {
            "summary": parsed_result["summary"],
            "main_topic": parsed_result["main_topic"],
            "technique_used": technique,
            "content_length": len(content)
        }
        
    except Exception as e:
        logger.error(f"❌ Error summarizing web content: {e}")
        raise Exception(f"Failed to summarize web content: {e}")

def get_web_content(url: str) -> str:
    """Get web content using WebBrowserTool."""
    try:
        web_tool = WebBrowserTool()
        content = web_tool._run(url, None)
        
        if not content or len(content.strip()) < 100:
            raise ValueError("Web page content is too short or empty")
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"❌ Error getting web content: {e}")
        raise e

def parse_structured_summary(summary_text: str) -> Dict[str, str]:
    """Parse the structured summary output to extract main_topic and summary."""
    try:
        # Initialize default values
        main_topic = ""
        summary = ""
        
        # Split the text into lines
        lines = summary_text.strip().split('\n')
        
        # Parse the structured format
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.upper().startswith("MAIN TOPIC:"):
                current_section = "main_topic"
                main_topic = line.replace("MAIN TOPIC:", "").strip()
            elif line.upper().startswith("KEY POINTS:"):
                current_section = "key_points"
            elif line.upper().startswith("SUMMARY:"):
                current_section = "summary"
            elif current_section == "summary" and line:
                # Collect summary content
                if summary:
                    summary += " " + line
                else:
                    summary = line
            elif current_section == "main_topic" and line and not main_topic:
                # If main topic is empty, use the first line after "MAIN TOPIC:"
                main_topic = line
        
        # If we couldn't parse structured format, try to extract intelligently
        if not main_topic or not summary:
            # Split the text and use first part as topic, rest as summary
            parts = summary_text.strip().split('\n\n', 1)
            if len(parts) >= 2:
                main_topic = parts[0].strip()
                summary = parts[1].strip()
            else:
                # Fallback: use first sentence as topic, rest as summary
                sentences = summary_text.strip().split('. ')
                if len(sentences) >= 2:
                    main_topic = sentences[0].strip() + "."
                    summary = '. '.join(sentences[1:]).strip()
                else:
                    # Last resort: use first 100 chars as topic, rest as summary
                    main_topic = summary_text[:100].strip()
                    summary = summary_text[100:].strip()
        
        # Clean up and ensure we have content
        if not main_topic:
            main_topic = "Web page content"
        if not summary:
            summary = summary_text.strip()
        
        return {
            "main_topic": main_topic,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error parsing structured summary: {e}")
        # Return fallback values
        return {
            "main_topic": "Web page content",
            "summary": summary_text.strip()
        }
        