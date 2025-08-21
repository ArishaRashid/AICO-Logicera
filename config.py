"""
Configuration file for AICO Web Summarizer
Contains all prompts, model IDs, and other configuration parameters
"""

# =============================================================================
# OPENAI MODEL CONFIGURATION
# =============================================================================

# Model IDs for different tasks
MODELS = {
    "agent": "gpt-4o-mini",  # Fast and cost-effective for agent tasks
    "summarizer": "gpt-4o",  # High-quality summarization model
    "chunk_summarizer": "gpt-4o",  # Same as summarizer for consistency
    "final_summarizer": "gpt-4o"  # Same as summarizer for consistency
}

# Alternative model configurations (uncomment to use)
# MODELS = {
#     "agent": "gpt-3.5-turbo",  # Faster and more cost-effective
#     "summarizer": "gpt-4o",  # Best quality for summarization
#     "chunk_summarizer": "gpt-4o",
#     "final_summarizer": "gpt-4o"
# }

# Model notes and alternatives
MODEL_NOTES = {
    "gpt-4o": "USE: High-quality summarization, excellent reasoning",
    "gpt-4o-mini": "USE: Fast, cost-effective for agent tasks",
    "gpt-3.5-turbo": "ALTERNATIVE: Good balance of speed and quality",
    "gpt-4-turbo": "ALTERNATIVE: High quality with good speed"
}

# =============================================================================
# OPENAI API CONFIGURATION
# =============================================================================

OPENAI_API = {
    "base_url": "https://api.openai.com/v1",
    "api_version": "2024-01-01",
    "timeout": 60,
    "max_retries": 3,
    "temperature": 0.1,  # Low temperature for consistent summarization
    "max_tokens": 2000,  # Maximum tokens for summaries
    "model_parameters": {
        "gpt-4o": {
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "gpt-4o-mini": {
            "temperature": 0.2,
            "max_tokens": 1500
        },
        "gpt-3.5-turbo": {
            "temperature": 0.1,
            "max_tokens": 1500
        }
    }
}

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNKING = {
    "chunk_size": 8000,  # Increased for OpenAI's larger context windows
    "chunk_overlap": 500,  # Increased overlap for better context
    "chunk_threshold": 8000,  # When to use chunked processing
    "separators": ["\n\n", "\n", ". ", " ", ""]
}

# =============================================================================
# OUTPUT FORMATTING CONFIGURATION
# =============================================================================

OUTPUT_FORMAT = {
    "summary_label": "Summary:",
    "topic_label": "Main Topic:",
    "max_topic_words": 50,
    "max_summary_length": 1000,
    "fallback_topic": "Content Summary"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "debug_log_prefix": "summarization_debug_",
    "simple_log_prefix": "simple_summarization_",
    "log_timestamp_format": "%Y%m%d_%H%M%S"
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API = {
    "timeout": 60,
    "base_url": "https://api.openai.com/v1",
    "headers_key": "Authorization",
    "headers_prefix": "Bearer "
}

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT = {
    "memory_window": 3,  # Remember last 3 messages
    "return_messages": True,
    "verbose": True,
    "handle_parsing_errors": True,
    "max_iterations": 5,  # Reduced to prevent infinite loops
    "max_execution_time": 60,  # 1 minute max execution time
}

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    "openai_api_error": "OpenAI API request error: {error}",
    "openai_general_error": "OpenAI API error: {error}",
    "quota_error": "Due to technical limitations, I'm unable to provide a complete summary at the moment. Please try again later.",
    "model_error": "Due to technical difficulties with the AI model, I'm unable to provide a summary at the moment. Please try again later.",
    "processing_error": "Error processing summary output",
    "extraction_error": "Error in extract_summary_and_topic: {error}"
}

# =============================================================================
# VALIDATION RULES
# =============================================================================

VALIDATION = {
    "min_summary_length": 10,
    "min_topic_length": 2,
    "max_summary_length": 5000,
    "max_topic_length": 300
}

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

DEBUG = {
    "enable_detailed_logging": True,
    "log_chunk_details": True,
    "log_api_payloads": True,
    "log_raw_responses": True,
    "save_debug_files": True
}
