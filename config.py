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
# PROMPT TEMPLATES
# =============================================================================

# Simple Summarizer Prompts - Strong anti-hallucination guardrails
SIMPLE_SUMMARIZER_PROMPTS = {
    "first_pass": """Extract ONLY factual information from this text. DO NOT invent, speculate, or reinterpret metaphors as literal events.

CRITICAL RULES:
- Extract ONLY facts explicitly stated in the text
- DO NOT add new entities, actions, or events not in the source
- DO NOT reinterpret metaphors, symbols, or figurative language as literal facts
- If you encounter ambiguous phrases, quote them exactly without interpretation
- Every fact must be directly supported by exact wording from the source

VERIFICATION STEP:
Before outputting, check that NO fact adds new entities, actions, or events not explicitly in the source.

Text to analyze:
{text}

Extract ONLY literal facts (no interpretation, no invention):""",

    "second_pass": """Create a summary from the extracted facts. VERIFY that every fact is literally true to the source.

CRITICAL VERIFICATION:
- Before outputting, check that NO fact adds new entities, actions, or events not explicitly in the source
- Every fact must be supported by exact wording from the first_pass output
- DO NOT reinterpret metaphors as literal events
- If you cannot resolve something factually, quote the ambiguous phrase exactly

Extracted facts:
{text}

Write a complete summary (2-4 sentences, no mid-sentence cuts):
Summary:

Write a main topic as a concise, specific noun phrase (1-5 words, no partial sentences):
Main Topic:"""
}

# Chunk Summarizer Prompts - Strong anti-hallucination guardrails
CHUNK_SUMMARIZER_PROMPTS = {
    "first_pass": """Extract ONLY factual information from this text chunk. NO invention or interpretation.

CRITICAL RULES:
- Extract ONLY facts explicitly stated in the text
- DO NOT add new entities, actions, or events not in the source
- DO NOT reinterpret metaphors or figurative language as literal facts
- Quote ambiguous phrases exactly without interpretation

Text chunk:
{text}

Extract ONLY literal facts:""",

    "second_pass": """Summarize the extracted facts. VERIFY no invention or interpretation.

CRITICAL VERIFICATION:
- Every fact must be literally true to the source
- NO new entities, actions, or events not explicitly stated
- DO NOT reinterpret metaphors as literal events

Extracted facts:
{text}

Summarize in complete sentences:"""
}

# Final Summarizer Prompts - Strong anti-hallucination guardrails
FINAL_SUMMARIZER_PROMPTS = {
    "first_pass": """Extract ONLY factual information from these summaries. NO invention or interpretation.

CRITICAL RULES:
- Extract ONLY facts explicitly stated in the summaries
- DO NOT add new entities, actions, or events not in the source
- DO NOT reinterpret metaphors or figurative language as literal facts
- If you encounter ambiguous phrases, quote them exactly without interpretation

Summary texts:
{text}

Extract ONLY literal facts (no interpretation, no invention):""",

    "second_pass": """Create a final summary from the extracted facts. VERIFY literal truth to source.

CRITICAL VERIFICATION:
- Before outputting, check that NO fact adds new entities, actions, or events not explicitly in the source
- Every fact must be supported by exact wording from the first_pass output
- DO NOT reinterpret metaphors as literal events
- If you cannot resolve something factually, quote the ambiguous phrase exactly

Extracted facts:
{text}

Write a complete summary (no mid-sentence cuts):
Summary:

Write a main topic as a concise, specific noun phrase (1-5 words, no partial sentences):
Main Topic:"""
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
