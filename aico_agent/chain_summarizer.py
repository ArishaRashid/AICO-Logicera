
import logging
import operator
import os
from typing import Annotated, List, Literal, TypedDict, Dict, Any

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import MODELS, OPENAI_API, CHUNKING

logger = logging.getLogger(__name__)

# =============================================================================
# ADVANCED SUMMARIZATION TECHNIQUES
# =============================================================================

def build_stuff_summarizer() -> Runnable:
    """Build an enhanced 'stuff' summarization chain with structured output."""
    logger.info("🔧 Building enhanced stuff summarization chain...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Enhanced prompt for structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert summarizer. Create a comprehensive summary with structured output.

TASK: Analyze the text and provide structured information.

IMPORTANT: 
- Focus on the most relevant and high-value content
- Extract key facts and insights
- Identify the main topic and supporting points
- Maintain logical flow and coherence

CRITICAL: Avoid including navigation elements, ads, or irrelevant content.

REQUIRED OUTPUT FORMAT:
Main Topic: [Provide a brief, focused topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Provide a comprehensive summary paragraph that covers the main content, key facts, and important details]

Text to summarize:
{text}

Now complete the task step by step:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    logger.info("✅ Enhanced stuff summarization chain built successfully")
    return chain

def build_hierarchical_summarizer() -> Runnable:
    """Build a hierarchical summarization chain with multiple levels of detail."""
    logger.info("🔧 Building hierarchical summarization chain...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Level 1: Extract key facts
    level1_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract the key facts and main points from this text.

TASK: Identify and extract the most important information.

IMPORTANT:
- Focus on factual information
- Extract key insights and data points
- Identify main themes and concepts
- Filter out irrelevant content

Text to analyze:
{text}

Extract the key facts:""")
    ])
    
    # Level 2: Create structured summary
    level2_prompt = ChatPromptTemplate.from_messages([
        ("system", """Based on the extracted facts, create a structured summary.

TASK: Organize the information into a clear, structured format.

REQUIREMENTS:
1. Main topic and context
2. Key points
3. Important details and supporting information
4. Overall conclusion or takeaway

Extracted facts:
{text}

Create the structured summary:""")
    ])
    
    # Level 3: Final polished summary with required format
    level3_prompt = ChatPromptTemplate.from_messages([
        ("system", """Create a final, polished summary with the EXACT required format.

TASK: Produce a professional, coherent summary in the specified format.

REQUIRED OUTPUT FORMAT:
Main Topic: [Brief topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Comprehensive summary paragraph covering main content, key facts, and important details]

Structured content:
{text}

Provide the final summary in the EXACT format above:""")
    ])
    
    # Create the hierarchical chain
    level1_chain = level1_prompt | llm | StrOutputParser()
    level2_chain = level2_prompt | llm | StrOutputParser()
    level3_chain = level3_prompt | llm | StrOutputParser()
    
    def hierarchical_summarize(input_dict):
        text = input_dict["text"]
        # Level 1: Extract facts
        facts = level1_chain.invoke({"text": text})
        # Level 2: Create structure
        structure = level2_chain.invoke({"text": facts})
        # Level 3: Final summary
        final = level3_chain.invoke({"text": structure})
        return final
    
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(hierarchical_summarize)
    
    logger.info("✅ Hierarchical summarization chain built successfully")
    return chain

def build_recursive_summarizer() -> Runnable:
    """Build a recursive summarization chain that handles very long texts."""
    logger.info("🔧 Building recursive summarization chain...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Recursive summarization prompt with required format
    recursive_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert summarizer. Create a comprehensive summary of the given text.

TASK: If the text is very long, break it down into logical sections and summarize each section, then combine them into a final summary.

IMPORTANT:
- Focus on key facts and main points
- Important details and context
- Logical flow and connections between ideas
- Avoid repetition while maintaining completeness

REQUIRED OUTPUT FORMAT:
Main Topic: [Brief topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Comprehensive summary paragraph covering all sections and main content]

Text to summarize:
{text}

Provide a comprehensive summary in the EXACT format above:""")
    ])
    
    chain = recursive_prompt | llm | StrOutputParser()
    
    logger.info("✅ Recursive summarization chain built successfully")
    return chain

def build_chain_of_thought_summarizer() -> Runnable:
    """Build a chain-of-thought summarization chain that shows reasoning steps."""
    logger.info("🔧 Building chain-of-thought summarization chain...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Chain-of-thought prompt with required format
    cot_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert summarizer. Use chain-of-thought reasoning to create a comprehensive summary.

TASK: Follow these steps to create a well-reasoned summary.

STEP-BY-STEP PROCESS:
1. First, identify the main topic and key themes
2. Then, extract the most important facts and details
3. Next, organize the information logically
4. Finally, create a coherent summary

REQUIRED OUTPUT FORMAT:
Main Topic: [Brief topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Comprehensive summary paragraph based on your reasoning]

Text to summarize:
{text}

Let me think through this step by step:""")
    ])
    
    chain = cot_prompt | llm | StrOutputParser()
    
    logger.info("✅ Chain-of-thought summarization chain built successfully")
    return chain

def build_tree_of_thoughts_summarizer() -> Runnable:
    """Build a tree-of-thoughts summarization chain that explores multiple approaches."""
    logger.info("🔧 Building tree-of-thoughts summarization chain...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Tree-of-thoughts prompt with required format
    tot_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert summarizer. Use tree-of-thoughts reasoning to explore multiple summarization approaches.

TASK: Consider different approaches and choose the best one.

APPROACHES TO CONSIDER:
1. Factual extraction approach
2. Thematic analysis approach  
3. Chronological approach
4. Problem-solution approach
5. Comparative approach

REQUIRED OUTPUT FORMAT:
Main Topic: [Brief topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Comprehensive summary using the best approach you selected]

Text to summarize:
{text}

Let me explore different approaches and provide the summary in the EXACT format above:""")
    ])
    
    chain = tot_prompt | llm | StrOutputParser()
    
    logger.info("✅ Tree-of-thoughts summarization chain built successfully")
    return chain

def build_adaptive_summarizer() -> Runnable:
    """Build an adaptive summarization chain that chooses the best approach based on content length."""
    logger.info("🔧 Building adaptive summarization chain...")
    
    # Get all summarization approaches
    stuff_chain = build_stuff_summarizer()
    hierarchical_chain = build_hierarchical_summarizer()
    recursive_chain = build_recursive_summarizer()
    
    def adaptive_summarize(input_dict):
        text = input_dict["text"]
        text_length = len(text)
        
        logger.info(f"🔍 Content length: {text_length} characters")
        
        # Choose approach based on content length
        if text_length <= CHUNKING["chunk_threshold"]:
            logger.info("📝 Using enhanced stuff summarization for short content")
            return stuff_chain.invoke({"text": text})
        elif text_length <= CHUNKING["chunk_threshold"] * 3:
            logger.info("🔄 Using hierarchical summarization for medium content")
            return hierarchical_chain.invoke({"text": text})
        else:
            logger.info("🔄 Using recursive summarization for long content")
            return recursive_chain.invoke({"text": text})
    
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(adaptive_summarize)
    
    logger.info("✅ Adaptive summarization chain built successfully")
    return chain

def build_advanced_summarizer() -> Runnable:
    """Build the main advanced summarization orchestrator."""
    logger.info("🔧 Building advanced summarization orchestrator...")
    
    # Get all summarization techniques
    techniques = {
        "stuff": build_stuff_summarizer(),
        "hierarchical": build_hierarchical_summarizer(),
        "recursive": build_recursive_summarizer(),
        "chain_of_thought": build_chain_of_thought_summarizer(),
        "tree_of_thoughts": build_tree_of_thoughts_summarizer(),
        "adaptive": build_adaptive_summarizer()
    }
    
    def advanced_summarize(input_dict):
        text = input_dict["text"]
        technique = input_dict.get("technique", "adaptive")
        
        logger.info(f"🔍 Using summarization technique: {technique}")
        
        if technique == "adaptive":
            # Use the adaptive summarizer
            adaptive_chain = build_adaptive_summarizer()
            return adaptive_chain.invoke({"text": text})
        elif technique in techniques:
            # Use the specified technique
            return techniques[technique].invoke({"text": text})
        else:
            # Default to adaptive
            logger.warning(f"⚠️ Unknown technique '{technique}', using adaptive")
            adaptive_chain = build_adaptive_summarizer()
            return adaptive_chain.invoke({"text": text})
    
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(advanced_summarize)
    
    logger.info("✅ Advanced summarization orchestrator built successfully")
    return chain

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_techniques() -> List[str]:
    """Get list of available summarization techniques."""
    return [
        "adaptive",           # Automatically chooses best approach
        "stuff",             # Enhanced single-pass summarization
        "hierarchical",      # Multi-level summarization
        "recursive",         # Recursive breakdown approach
        "chain_of_thought",  # Step-by-step reasoning
        "tree_of_thoughts"   # Multiple approach exploration
    ]

def get_technique_description(technique: str) -> str:
    """Get description of a summarization technique."""
    descriptions = {
        "adaptive": "Automatically chooses the best summarization approach based on content length and complexity",
        "stuff": "Enhanced single-pass summarization with structured output and key facts extraction",
        "hierarchical": "Multi-level summarization with increasing detail and structure (analysis → extraction → synthesis)",
        "recursive": "Recursively breaks down long texts into manageable sections",
        "chain_of_thought": "Step-by-step reasoning approach for complex content",
        "tree_of_thoughts": "Explores multiple summarization approaches before choosing the best one"
    }
    return descriptions.get(technique, "Unknown technique")
