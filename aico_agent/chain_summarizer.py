
import logging
import os
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from config import MODELS, OPENAI_API, CHUNKING

logger = logging.getLogger(__name__)

# =============================================================================
# LANGCHAIN SUMMARIZATION TECHNIQUES
# =============================================================================

def build_stuff_summarizer() -> Runnable:
    """Build a 'stuff' summarization chain using LangChain's load_summarize_chain."""
    logger.info("🔧 Building stuff summarization chain using LangChain...")
    
    try:
        llm = ChatOpenAI(
            model=MODELS["summarizer"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
            max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
            timeout=OPENAI_API["timeout"]
        )
        
        # Use LangChain's built-in stuff chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """You are an expert summarizer. Create a comprehensive summary with structured output.

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
        )
        
        # Create a wrapper that converts text input to the format expected by LangChain
        def stuff_summarize_wrapper(input_dict):
            try:
                text = input_dict["text"]
                logger.info(f"🔍 Stuff summarizer received text of length: {len(text)}")
                # Convert text to Document format expected by LangChain
                from langchain.schema import Document
                documents = [Document(page_content=text)]
                logger.info(f"📄 Converted to {len(documents)} documents for LangChain")
                result = chain.invoke({"input_documents": documents})
                # Extract the text content from the chain output
                logger.info(f"🔍 Chain output type: {type(result)}")
                logger.info(f"🔍 Chain output content: {result}")
                if isinstance(result, dict) and "output_text" in result:
                    logger.info("✅ Found output_text in dict result")
                    return result["output_text"]
                elif isinstance(result, str):
                    logger.info("✅ Result is already a string")
                    return result
                else:
                    logger.warning(f"⚠️ Unexpected chain output format: {type(result)}")
                    logger.warning(f"⚠️ Available keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    return str(result)
            except Exception as e:
                logger.error(f"❌ Error in stuff summarizer wrapper: {e}")
                logger.error(f"❌ Input dict keys: {list(input_dict.keys())}")
                raise e
        
        from langchain_core.runnables import RunnableLambda
        wrapped_chain = RunnableLambda(stuff_summarize_wrapper)
        
        logger.info("✅ Stuff summarization chain built successfully using LangChain")
        return wrapped_chain
        
    except Exception as e:
        logger.error(f"❌ Error building stuff summarizer: {e}")
        # Fallback to simple prompt-based approach
        logger.info("🔄 Falling back to simple prompt-based summarizer")
        return build_fallback_summarizer()

def build_fallback_summarizer() -> Runnable:
    """Build a fallback summarizer using simple prompts when LangChain chains fail."""
    logger.info("🔧 Building fallback summarizer...")
    
    llm = ChatOpenAI(
        model=MODELS["summarizer"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
        max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
        timeout=OPENAI_API["timeout"]
    )
    
    # Simple prompt-based approach
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert summarizer. Create a comprehensive summary with structured output.

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
    
    logger.info("✅ Fallback summarizer built successfully")
    return chain

def build_map_reduce_summarizer() -> Runnable:
    """Build a map-reduce summarization chain using LangChain's load_summarize_chain."""
    logger.info("🔧 Building map-reduce summarization chain using LangChain...")
    
    try:
        llm = ChatOpenAI(
            model=MODELS["summarizer"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
            max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
            timeout=OPENAI_API["timeout"]
        )
        
        # Use LangChain's built-in map-reduce chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=ChatPromptTemplate.from_messages([
                ("system", """Extract the key facts and main points from this text section.

Focus on:
- Factual information
- Key insights and data points
- Main themes and concepts
- Important details

Text section:
{text}

Extract key facts:""")
            ]),
            combine_prompt=ChatPromptTemplate.from_messages([
                ("system", """Based on the extracted facts from all sections, create a comprehensive summary.

REQUIRED OUTPUT FORMAT:
Main Topic: [Provide a brief, focused topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Provide a comprehensive summary paragraph that covers the main content, key facts, and important details]

Extracted facts:
{text}

Create the final summary:""")
            ])
        )
        
        # Create a wrapper that converts text input to the format expected by LangChain
        def map_reduce_summarize_wrapper(input_dict):
            try:
                text = input_dict["text"]
                logger.info(f"🔍 Map-reduce summarizer received text of length: {len(text)}")
                # Convert text to Document format expected by LangChain
                from langchain.schema import Document
                documents = [Document(page_content=text)]
                logger.info(f"📄 Converted to {len(documents)} documents for LangChain")
                result = chain.invoke({"input_documents": documents})
                # Extract the text content from the chain output
                logger.info(f"🔍 Chain output type: {type(result)}")
                logger.info(f"🔍 Chain output content: {result}")
                if isinstance(result, dict) and "output_text" in result:
                    logger.info("✅ Found output_text in dict result")
                    return result["output_text"]
                elif isinstance(result, str):
                    logger.info("✅ Result is already a string")
                    return result
                else:
                    logger.warning(f"⚠️ Unexpected chain output format: {type(result)}")
                    logger.warning(f"⚠️ Available keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    return str(result)
            except Exception as e:
                logger.error(f"❌ Error in map-reduce summarizer wrapper: {e}")
                logger.error(f"❌ Input dict keys: {list(input_dict.keys())}")
                raise e
        
        from langchain_core.runnables import RunnableLambda
        wrapped_chain = RunnableLambda(map_reduce_summarize_wrapper)
        
        logger.info("✅ Map-reduce summarization chain built successfully using LangChain")
        return wrapped_chain
        
    except Exception as e:
        logger.error(f"❌ Error building map-reduce summarizer: {e}")
        logger.info("🔄 Falling back to simple prompt-based summarizer")
        return build_fallback_summarizer()

def build_refine_summarizer() -> Runnable:
    """Build a refine summarization chain using LangChain's load_summarize_chain."""
    logger.info("🔧 Building refine summarization chain using LangChain...")
    
    try:
        llm = ChatOpenAI(
            model=MODELS["summarizer"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("temperature", 0.1),
            max_tokens=OPENAI_API["model_parameters"].get(MODELS["summarizer"], {}).get("max_tokens", 2000),
            timeout=OPENAI_API["timeout"]
        )
        
        # Use LangChain's built-in refine chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            refine_prompt=ChatPromptTemplate.from_messages([
                ("system", """Based on the existing summary and new information, refine and improve the summary.

REQUIRED OUTPUT FORMAT:
Main Topic: [Provide a brief, focused topic description in 1-2 sentences]

Key Points:
- [First key point]
- [Second key point]
- [Third key point]
- [Fourth key point if applicable]

Summary: [Provide a comprehensive summary paragraph that covers the main content, key facts, and important details]

Existing summary:
{existing_answer}

New information to incorporate:
{text}

Refined summary:""")
            ]),
            question_prompt=ChatPromptTemplate.from_messages([
                ("system", """Extract the key facts and main points from this text section.

Focus on:
- Factual information
- Key insights and data points
- Main themes and concepts
- Important details

Text section:
{text}

Extract key facts:""")
            ])
        )
        
        # Create a wrapper that converts text input to the format expected by LangChain
        def refine_summarize_wrapper(input_dict):
            try:
                text = input_dict["text"]
                logger.info(f"🔍 Refine summarizer received text of length: {len(text)}")
                # Convert text to Document format expected by LangChain
                from langchain.schema import Document
                documents = [Document(page_content=text)]
                logger.info(f"📄 Converted to {len(documents)} documents for LangChain")
                result = chain.invoke({"input_documents": documents})
                # Extract the text content from the chain output
                logger.info(f"🔍 Chain output type: {type(result)}")
                logger.info(f"🔍 Chain output content: {result}")
                if isinstance(result, dict) and "output_text" in result:
                    logger.info("✅ Found output_text in dict result")
                    return result["output_text"]
                elif isinstance(result, str):
                    logger.info("✅ Result is already a string")
                    return result
                else:
                    logger.warning(f"⚠️ Unexpected chain output format: {type(result)}")
                    logger.warning(f"⚠️ Available keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    return str(result)
            except Exception as e:
                logger.error(f"❌ Error in refine summarizer wrapper: {e}")
                logger.error(f"❌ Input dict keys: {list(input_dict.keys())}")
                raise e
        
        from langchain_core.runnables import RunnableLambda
        wrapped_chain = RunnableLambda(refine_summarize_wrapper)
        
        logger.info("✅ Refine summarization chain built successfully using LangChain")
        return wrapped_chain
        
    except Exception as e:
        logger.error(f"❌ Error building refine summarizer: {e}")
        logger.info("🔄 Falling back to simple prompt-based summarizer")
        return build_fallback_summarizer()

def build_adaptive_summarizer() -> Runnable:
    """Build an adaptive summarization chain that chooses the best approach based on content length."""
    logger.info("🔧 Building adaptive summarization chain using LangChain...")
    
    def adaptive_summarize(input_dict):
        text = input_dict["text"]
        text_length = len(text)
        
        logger.info(f"🔍 Content length: {text_length} characters")
        
        # Choose approach based on content length using LangChain chains
        if text_length <= CHUNKING["chunk_threshold"]:
            logger.info("📝 Using stuff summarization for short content")
            chain = build_stuff_summarizer()
            return chain.invoke({"text": text})
        elif text_length <= CHUNKING["chunk_threshold"] * 3:
            logger.info("🔄 Using map-reduce summarization for medium content")
            chain = build_map_reduce_summarizer()
            return chain.invoke({"text": text})
        else:
            logger.info("🔄 Using refine summarization for long content")
            chain = build_refine_summarizer()
            return chain.invoke({"text": text})
    
    from langchain_core.runnables import RunnableLambda
    chain = RunnableLambda(adaptive_summarize)
    
    logger.info("✅ Adaptive summarization chain built successfully using LangChain")
    return chain

def build_advanced_summarizer() -> Runnable:
    """Build the main advanced summarization orchestrator using LangChain chains."""
    logger.info("🔧 Building advanced summarization orchestrator using LangChain...")
    
    # Get all summarization techniques
    techniques = {
        "stuff": build_stuff_summarizer(),
        "map_reduce": build_map_reduce_summarizer(),
        "refine": build_refine_summarizer(),
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
    
    logger.info("✅ Advanced summarization orchestrator built successfully using LangChain")
    return chain

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_techniques() -> List[str]:
    """Get list of available summarization techniques."""
    return [
        "adaptive",           # Automatically chooses best approach
        "stuff",             # Single-pass summarization using LangChain
        "map_reduce",        # Map-reduce approach using LangChain
        "refine"             # Refine approach using LangChain
    ]

def get_technique_description(technique: str) -> str:
    """Get description of a summarization technique."""
    descriptions = {
        "adaptive": "Automatically chooses the best summarization approach based on content length and complexity using LangChain",
        "stuff": "Single-pass summarization using LangChain's stuff chain for short content",
        "map_reduce": "Map-reduce summarization using LangChain for medium-length content",
        "refine": "Refine summarization using LangChain for long content with iterative improvement"
    }
    return descriptions.get(technique, "Unknown technique")

def split_text_for_summarization(text: str) -> List[Document]:
    """Split text into chunks suitable for summarization using LangChain's text splitter."""
    logger.info(f"✂️ Splitting text of length {len(text)} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING["chunk_size"],
        chunk_overlap=CHUNKING["chunk_overlap"],
        separators=CHUNKING["separators"]
    )
    
    # Split text into documents
    documents = text_splitter.create_documents([text])
    logger.info(f"✅ Split text into {len(documents)} chunks")
    
    return documents
