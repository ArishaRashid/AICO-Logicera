from typing import Optional, Type
from pydantic import BaseModel, Field
import httpx
from bs4 import BeautifulSoup
from readability import Document
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)

class WebBrowserToolInput(BaseModel):
    url: str = Field(..., description="A fully-qualified URL starting with http or https")

def _extract_main_text(html: str) -> str:
    """
    This function tries to extract the main content from a web page.
    It first tries to use the 'readability' library to find the main article),
    and if that fails, it falls back to a simpler approach.
    """
    try:
        # Try the readability approach first 
        doc = Document(html)
        html_article = doc.summary(html_partial=True)
        text = BeautifulSoup(html_article, "html.parser").get_text("\n", strip=True)
        if len(text.split()) > 50:
            return text
    except Exception:
        pass
    
    # If readability didn't work, let's try a simpler approach
    soup = BeautifulSoup(html, "html.parser")
    # Clean up the HTML by removing scripts, styles, and other non-content stuff
    remove_tags = [
        "script", "style", "noscript", "head", "meta", "link", "title",
        "nav", "header", "footer", "aside",
        "iframe", "embed", "object", "video", "audio", "picture",
        "form", "input", "button", "select", "textarea", "label",
        "ins", "svg", "canvas"
    ]

    for tag in soup(remove_tags):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text

class WebBrowserTool(BaseTool):
    """
    This tool is like giving your AI a web browser! It can fetch web pages and extract the main text content.
    Just give it a URL and it'll read the page for you.
    """
    name: str = "WebBrowserTool"
    description: str = (
        "Fetch and read the main text content of a web page. "
        "Use this BEFORE summarizing or answering follow-up questions about a URL."
    )
    args_schema: Type[BaseModel] = WebBrowserToolInput
    timeout_s: int = 20

    def _run(self,url: str,run_manager: Optional[CallbackManagerForToolRun] = None,) -> str:
        # Make sure we have a proper URL
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        
        try:
            # Set up a web client with a friendly user agent and reasonable timeout
            with httpx.Client(follow_redirects=True, timeout=self.timeout_s, headers={
                "User-Agent": "Mozilla/5.0 (LangChain-Agent;)"
            }) as client:
                resp = client.get(url)
                resp.raise_for_status()
                
                # Make sure we're getting HTML content
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    raise ValueError(f"Unsupported content-type: {content_type}")
                
                # Extract the main text from the HTML
                text = _extract_main_text(resp.text)
                if not text or len(text.strip()) < 50:
                    raise ValueError("Page text too short or unreadable.")
                
                # For really long pages, we'll truncate to keep things manageable
                return text[:150_000]
        except Exception as e:
            raise ValueError(f"Failed to fetch or parse page: {e}") from e

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(url)
