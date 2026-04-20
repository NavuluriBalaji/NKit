import requests
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger("nkit.tools.builtin")

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DIRECTORY_RESULTS = 1000
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 0.5  # exponential backoff multiplier


class ValidationError(Exception):
    """Raised when tool input validation fails."""
    pass


class ToolError(Exception):
    """Base exception for tool execution errors."""
    pass


def _validate_query(query: str, max_length: int = 500) -> None:
    """Validate search query input.
    
    Args:
        query: Search query string
        max_length: Maximum query length
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty string")
    if len(query) > max_length:
        raise ValidationError(f"Query too long: {len(query)} chars (max {max_length})")
    if query.strip() == "":
        raise ValidationError("Query cannot be empty or whitespace only")


def _validate_file_path(file_path: str) -> Path:
    """Validate and sanitize file path.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or unsafe
    """
    if not file_path or not isinstance(file_path, str):
        raise ValidationError("File path must be a non-empty string")
    
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")
    
    # Prevent directory traversal
    if ".." in str(path):
        raise ValidationError("Path traversal (..) not allowed")
    
    return path


def _retry_with_backoff(func, max_retries: int = MAX_RETRIES, initial_wait: float = RETRY_BACKOFF):
    """Execute function with exponential backoff retry.
    
    Args:
        func: Callable to execute
        max_retries: Maximum retry attempts
        initial_wait: Initial wait time in seconds
        
    Returns:
        Function result
        
    Raises:
        ToolError: If all retries fail
    """
    wait_time = initial_wait
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except (Timeout, ConnectionError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                wait_time *= 2  # exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed")
        except RequestException as e:
            # Don't retry non-transient errors
            raise ToolError(f"Request failed: {e}")
    
    raise ToolError(f"Failed after {max_retries} retries: {last_error}")


class BuiltinTools:
    """Production-ready built-in tools with retry, validation, and resource limits.
    
    Instance-based approach for dependency injection, testability, and configuration.
    
    Example:
        ```python
        # Default configuration
        tools = BuiltinTools()
        result = tools.web_search("Python tutorial")
        
        # Custom configuration
        tools = BuiltinTools(timeout=30, max_retries=5)
        
        # Mock HTTP client for testing
        mock_client = MockHTTPClient()
        tools = BuiltinTools(http_client=mock_client)
        ```
    """
    
    def __init__(
        self,
        http_client=None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        max_file_size: int = MAX_FILE_SIZE,
        max_directory_results: int = MAX_DIRECTORY_RESULTS,
    ):
        """Initialize BuiltinTools with configurable dependencies.
        
        Args:
            http_client: HTTP client for web requests (default: requests module).
                        Can be mocked for testing.
            timeout: Default timeout for network requests (default 10s)
            max_retries: Default retry attempts for transient failures (default 3)
            max_file_size: Maximum file size for read_file (default 10MB)
            max_directory_results: Maximum items in directory listing (default 1000)
        """
        self.http_client = http_client or requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_file_size = max_file_size
        self.max_directory_results = max_directory_results
        
        logger.debug(
            f"BuiltinTools initialized: timeout={timeout}s, "
            f"retries={max_retries}, max_file_size={max_file_size}"
        )
    
    def web_search(self, query: str, num_results: int = 3, timeout: Optional[int] = None) -> str:
        """Search the web using DuckDuckGo API.
        
        Args:
            query: Search query string
            num_results: Number of results (currently for documentation)
            timeout: Request timeout in seconds (default: instance timeout)
            
        Returns:
            Search result string
            
        Raises:
            ValidationError: If query is invalid
            ToolError: If search fails after retries
        """
        _validate_query(query)
        
        effective_timeout = timeout or self.timeout
        if effective_timeout <= 0:
            raise ValidationError("Timeout must be positive")
        
        def execute():
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = self.http_client.get(url, timeout=effective_timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        
        try:
            return _retry_with_backoff(execute, max_retries=self.max_retries)
        except ToolError as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in web_search: {e}")
            raise ToolError(f"Web search failed: {e}")
    
    def read_file(self, file_path: str, max_size: Optional[int] = None) -> str:
        """Read file content with size limits.
        
        Args:
            file_path: Path to file
            max_size: Maximum file size in bytes (default: instance max_file_size)
            
        Returns:
            File content string
            
        Raises:
            ValidationError: If path is invalid
            ToolError: If read fails
        """
        try:
            path = _validate_file_path(file_path)
        except ValidationError as e:
            logger.error(f"Invalid file path: {e}")
            raise
        
        effective_max_size = max_size or self.max_file_size
        
        try:
            if not path.exists():
                raise ToolError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise ToolError(f"Path is not a file: {file_path}")
            
            file_size = path.stat().st_size
            if file_size > effective_max_size:
                raise ToolError(f"File too large: {file_size} bytes (max {effective_max_size})")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Successfully read {len(content)} bytes from {file_path}")
            return f"File content ({len(content)} characters):\n{content}"
        
        except ToolError:
            raise
        except UnicodeDecodeError as e:
            logger.error(f"File encoding error: {e}")
            raise ToolError(f"Could not decode file as UTF-8: {e}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise ToolError(f"Error reading file: {e}")
    
    def write_file(self, file_path: str, content: str, max_size: Optional[int] = None) -> str:
        """Write file content with size limits.
        
        Args:
            file_path: Path to file
            content: Content to write
            max_size: Maximum content size in bytes (default: instance max_file_size)
            
        Returns:
            Success message
            
        Raises:
            ValidationError: If path is invalid
            ToolError: If write fails
        """
        try:
            path = _validate_file_path(file_path)
        except ValidationError as e:
            logger.error(f"Invalid file path: {e}")
            raise
        
        if not isinstance(content, str):
            raise ValidationError("Content must be a string")
        
        effective_max_size = max_size or self.max_file_size
        if len(content) > effective_max_size:
            raise ToolError(f"Content too large: {len(content)} bytes (max {effective_max_size})")
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully wrote {len(content)} bytes to {file_path}")
            return f"Successfully wrote {len(content)} characters to {file_path}"
        
        except PermissionError as e:
            logger.error(f"Permission denied writing to {file_path}: {e}")
            raise ToolError(f"Permission denied: {e}")
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise ToolError(f"Error writing file: {e}")
    
    def list_files(self, directory: str = ".", max_results: Optional[int] = None) -> str:
        """List directory contents with result limits.
        
        Args:
            directory: Directory path
            max_results: Maximum results to return (default: instance max_directory_results)
            
        Returns:
            Formatted directory listing
            
        Raises:
            ValidationError: If path is invalid
            ToolError: If listing fails
        """
        try:
            path = _validate_file_path(directory)
        except ValidationError as e:
            logger.error(f"Invalid directory path: {e}")
            raise
        
        effective_max_results = max_results or self.max_directory_results
        
        try:
            if not path.exists():
                raise ToolError(f"Directory not found: {directory}")
            
            if not path.is_dir():
                raise ToolError(f"Path is not a directory: {directory}")
            
            files = []
            dirs = []
            count = 0
            
            for item in path.iterdir():
                if count >= effective_max_results:
                    logger.warning(f"Directory listing truncated at {effective_max_results} items")
                    break
                
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    dirs.append(item.name + "/")
                
                count += 1
            
            all_items = sorted(dirs + files)
            logger.info(f"Listed {len(all_items)} items from {directory}")
            return f"Contents of {directory}:\n" + "\n".join(all_items)
        
        except ToolError:
            raise
        except PermissionError as e:
            logger.error(f"Permission denied listing {directory}: {e}")
            raise ToolError(f"Permission denied: {e}")
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")
            raise ToolError(f"Error listing directory: {e}")
    
    def get_current_time(self, timezone: str = "UTC") -> str:
        """Get current date and time.
        
        Args:
            timezone: Timezone (currently returns UTC)
            
        Returns:
            Current time string
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"Current time: {current_time}")
        return f"Current time: {current_time} ({timezone})"
    
    async def async_web_search(self, query: str, num_results: int = 3, timeout: Optional[int] = None) -> str:
        """Async web search using DuckDuckGo API.
        
        Args:
            query: Search query string
            num_results: Number of results (for documentation)
            timeout: Request timeout in seconds (default: instance timeout)
            
        Returns:
            Search result string
            
        Raises:
            ValidationError: If query is invalid
            ToolError: If search fails after retries
        """
        import aiohttp
        
        _validate_query(query)
        
        effective_timeout = timeout or self.timeout
        if effective_timeout <= 0:
            raise ValidationError("Timeout must be positive")
        
        async def execute():
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=effective_timeout)) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        
        try:
            return await execute()
        except Exception as e:
            logger.error(f"Async web search failed for query '{query}': {e}")
            raise ToolError(f"Async web search failed: {e}")


__all__ = ["BuiltinTools", "ToolError", "ValidationError"]
