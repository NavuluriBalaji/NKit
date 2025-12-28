import requests
from datetime import datetime
from pathlib import Path


class BuiltinTools:
    @staticmethod
    def web_search(query: str, num_results: int = 3) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        except Exception as e:
            return f"Web search failed: {e}"
    
    @staticmethod
    def read_file(file_path: str) -> str:
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File content ({len(content)} characters):\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    def list_files(directory: str = ".") -> str:
        try:
            path = Path(directory)
            if not path.exists():
                return f"Directory not found: {directory}"
            files = [f.name for f in path.iterdir() if f.is_file()]
            dirs = [f.name + "/" for f in path.iterdir() if f.is_dir()]
            all_items = sorted(dirs + files)
            return f"Contents of {directory}:\n" + "\n".join(all_items)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    @staticmethod
    def get_current_time() -> str:
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    @staticmethod
    async def async_web_search(query: str, num_results: int = 3) -> str:
        import aiohttp
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        except Exception as e:
            return f"Async web search failed: {e}"

__all__ = ["BuiltinTools"]
