import sys
import subprocess
import tempfile
import os
import traceback
from typing import Any, List

from . import Tool, ToolRegistry


def execute_python(code: str) -> str:
    """
    Executes raw Python code in an isolated subprocess.
    This acts as the 'God Tool', allowing the agent to dynamically explore,
    evaluate, or script complex actions on the fly instead of relying on
    pre-installed IDE limits.
    
    Args:
        code: The raw Python code to execute.
    
    Returns:
        The stdout/stderr of the executed script.
    """
    # Write the dynamically generated code to a temporary sandbox file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name
        
    try:
        # Run python in a 30s timeout sandbox natively
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=30)
        output = result.stdout
        
        # Merge StdErr if exists so the agent can self-heal exceptions
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"
            
        return output if output.strip() else "Execution successful, no output."
        
    except subprocess.TimeoutExpired:
        return "Execution Error: Script timed out after 30 seconds."
    except Exception as e:
        return f"Execution Error: {str(e)}"
    finally:
        # Always clean up the sandbox
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class ToolSmith:
    """
    Runtime Tool Injection Factory.
    Allows an agent to dynamically write a Python function, evaluate it into memory,
    and bind it directly into its active ToolRegistry mid-execution!
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def register_dynamic_tool(self, tool_name: str, python_code: str, description: str) -> str:
        """
        Dynamically registers a new tool for the agent to use.
        
        Args:
            tool_name: The EXACT name of the function defined in the code.
            python_code: The Python script that defines the tool function. 
            description: What the tool does and what arguments it expects.
            
        Returns:
            Success or error message string.
        """
        local_vars = {}
        try:
            # We safely execute the string in an isolated local dictionary scope
            exec(python_code, globals(), local_vars)
            
            func = local_vars.get(tool_name)
            if not func or not callable(func):
                return f"Injection Error: The code did not define a callable function named '{tool_name}'."
                
            # Wrap in Native NKit Tool class explicitly bypassing static limits
            dynamic_tool = Tool(
                name=tool_name,
                func=func,
                desc=description
            )
            
            # Inject directly into the running instance's memory!
            self.registry.register(dynamic_tool)
            return f"Success! Dynamic tool '{tool_name}' has been compiled and injected into the registry. You may now call it identically to built-in tools."
            
        except Exception as e:
            trace = traceback.format_exc()
            return f"Injection Error: Failed to compile and bind the tool.\\n{e}\\n{trace}"


def get_codeact_tools(registry: ToolRegistry) -> List[Tool]:
    """
    Helper function to instantly arm an Agent with CodeACT dynamic capabilities.
    """
    toolsmith = ToolSmith(registry)
    
    python_executor = Tool(
        name="execute_python",
        func=execute_python,
        desc="Executes raw Python code string locally and returns the printed output/errors. Use this to dynamically script actions, test code, or explore the filesystem without hardcoded tools. Pass valid python in the 'code' parameter."
    )
    
    tool_injector = Tool(
        name="inject_dynamic_tool",
        func=toolsmith.register_dynamic_tool,
        desc="Writes a new Python tool function and binds it into your ToolRegistry so you can use it iteratively later. You MUST provide function source code that defines a function named exactly `tool_name`."
    )
    
    return [python_executor, tool_injector]
