"""Command-line interface for nkit framework.

This module provides CLI commands for:
- Project scaffolding
- Agent/crew templates
- Configuration management

Architecture:
    - Commands: Individual CLI operations
    - Templates: Project/agent boilerplate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


AGENT_TEMPLATE = '''"""Agent implementation."""

from nkit.agent import Agent
from nkit.tools import Tool


def my_tool(param: str) -> str:
    """Tool description.
    
    Args:
        param: Parameter description
    
    Returns:
        Result description
    """
    return f"Processed: {param}"


# Create agent
agent = Agent(
    llm_client=None,  # Configure your LLM client
    tools=[Tool(name="my_tool", func=my_tool)]
)

if __name__ == "__main__":
    result = agent.run("Your task here")
    print(result)
'''


CREW_TEMPLATE = '''"""Crew implementation."""

from nkit.crews import Crew, Agent, Task, ProcessType


# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="Expert researcher with attention to detail"
)

analyst = Agent(
    role="Analyst", 
    goal="Analyze data and identify insights",
    backstory="Experienced data analyst"
)

# Define tasks
research_task = Task(
    description="Research the topic",
    expected_output="Detailed research findings",
    agent=researcher
)

analysis_task = Task(
    description="Analyze research findings",
    expected_output="Analysis report with insights",
    agent=analyst,
    dependencies=[research_task]
)

# Create crew
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=ProcessType.SEQUENTIAL,
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result.raw)
'''


CONFIG_TEMPLATE = {
    "project_name": "my_project",
    "version": "0.1.0",
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7
    },
    "memory": {
        "backend": "json",
        "path": "./memory.json"
    },
    "tools": []
}


def create_agent(name: str, directory: Optional[Path] = None) -> None:
    """Create a new agent file.
    
    Args:
        name: Agent name
        directory: Output directory
    """
    directory = directory or Path.cwd()
    filepath = directory / f"{name}.py"
    
    if filepath.exists():
        print(f"Error: {filepath} already exists")
        sys.exit(1)
    
    filepath.write_text(AGENT_TEMPLATE)
    print(f"Created agent: {filepath}")


def create_crew(name: str, directory: Optional[Path] = None) -> None:
    """Create a new crew file.
    
    Args:
        name: Crew name
        directory: Output directory
    """
    directory = directory or Path.cwd()
    filepath = directory / f"{name}.py"
    
    if filepath.exists():
        print(f"Error: {filepath} already exists")
        sys.exit(1)
    
    filepath.write_text(CREW_TEMPLATE)
    print(f"Created crew: {filepath}")


def create_project(name: str, directory: Optional[Path] = None) -> None:
    """Create a new project structure.
    
    Args:
        name: Project name
        directory: Output directory
    """
    directory = directory or Path.cwd()
    project_dir = directory / name
    
    if project_dir.exists():
        print(f"Error: {project_dir} already exists")
        sys.exit(1)
    
    # Create project structure
    project_dir.mkdir(parents=True)
    (project_dir / "agents").mkdir()
    (project_dir / "crews").mkdir()
    (project_dir / "tools").mkdir()
    (project_dir / "data").mkdir()
    
    # Create config
    config = CONFIG_TEMPLATE.copy()
    config["project_name"] = name
    config_file = project_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2))
    
    # Create README
    readme = f"""# {name}

## Structure

- `agents/` - Individual agent implementations
- `crews/` - Multi-agent crews
- `tools/` - Custom tools
- `data/` - Data files
- `config.json` - Project configuration

## Usage

```python
# Run an agent
python agents/my_agent.py

# Run a crew
python crews/my_crew.py
```

## Configuration

Edit `config.json` to configure LLM provider, memory backend, etc.
"""
    (project_dir / "README.md").write_text(readme)
    
    # Create .gitignore
    gitignore = """__pycache__/
*.pyc
*.pyo
.env
memory.json
*.log
"""
    (project_dir / ".gitignore").write_text(gitignore)
    
    print(f"Created project: {project_dir}")
    print(f"\nNext steps:")
    print(f"  cd {name}")
    print(f"  # Edit config.json")
    print(f"  # Create agents: python -m nkit.cli create agent my_agent")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nkit",
        description="nkit CLI - Agent framework toolkit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create new components")
    create_subparsers = create_parser.add_subparsers(dest="component")
    
    # Create agent
    agent_parser = create_subparsers.add_parser("agent", help="Create new agent")
    agent_parser.add_argument("name", help="Agent name")
    agent_parser.add_argument("-d", "--directory", help="Output directory")
    
    # Create crew
    crew_parser = create_subparsers.add_parser("crew", help="Create new crew")
    crew_parser.add_argument("name", help="Crew name")
    crew_parser.add_argument("-d", "--directory", help="Output directory")
    
    # Create project
    project_parser = create_subparsers.add_parser("project", help="Create new project")
    project_parser.add_argument("name", help="Project name")
    project_parser.add_argument("-d", "--directory", help="Output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "create":
        directory = Path(args.directory) if hasattr(args, "directory") and args.directory else None
        
        if args.component == "agent":
            create_agent(args.name, directory)
        elif args.component == "crew":
            create_crew(args.name, directory)
        elif args.component == "project":
            create_project(args.name, directory)
        else:
            create_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["create_agent", "create_crew", "create_project", "main"]
