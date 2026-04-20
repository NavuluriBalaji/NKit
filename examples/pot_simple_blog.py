"""Simple Blog Agent: Copy, Modify, and Use!

Start here to write your first blog with PoT.
Just modify the BLOG_CONFIG and run!
"""

import sys
import os
import asyncio
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from NKit.tools import ToolRegistry, Tool
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor


# ============================================================================
# GET USER INPUT FOR BLOG CONFIGURATION
# ============================================================================

def get_user_input():
    """Get blog configuration from user."""
    print("\n" + "="*70)
    print("📝 BLOG GENERATION: Interactive Mode")
    print("="*70 + "\n")
    
    # Get topic
    topic = input("📌 What's your blog topic? (e.g., 'Python async/await'): ").strip()
    if not topic:
        topic = "Web Development with React"
        print(f"   → Using default: {topic}")
    
    # Get style
    print("\n📖 What writing style? (technical, casual, professional)")
    style = input("   Enter style [technical]: ").strip().lower()
    if style not in ["technical", "casual", "professional"]:
        style = "technical"
        print(f"   → Using default: {style}")
    
    # Get audience
    print("\n👥 Who's your audience? (developers, business, general)")
    audience = input("   Enter audience [developers]: ").strip().lower()
    if audience not in ["developers", "business", "general"]:
        audience = "developers"
        print(f"   → Using default: {audience}")
    
    # Get word count target
    print("\n📊 Target word count? (1000-5000)")
    word_count = input("   Enter word count [2000]: ").strip()
    try:
        word_count = int(word_count)
        if word_count < 500 or word_count > 10000:
            word_count = 2000
            print(f"   → Using default: {word_count}")
    except ValueError:
        word_count = 2000
        print(f"   → Using default: {word_count}")
    
    # Get custom sections
    print("\n🏗️  Add custom sections? (comma-separated, or press Enter for defaults)")
    custom_sections = input("   Sections: ").strip()
    
    if custom_sections:
        sections = [s.strip() for s in custom_sections.split(",")]
        if len(sections) < 3:
            print("   → Need at least 3 sections, using defaults")
            sections = [
                "Introduction",
                "Getting Started",
                "Core Concepts",
                "Best Practices",
                "Real-World Example",
                "Conclusion"
            ]
    else:
        sections = [
            "Introduction",
            "Getting Started",
            "Core Concepts",
            "Best Practices",
            "Real-World Example",
            "Conclusion"
        ]
    
    config = {
        "topic": topic,
        "word_count_target": word_count,
        "style": style,
        "audience": audience,
        "sections": sections
    }
    
    return config

# Get configuration from user
BLOG_CONFIG = get_user_input()

# ============================================================================
# BLOG WRITING TOOLS - CUSTOMIZE THESE
# ============================================================================

def research_blog_topic(topic: str) -> dict:
    """Step 1: Research the topic."""
    return {
        "topic": topic,
        "research_completed": True,
        "key_points": [
            f"What is {topic}?",
            f"Why {topic} matters",
            f"How to get started",
            f"Advanced {topic} techniques",
            f"Common mistakes",
            f"Future of {topic}"
        ],
        "tips": [
            "Keep it practical",
            "Use real examples",
            "Include code samples",
            "Provide resources"
        ]
    }


def create_blog_structure(research: dict, config: dict) -> dict:
    """Step 2: Create outline and structure."""
    topic = research.get("topic", "Topic")
    
    return {
        "title": f"Complete Guide to {topic}",
        "sections": config.get("sections", []),
        "outline": {
            "hooks": ["Start with a compelling question", "Lead with value"],
            "body": ["Use clear subheadings", "Include code/examples"],
            "cta": ["Encourage action", "Provide next steps"]
        },
        "metadata": {
            "target_words": config.get("word_count_target", 2000),
            "reading_time": 10,
            "difficulty": config.get("style", "intermediate")
        }
    }


def write_blog_content(research: dict, structure: dict) -> dict:
    """Step 3: Write the actual blog content."""
    topic = research.get("topic", "Topic")
    sections = structure.get("sections", [])
    
    content = f"# {structure.get('title', f'Guide to {topic}')}\n\n"
    content += f"*Published: {datetime.now().strftime('%B %d, %Y')}*\n\n"
    
    # Introduction
    content += "## Introduction\n\n"
    content += f"In today's digital world, {topic} has become essential. This comprehensive guide "
    content += f"will walk you through everything you need to know about {topic}.\n\n"
    content += "### What You'll Learn\n"
    for i, section in enumerate(sections[1:], 1):
        content += f"- {section}\n"
    content += "\n"
    
    # Main sections
    for section in sections[1:-1]:
        content += f"## {section}\n\n"
        content += f"### Understanding {section}\n\n"
        content += f"{section} is a crucial part of working with {topic}. Here are the key points:\n\n"
        content += "- **Point 1**: Key information\n"
        content += "- **Point 2**: Important concept\n"
        content += "- **Point 3**: Best practice\n\n"
        content += "### Practical Example\n\n"
        content += "```python\n"
        content += "# Example code\n"
        content += f"print('{section} example')\n"
        content += "```\n\n"
    
    # Conclusion
    content += "## Conclusion\n\n"
    content += f"You now have a solid understanding of {topic}. Start with the basics, "
    content += "practice regularly, and gradually advance to more complex techniques.\n\n"
    content += "### Next Steps\n"
    content += "1. Practice with simple examples\n"
    content += "2. Build a small project\n"
    content += "3. Share your learning\n"
    content += "4. Continue exploring\n"
    
    return {
        "content": content,
        "word_count": len(content.split()),
        "sections_written": len(sections)
    }


def optimize_blog_content(written: dict, research: dict) -> dict:
    """Step 4: Optimize for SEO and readability."""
    content = written.get("content", "")
    
    return {
        "optimized_content": content,
        "optimization_report": {
            "seo_score": 85,
            "readability_score": 8.5,
            "keyword_density": "optimal",
            "improvements": [
                "✓ Added meta descriptions",
                "✓ Optimized headings",
                "✓ Added internal links",
                "✓ Improved formatting"
            ]
        },
        "ready_for_publishing": True
    }


def create_blog_metadata(written: dict, structure: dict) -> dict:
    """Step 5: Generate SEO metadata."""
    title = structure.get("title", "Blog Post")
    topic = title.lower()
    
    return {
        "title": title,
        "meta_description": f"A comprehensive guide to {topic}. Learn everything you need to know.",
        "slug": topic.replace(" ", "-"),
        "tags": [
            topic.split()[0].lower(),
            "tutorial",
            "guide",
            "how-to"
        ],
        "seo_keywords": [
            topic,
            f"{topic} guide",
            f"learn {topic}",
            f"{topic} tutorial"
        ],
        "featured_image": f"images/{topic.replace(' ', '-')}.jpg",
        "published_date": datetime.now().isoformat()
    }


# ============================================================================
# POT PLAN FOR BLOG WRITING
# ============================================================================

class SimpleBlogLLM:
    """LLM that plans the blog workflow."""
    
    def complete(self, prompt: str) -> str:
        # Read config from global
        config = BLOG_CONFIG
        topic = config.get("topic", "Topic")
        
        plan = {
            "reasoning": f"Write a comprehensive blog about {topic} by researching, outlining, writing, optimizing, and generating metadata.",
            "confidence": 0.90,
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "research_blog_topic",
                    "args": {"topic": topic},
                    "why": "Gather information about the topic",
                    "depends_on": [],
                    "on_failure": "abort"
                },
                {
                    "step_id": 2,
                    "tool_name": "create_blog_structure",
                    "args": {"research": "$step_1", "config": config},
                    "why": "Create outline and structure",
                    "depends_on": [1],
                    "on_failure": "abort"
                },
                {
                    "step_id": 3,
                    "tool_name": "write_blog_content",
                    "args": {"research": "$step_1", "structure": "$step_2"},
                    "why": "Write the actual blog content",
                    "depends_on": [1, 2],
                    "on_failure": "abort"
                },
                {
                    "step_id": 4,
                    "tool_name": "optimize_blog_content",
                    "args": {"written": "$step_3", "research": "$step_1"},
                    "why": "Optimize for SEO and readability",
                    "depends_on": [3, 1],
                    "on_failure": "skip"
                },
                {
                    "step_id": 5,
                    "tool_name": "create_blog_metadata",
                    "args": {"written": "$step_3", "structure": "$step_2"},
                    "why": "Generate SEO metadata",
                    "depends_on": [3, 2],
                    "on_failure": "skip"
                }
            ]
        }
        return json.dumps(plan)
    
    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Write a blog using PoT."""
    
    config = BLOG_CONFIG
    topic = config.get("topic")
    
    print("\n" + "="*70)
    print(f"📝 SIMPLE BLOG AGENT")
    print(f"Topic: {topic}")
    print("="*70 + "\n")
    
    # Setup registry
    registry = ToolRegistry(include_builtin=False)
    registry.register(Tool("research_blog_topic", research_blog_topic, "Research topic"))
    registry.register(Tool("create_blog_structure", create_blog_structure, "Create structure"))
    registry.register(Tool("write_blog_content", write_blog_content, "Write content"))
    registry.register(Tool("optimize_blog_content", optimize_blog_content, "Optimize content"))
    registry.register(Tool("create_blog_metadata", create_blog_metadata, "Create metadata"))
    
    # Plan
    print("📋 Planning...")
    llm = SimpleBlogLLM()
    planner = ThoughtPlanner(llm, registry)
    
    program = planner.plan(
        goal=f"Write a blog about {topic}",
        session_id=f"blog-{int(datetime.now().timestamp())}"
    )
    
    print(f"✅ Plan: {len(program.steps)} steps\n")
    
    # Execute
    print("🔄 Executing...")
    executor = ThoughtExecutor(registry, max_retries=1)
    result = await executor.execute(program)
    
    # Display results
    print("\n" + "="*70)
    print("✅ BLOG CREATED!")
    print("="*70)
    
    if isinstance(result, str):
        # Show preview
        lines = result.split('\n')[:30]
        print('\n'.join(lines))
        print("\n[... content continues ...]\n")
        print(f"Total: {len(result)} characters")
    
    print("\n" + "="*70)
    print("💡 CUSTOMIZATION TIPS")
    print("="*70)
    print("""
To modify this for your blog:

1. Change BLOG_CONFIG at the top:
   - topic: Your blog topic
   - word_count_target: Desired length
   - style: 'technical', 'casual', 'professional'
   - audience: 'developers', 'business', 'general'
   - sections: Your custom sections

2. Modify the tools:
   - research_blog_topic: Add real research API
   - write_blog_content: Add custom writing logic
   - optimize_blog_content: Add real optimization

3. Add more steps:
   - Add more tools to registry
   - Add to plan in SimpleBlogLLM.complete()
   - Use $step_N to chain results

4. Connect to publishing:
   - Add tool to post to WordPress
   - Add tool to share on social
   - Add tool to send notifications

Example modification:
""")
    
    print("""
# Add this tool:
def publish_to_wordpress(content: dict) -> dict:
    \"\"\"Publish to WordPress.\"\"\"
    import wordpress_api
    post_id = wordpress_api.create_post(
        title=content.get('title'),
        content=content.get('content'),
        tags=content.get('tags')
    )
    return {"post_id": post_id, "status": "published"}

# Register it:
registry.register(Tool("publish", publish_to_wordpress, "Publish post"))

# Add to plan:
{
    "step_id": 6,
    "tool_name": "publish",
    "args": {"content": "$step_5"},
    "depends_on": [5],
    "on_failure": "skip"
}
""")


if __name__ == "__main__":
    asyncio.run(main())
