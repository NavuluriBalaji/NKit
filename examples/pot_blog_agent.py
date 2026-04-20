"""Real-World Example: Blog Writing Agent with PoT.

A complete example of using Program of Thought for content creation.

Workflow:
1. Research topic (search for info)
2. Create outline (structure the content)
3. Write introduction (hook readers)
4. Write main sections (body content)
5. Write conclusion (summary and CTA)
6. Edit & polish (grammar, flow)
7. Format for publishing (markdown)
8. Generate metadata (title, description, tags)

This demonstrates:
- Sequential workflows
- Result chaining between steps
- Real content generation
- Production-ready blog output
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add NKit to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from NKit.tools import ToolRegistry, Tool
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor
from NKit.observer import LiveObserver


# ============================================================================
# Blog Writing Tools
# ============================================================================

def research_topic(topic: str, depth: str = "medium"):
    """Research a topic to gather information."""
    # Mock research data
    research_data = {
        "topic": topic,
        "depth": depth,
        "key_points": [
            f"Introduction to {topic}",
            f"Core principles of {topic}",
            f"Best practices for {topic}",
            f"Common challenges with {topic}",
            f"Future trends in {topic}",
            f"Real-world applications of {topic}"
        ],
        "sources": [
            "Industry report 2026",
            "Expert interviews",
            "Case studies",
            "Academic papers"
        ],
        "data_points": {
            "market_size": "$50B+",
            "adoption_rate": "78%",
            "growth_rate": "25% YoY"
        }
    }
    return research_data


def create_outline(research: dict, style: str = "professional"):
    """Create blog outline from research."""
    topic = research.get("topic", "Topic")
    points = research.get("key_points", [])
    
    outline = {
        "title": f"Complete Guide to {topic}: Everything You Need to Know",
        "sections": [
            {
                "num": 1,
                "title": "Introduction",
                "subsections": ["Hook", "What is it?", "Why it matters"],
                "key_points": [points[0] if points else "Overview"]
            },
            {
                "num": 2,
                "title": "Fundamentals",
                "subsections": ["Core concepts", "Key principles", "How it works"],
                "key_points": points[1:3] if len(points) > 2 else points
            },
            {
                "num": 3,
                "title": "Best Practices",
                "subsections": ["Getting started", "Advanced techniques", "Common pitfalls"],
                "key_points": [points[2] if len(points) > 2 else "Best practices"]
            },
            {
                "num": 4,
                "title": "Real-World Applications",
                "subsections": ["Industry examples", "Case studies", "Success stories"],
                "key_points": [points[5] if len(points) > 5 else "Applications"]
            },
            {
                "num": 5,
                "title": "Challenges & Solutions",
                "subsections": ["Common issues", "How to overcome them", "Pro tips"],
                "key_points": [points[3] if len(points) > 3 else "Challenges"]
            },
            {
                "num": 6,
                "title": "Future Outlook",
                "subsections": ["Upcoming trends", "What's next?", "Predictions"],
                "key_points": [points[4] if len(points) > 4 else "Trends"]
            },
            {
                "num": 7,
                "title": "Conclusion",
                "subsections": ["Key takeaways", "Call to action"],
                "key_points": ["Summary and next steps"]
            }
        ],
        "style": style,
        "target_audience": "Technical professionals",
        "estimated_word_count": 2500
    }
    return outline


def write_introduction(research: dict, outline: dict):
    """Write engaging introduction."""
    topic = research.get("topic", "Topic")
    hook_data = research.get("data_points", {})
    market_size = hook_data.get("market_size", "billions")
    
    intro = f"""# {outline.get('title', f'Guide to {topic}')}

## Introduction

The {topic} industry is experiencing unprecedented growth, with a market size reaching {market_size} and a year-over-year growth rate of 25%. But what does this mean for you?

In this comprehensive guide, we'll explore everything you need to know about {topic}. Whether you're a newcomer looking to understand the basics or an experienced professional seeking to deepen your expertise, this article will provide valuable insights.

### What We'll Cover

This guide is structured to take you from zero to expert:
- **Fundamentals**: Understanding core concepts
- **Best Practices**: Industry-proven strategies
- **Real-world Applications**: How companies are using {topic}
- **Challenges & Solutions**: Overcoming common obstacles
- **Future Outlook**: Where {topic} is headed

By the end of this article, you'll have a complete understanding of {topic} and be ready to implement these insights in your own projects.

---
"""
    return {"section": "Introduction", "content": intro}


def write_main_sections(research: dict, outline: dict, intro: dict):
    """Write main content sections."""
    topic = research.get("topic", "Topic")
    sections = []
    
    # Fundamentals
    fundamentals = f"""## 1. Fundamentals of {topic}

### Core Concepts

{topic} is built on several foundational principles that are essential to understand:

1. **First Principle**: At its core, {topic} relies on understanding...
2. **Second Principle**: The ecosystem requires...
3. **Third Principle**: Success depends on...

### How It Works

The mechanism behind {topic} involves:

- **Research Phase**: Understanding requirements
- **Planning Phase**: Structuring the approach
- **Execution Phase**: Implementing the solution
- **Optimization Phase**: Refining the results

### Why It Matters

In today's digital landscape, {topic} has become critical because:
- Improves efficiency by 40%
- Reduces costs significantly
- Enables better decision-making
- Supports scalability

---
"""
    sections.append(fundamentals)
    
    # Best Practices
    best_practices = f"""## 2. Best Practices for {topic}

### Getting Started

If you're new to {topic}, follow these steps:

**Step 1**: Understand the fundamentals (✓ you're reading this!)
**Step 2**: Set up your infrastructure
**Step 3**: Start with a small pilot project
**Step 4**: Measure and iterate
**Step 5**: Scale to production

### Advanced Techniques

For experienced professionals:

- Implement automated workflows
- Use advanced analytics
- Leverage machine learning
- Build custom solutions
- Create feedback loops

### Common Pitfalls to Avoid

1. **Skipping the research phase**: Always understand requirements first
2. **Over-complicating the solution**: Start simple, add complexity as needed
3. **Ignoring monitoring**: Track performance continuously
4. **Failing to document**: Keep detailed records for future reference

---
"""
    sections.append(best_practices)
    
    # Real-world Applications
    applications = f"""## 3. Real-World Applications

### Case Study 1: Enterprise Implementation

**Company**: Tech Corp (Fortune 500)
**Challenge**: Managing complex workflows
**Solution**: Implemented {topic}
**Results**: 
- 50% faster processing
- $5M annual savings
- Improved employee satisfaction

### Case Study 2: Startup Success

**Company**: Growth Startup (Series B)
**Challenge**: Limited resources, high demand
**Solution**: Leveraged {topic} for automation
**Results**:
- Scaled to 10x without hiring
- Improved customer satisfaction
- Reduced operational costs

### Industry Examples

{topic} is being used across industries:

- **Finance**: Risk assessment and fraud detection
- **Healthcare**: Patient data management and analysis
- **E-commerce**: Customer personalization
- **Manufacturing**: Supply chain optimization
- **Energy**: Predictive maintenance

---
"""
    sections.append(applications)
    
    # Challenges & Solutions
    challenges = f"""## 4. Challenges & Solutions

### Common Challenges

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Integration complexity | High | Use APIs and middleware |
| Data quality | Critical | Implement validation |
| Performance issues | Medium | Optimize queries |
| Security concerns | Critical | Follow best practices |
| Cost management | High | Monitor and optimize |

### How to Overcome Obstacles

**Challenge 1: Learning curve**
- **Solution**: Start with official documentation and tutorials
- **Time investment**: 2-4 weeks for fundamentals

**Challenge 2: Implementation complexity**
- **Solution**: Break into smaller phases
- **Time investment**: 3-6 months for full rollout

**Challenge 3: Team training**
- **Solution**: Invest in professional development
- **Time investment**: Ongoing

### Pro Tips from Industry Experts

1. **Start small and iterate**: Don't try to do everything at once
2. **Invest in the right tools**: Quality tools save time and money
3. **Build a strong team**: People are your greatest asset
4. **Document everything**: Future you will be grateful
5. **Stay updated**: The field evolves rapidly

---
"""
    sections.append(challenges)
    
    # Future Outlook
    future = f"""## 5. Future of {topic}

### Emerging Trends

The {topic} landscape is evolving rapidly. Key trends to watch:

1. **AI Integration**: Machine learning is becoming standard
2. **Edge Computing**: Processing data closer to the source
3. **Automation**: More processes becoming automated
4. **Interoperability**: Better integration between systems
5. **Sustainability**: Greener approaches gaining traction

### What's Next?

**2026-2027**: Consolidation of tools and platforms
**2027-2028**: Mainstream adoption across industries
**2028+**: Maturation and standardization

### Predictions

Industry analysts predict:
- Market will reach $100B+ by 2028
- 80%+ enterprise adoption
- New job market worth $50B+
- Significant innovation in AI integration

---
"""
    sections.append(future)
    
    return {"sections": "\n".join(sections), "word_count": 2000}


def write_conclusion(research: dict, outline: dict, main_sections: dict):
    """Write compelling conclusion."""
    topic = research.get("topic", "Topic")
    
    conclusion = f"""## 6. Conclusion

### Key Takeaways

Let's recap what we've covered:

1. **{topic} fundamentals** are essential to understand
2. **Best practices** can significantly improve your results
3. **Real-world applications** show the value across industries
4. **Challenges are manageable** with the right approach
5. **The future is bright** for this technology

### Why This Matters to You

Whether you're looking to:
- **Improve efficiency**: {topic} can help you work smarter
- **Reduce costs**: Significant savings are possible
- **Stay competitive**: Early adoption provides advantages
- **Build skills**: This is a high-demand capability

Now is the perfect time to dive in.

### Your Next Steps

1. **Learn**: Study the fundamentals and best practices
2. **Experiment**: Start a small proof-of-concept
3. **Build**: Implement {topic} in your organization
4. **Share**: Document and share your success
5. **Grow**: Scale your implementation

### Call to Action

Ready to get started with {topic}? Here's what we recommend:

**For Beginners**: 
- Take the free online course
- Join the community forum
- Start with a simple project

**For Professionals**:
- Get certified
- Explore advanced topics
- Build enterprise solutions

**For Organizations**:
- Assess your readiness
- Plan your implementation
- Invest in team training

### Final Thoughts

{topic} is more than just a technology—it's a paradigm shift in how we approach problems. By implementing the strategies and insights from this guide, you'll be well-positioned to succeed in the {topic} revolution.

The future is being built now. Don't get left behind.

---

*Last updated: {datetime.now().strftime('%B %d, %Y')}*
*Word count: 2,500+*
*Reading time: ~10 minutes*
"""
    return {"section": "Conclusion", "content": conclusion}


def edit_and_polish(intro: dict, main: dict, conclusion: dict):
    """Edit for grammar, flow, and readability."""
    full_content = intro.get("content", "") + main.get("sections", "") + conclusion.get("content", "")
    
    editing_report = {
        "grammar_check": "✓ Passed",
        "flow_check": "✓ Improved",
        "readability_score": 8.5,
        "reading_level": "Professional (Grade 12+)",
        "improvements_made": [
            "Enhanced transitions between sections",
            "Improved sentence structure",
            "Added subheadings for clarity",
            "Balanced paragraph lengths",
            "Enhanced consistency"
        ],
        "total_words": 2500,
        "estimated_read_time": "10 minutes"
    }
    
    return {"report": editing_report, "content": full_content}


def format_for_publishing(edited: dict, outline: dict):
    """Format content for publishing (Markdown)."""
    content = edited.get("content", "")
    
    formatted = {
        "format": "Markdown",
        "structure": {
            "frontmatter": {
                "type": "YAML",
                "fields": ["title", "author", "date", "tags", "category"]
            },
            "body": "Standard Markdown",
            "metadata": {
                "word_count": 2500,
                "reading_time": "10 min",
                "difficulty": "Intermediate"
            }
        },
        "output": content,
        "ready_for_publishing": True
    }
    
    return formatted


def generate_metadata(research: dict, outline: dict):
    """Generate SEO metadata and publishing details."""
    topic = research.get("topic", "Topic")
    
    metadata = {
        "title": outline.get("title", f"Guide to {topic}"),
        "slug": topic.lower().replace(" ", "-"),
        "excerpt": f"A comprehensive guide to {topic}. Learn the fundamentals, best practices, real-world applications, and future trends. Perfect for professionals looking to master {topic}.",
        "seo_description": f"Complete guide to {topic} - Learn fundamentals, best practices, real-world applications, and industry trends. 2500+ word guide for professionals.",
        "tags": [
            topic.lower(),
            "technology",
            "guide",
            "tutorial",
            "best-practices",
            "professional-development"
        ],
        "category": "Technology",
        "author": "NKit Blog Agent",
        "published_date": datetime.now().isoformat(),
        "word_count": 2500,
        "reading_time_minutes": 10,
        "featured_image": f"/images/{topic.lower().replace(' ', '-')}-featured.jpg",
        "social_share": {
            "title": f"Master {topic}: Complete Guide & Best Practices",
            "description": "A comprehensive guide covering fundamentals, best practices, and real-world applications"
        },
        "seo_keywords": [
            topic,
            f"{topic} guide",
            f"{topic} best practices",
            f"how to learn {topic}",
            f"{topic} for beginners"
        ]
    }
    
    return metadata


# ============================================================================
# Blog Agent with PoT
# ============================================================================

def create_blog_llm(topic: str):
    """Create LLM that plans the blog writing workflow."""
    
    class BlogLLM:
        def complete(self, prompt: str) -> str:
            plan = {
                "reasoning": f"To write a comprehensive blog about '{topic}', I'll follow a structured approach: research the topic, create an outline, write each section, edit for quality, format for publishing, and generate metadata.",
                "confidence": 0.93,
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "research_topic",
                        "args": {"topic": topic, "depth": "deep"},
                        "why": "Gather comprehensive information and data about the topic",
                        "depends_on": [],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 2,
                        "tool_name": "create_outline",
                        "args": {"research": "$step_1", "style": "professional"},
                        "why": "Structure the blog with clear sections and flow",
                        "depends_on": [1],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 3,
                        "tool_name": "write_introduction",
                        "args": {"research": "$step_1", "outline": "$step_2"},
                        "why": "Create an engaging introduction that hooks readers",
                        "depends_on": [1, 2],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 4,
                        "tool_name": "write_main_sections",
                        "args": {"research": "$step_1", "outline": "$step_2", "intro": "$step_3"},
                        "why": "Write the main body content with detailed information",
                        "depends_on": [1, 2, 3],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 5,
                        "tool_name": "write_conclusion",
                        "args": {"research": "$step_1", "outline": "$step_2", "main_sections": "$step_4"},
                        "why": "Create a compelling conclusion with call-to-action",
                        "depends_on": [1, 2, 4],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 6,
                        "tool_name": "edit_and_polish",
                        "args": {"intro": "$step_3", "main": "$step_4", "conclusion": "$step_5"},
                        "why": "Improve grammar, flow, and overall readability",
                        "depends_on": [3, 4, 5],
                        "on_failure": "skip"
                    },
                    {
                        "step_id": 7,
                        "tool_name": "format_for_publishing",
                        "args": {"edited": "$step_6", "outline": "$step_2"},
                        "why": "Format content as Markdown for publishing",
                        "depends_on": [6, 2],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 8,
                        "tool_name": "generate_metadata",
                        "args": {"research": "$step_1", "outline": "$step_2"},
                        "why": "Generate SEO metadata and publishing details",
                        "depends_on": [1, 2],
                        "on_failure": "skip"
                    }
                ]
            }
            return json.dumps(plan)
        
        def __call__(self, prompt: str) -> str:
            return self.complete(prompt)
    
    return BlogLLM()


def setup_observer():
    """Setup observer with blog-specific formatting."""
    observer = LiveObserver()
    
    @observer.on("agent.start")
    def on_start(event):
        print("\n" + "="*70)
        print(f"📝 BLOG WRITING AGENT STARTED")
        print(f"Total steps: {event['total_steps']}")
        print("="*70 + "\n")
    
    @observer.on("tool.before")
    def on_before(event):
        step_map = {
            1: "Research",
            2: "Outline",
            3: "Introduction",
            4: "Main Sections",
            5: "Conclusion",
            6: "Editing",
            7: "Formatting",
            8: "Metadata"
        }
        step_name = step_map.get(event['step_id'], event['tool_name'])
        print(f"Step {event['step_id']}: {step_name}...")
    
    @observer.on("tool.after")
    def on_after(event):
        if event.get("success"):
            print(f"  ✅ Complete\n")
        else:
            print(f"  ❌ Failed\n")
    
    @observer.on("agent.end")
    def on_end(event):
        print("="*70)
        print("✅ BLOG COMPLETE")
        print("="*70 + "\n")
    
    return observer


# ============================================================================
# Main Execution
# ============================================================================

async def write_blog(topic: str):
    """Write a complete blog post using PoT."""
    print("\n" + "█"*70)
    print("█" + " BLOG WRITING AGENT - Program of Thought".center(68) + "█")
    print("█"*70)
    print(f"\n📚 Writing blog about: {topic}\n")
    
    # Setup
    registry = ToolRegistry(include_builtin=False)
    
    # Register all tools
    tools = [
        Tool("research_topic", research_topic, "Research a topic"),
        Tool("create_outline", create_outline, "Create blog outline"),
        Tool("write_introduction", write_introduction, "Write introduction"),
        Tool("write_main_sections", write_main_sections, "Write main sections"),
        Tool("write_conclusion", write_conclusion, "Write conclusion"),
        Tool("edit_and_polish", edit_and_polish, "Edit and polish"),
        Tool("format_for_publishing", format_for_publishing, "Format for publishing"),
        Tool("generate_metadata", generate_metadata, "Generate metadata"),
    ]
    
    for tool in tools:
        registry.register(tool)
    
    # Plan and execute
    llm = create_blog_llm(topic)
    observer = setup_observer()
    
    planner = ThoughtPlanner(llm, registry)
    executor = ThoughtExecutor(registry, observer=observer, max_retries=1)
    
    try:
        program = planner.plan(
            goal=f"Write a comprehensive, SEO-optimized blog post about {topic}",
            session_id=f"blog-{datetime.now().timestamp()}"
        )
        
        print(f"✅ Plan created: {len(program.steps)} steps\n")
        
        result = await executor.execute(program)
        
        # Extract and display results
        if isinstance(result, str):
            print("\n" + "="*70)
            print("📄 BLOG POST PREVIEW")
            print("="*70)
            print(result[:1500] + "\n[... truncated ...]\n")
            print(f"Total length: {len(result)} characters")
        else:
            print(f"\nResult: {str(result)[:200]}...")
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Examples
# ============================================================================

def main():
    """Run blog writing examples."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  Real-World: Blog Writing Agent with PoT".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("""
This example demonstrates Program of Thought for content creation.

The blog agent:
1. Research the topic (gather information)
2. Create an outline (structure the post)
3. Write introduction (engage readers)
4. Write main sections (deliver value)
5. Write conclusion (summarize & CTA)
6. Edit & polish (improve quality)
7. Format for publishing (markdown)
8. Generate metadata (SEO & sharing)

All done with ONE LLM call at the planning stage!
""")
    
    print("\n" + "="*70)
    print("Starting blog writing process...")
    print("="*70)
    
    # Choose a topic
    topics = [
        "Artificial Intelligence in Business",
        "Cloud Computing Best Practices",
        "DevOps for Modern Teams"
    ]
    
    topic = topics[0]
    print(f"\nSelected topic: {topic}\n")
    
    # Write the blog
    result = asyncio.run(write_blog(topic))
    
    if result:
        print("\n" + "="*70)
        print("✨ BLOG WRITING COMPLETE")
        print("="*70)
        print(f"""
Your blog post is ready for publishing!

Next steps:
1. Review the content in your editor
2. Add featured image
3. Review SEO metadata
4. Schedule publication
5. Share on social media

Features of this PoT approach:
✅ Single LLM call (no repeated thinking)
✅ 8-step workflow executed deterministically
✅ Full content generation
✅ SEO metadata included
✅ Publication-ready format
✅ Reproducible (same input = same output)
""")


if __name__ == "__main__":
    main()
