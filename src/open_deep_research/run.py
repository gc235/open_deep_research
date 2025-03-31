from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.open_deep_research.graph import builder

import asyncio
import uuid

from IPython.core.display import Markdown, Image
from IPython.core.display_functions import display

from src.open_deep_research.configuration import SearchAPI

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

REPORT_STRUCTURE = """使用此结构为用户提供的主题创建报告：

1. 引言（无需研究）
- 主题领域的简要概述

2. 主体部分：
每个部分应聚焦于用户提供主题下的子主题

3. 结论
力求包含 1 个提炼正文部分主要内容的组织结构元素（列表或表格）
提供报告的简要总结"""

# Claude 3.7 Sonnet for planning with perplexity search
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           }}



topic = "2024年笔记本电脑消费情况"


async def process_stream():
    # Run the graph until the interruption
    async for event in graph.astream({"topic":"2024年笔记本电脑消费情况",}, thread, stream_mode="updates"):
        if '__interrupt__' in event:
            interrupt_value = event['__interrupt__'][0].value
            display(Markdown(interrupt_value))

    async for event in graph.astream(
            Command(resume="包括苹果、联想、惠普、华硕、戴尔等品牌的笔记本电脑"),
            thread, stream_mode="updates"):
        if '__interrupt__' in event:
            interrupt_value = event['__interrupt__'][0].value
            display(Markdown(interrupt_value))

    async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
        print(event)
        print("\n")


asyncio.run(process_stream())