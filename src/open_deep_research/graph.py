import os
import logging
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from phoenix.otel import register

from configuration import Configuration
from prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)
from state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)
from utils import (
    format_sections,
    get_config_value,
    get_search_params,
    select_and_execute_search
)

print(os.getenv('PHOENIX_ENDPOINT'))
tracer_provider = register(
    project_name=os.getenv('PHOENIX_PROJECT_NAME'),
    endpoint=os.getenv('PHOENIX_ENDPOINT'),
    auto_instrument=True
)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(log_dir, 'open_deep_research.log'),
                    filemode='a')
# 'a' for append mode


## Nodes --

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    logging.debug("Starting generate_report_plan with state: %s and config: %s", state, config)

    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)
    logging.debug("Extracted topic: %s and feedback: %s", topic, feedback)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    logging.debug("Loaded configuration: %s", configurable)

    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    logging.debug("Report structure: %s, Number of queries: %d, Search API: %s, Search API Config: %s",
                  report_structure, number_of_queries, search_api, search_api_config)

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)
        logging.debug("Converted report_structure to string: %s", report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider,
                                   base_url=configurable.base_url,
                                   )
    structured_llm = writer_model.with_structured_output(Queries)
    logging.debug("Initialized writer model: %s with provider: %s", writer_model_name, writer_provider)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic,
                                                                                report_organization=report_structure,
                                                                                number_of_queries=number_of_queries)
    logging.debug("Formatted system instructions for query generation: %s", system_instructions_query)

    # Generate queries
    query_generation_message = """Generate search queries that will help with planning the sections of the report. """

    results = structured_llm.invoke([
        SystemMessage(content=system_instructions_query),
        SystemMessage(content=f'输出结果必须是JSON格式, pydantic json schema定义如下：\n {Queries.model_json_schema()}'),
        HumanMessage(content=query_generation_message)
    ])
    logging.debug("Generated queries: %s", results)

    # Web search
    query_list = [query.search_query for query in results.queries]
    logging.debug("Query list for web search: %s", query_list)

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
    logging.debug("Web search results: %s", source_str)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure,
                                                                      context=source_str, feedback=feedback)
    logging.debug("Formatted system instructions for section generation: %s", system_instructions_sections)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    logging.debug("Planner model: %s with provider: %s", planner_model, planner_provider)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""
    logging.debug("Planner message: %s", planner_message)

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model,
                                      model_provider=planner_provider,
                                      max_tokens=20_000,
                                      thinking={"type": "enabled", "budget_tokens": 16_000},
                                      base_url=configurable.base_url)
        logging.debug("Initialized planner model with thinking budget: %s", planner_model)
    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model,
                                      model_provider=planner_provider, base_url=configurable.base_url)
        logging.debug("Initialized planner model without thinking budget: %s", planner_model)

    # Generate the report sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                             SystemMessage(
                                                 content=f'输出结果必须是JSON格式, pydantic json schema定义如下：\n {Sections.model_json_schema()}'),
                                             HumanMessage(content=planner_message)])
    logging.debug("Generated report sections: %s", report_sections)

    # Get sections
    sections = report_sections.sections
    logging.debug("Extracted sections: %s", sections)

    return {"sections": sections}


def human_feedback(state: ReportState, config: RunnableConfig) -> Command[
    Literal["generate_report_plan", "build_section_with_web_research"]]:
    logging.debug("Starting human_feedback with state: %s and config: %s", state, config)

    topic = state["topic"]
    sections = state['sections']
    logging.debug("Extracted topic: %s and sections: %s", topic, sections)

    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )
    logging.debug("Formatted sections string for feedback: %s", sections_str)

    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    logging.debug("Generated interrupt message: %s", interrupt_message)

    feedback = interrupt(interrupt_message)
    logging.debug("Received feedback: %s", feedback)

    if isinstance(feedback, bool) and feedback is True:
        logging.debug("Feedback approved, proceeding to section writing.")
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0})
            for s in sections
            if s.research
        ])

    elif isinstance(feedback, str):
        logging.debug("Feedback provided for regeneration: %s", feedback)
        return Command(goto="generate_report_plan",
                       update={"feedback_on_report_plan": feedback})
    else:
        logging.error("Unsupported feedback type: %s", type(feedback))
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


def generate_queries(state: SectionState, config: RunnableConfig):
    logging.debug("Starting generate_queries with state: %s and config: %s", state, config)

    topic = state["topic"]
    section = state["section"]
    logging.debug("Extracted topic: %s and section: %s", topic, section)

    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries
    logging.debug("Loaded configuration: %s", configurable)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    logging.debug("Writer provider: %s, Writer model name: %s", writer_provider, writer_model_name)

    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider,
                                   base_url=configurable.base_url)
    structured_llm = writer_model.with_structured_output(Queries)
    logging.debug("Initialized writer model: %s", writer_model)

    system_instructions = query_writer_instructions.format(topic=topic,
                                                           section_topic=section.description,
                                                           number_of_queries=number_of_queries)
    logging.debug("Formatted system instructions for query generation: %s", system_instructions)

    query_generation_message = """Generate search queries on the provided topic."""

    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     SystemMessage(
                                         content=f'输出结果必须是JSON格式, pydantic json schema定义如下：\n {Queries.model_json_schema()}'),
                                     HumanMessage(content=query_generation_message)])
    logging.debug("Generated queries: %s", queries)

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    logging.debug("Starting search_web with state: %s and config: %s", state, config)

    search_queries = state["search_queries"]
    logging.debug("Extracted search queries: %s", search_queries)

    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)
    logging.debug("Search API: %s, Search API Config: %s, Params to pass: %s", search_api, search_api_config,
                  params_to_pass)

    query_list = [query.search_query for query in search_queries]
    logging.debug("Query list for web search: %s", query_list)

    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
    logging.debug("Web search results: %s", source_str)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    logging.debug("Starting write_section with state: %s and config: %s", state, config)

    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    logging.debug("Extracted topic: %s, section: %s, source_str: %s", topic, section, source_str)

    configurable = Configuration.from_runnable_config(config)
    logging.debug("Loaded configuration: %s", configurable)

    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic,
                                                                   section_name=section.name,
                                                                   section_topic=section.description,
                                                                   context=source_str,
                                                                   section_content=section.content)
    logging.debug("Formatted section writer inputs: %s", section_writer_inputs_formatted)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    logging.debug("Writer provider: %s, Writer model name: %s", writer_provider, writer_model_name)

    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider,
                                   base_url=configurable.base_url)

    section_content = writer_model.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    logging.debug("Generated section content: %s", section_content)

    section.content = section_content.content

    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    logging.debug("Section grader message: %s", section_grader_message)

    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic,
                                                                               section_topic=section.description,
                                                                               section=section.content,
                                                                               number_of_follow_up_queries=configurable.number_of_queries)
    logging.debug("Formatted section grader instructions: %s", section_grader_instructions_formatted)

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    logging.debug("Planner provider: %s, Planner model: %s", planner_provider, planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        reflection_model = init_chat_model(model=planner_model,
                                           model_provider=planner_provider,
                                           max_tokens=20_000,
                                           thinking={"type": "enabled", "budget_tokens": 16_000},
                                           base_url=configurable.base_url).with_structured_output(Feedback)
        logging.debug("Initialized reflection model with thinking budget: %s", planner_model)
    else:
        reflection_model = init_chat_model(model=planner_model,
                                           model_provider=planner_provider,
                                           base_url=configurable.base_url).with_structured_output(Feedback)
        logging.debug("Initialized reflection model without thinking budget: %s", planner_model)

    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        SystemMessage(
                                            content=f'输出结果必须是JSON格式, pydantic json schema定义如下：\n {Feedback.model_json_schema()}'),
                                        HumanMessage(content=section_grader_message)])
    logging.debug("Generated feedback: %s", feedback)

    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        logging.debug("Section passed or max search depth reached, completing section.")
        return Command(
            update={"completed_sections": [section]},
            goto=END
        )
    else:
        logging.debug("Section failed, updating with follow-up queries: %s", feedback.follow_up_queries)
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )


def write_final_sections(state: SectionState, config: RunnableConfig):
    logging.debug("Starting write_final_sections with state: %s and config: %s", state, config)

    configurable = Configuration.from_runnable_config(config)
    logging.debug("Loaded configuration: %s", configurable)

    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    logging.debug("Extracted topic: %s, section: %s, completed_report_sections: %s", topic, section,
                  completed_report_sections)

    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name,
                                                                   section_topic=section.description,
                                                                   context=completed_report_sections)
    logging.debug("Formatted system instructions for final section writing: %s", system_instructions)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    logging.debug("Writer provider: %s, Writer model name: %s", writer_provider, writer_model_name)

    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider,
                                   base_url=configurable.base_url)

    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(
                                               content="Generate a report section based on the provided sources.")])
    logging.debug("Generated final section content: %s", section_content)

    section.content = section_content.content

    return {"completed_sections": [section]}


def gather_completed_sections(state: ReportState):
    logging.debug("Starting gather_completed_sections with state: %s", state)

    completed_sections = state["completed_sections"]
    logging.debug("Extracted completed sections: %s", completed_sections)

    completed_report_sections = format_sections(completed_sections)
    logging.debug("Formatted completed report sections: %s", completed_report_sections)

    return {"report_sections_from_research": completed_report_sections}


def compile_final_report(state: ReportState):
    logging.debug("Starting compile_final_report with state: %s", state)

    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    logging.debug("Extracted sections: %s, completed_sections: %s", sections, completed_sections)

    for section in sections:
        section.content = completed_sections[section.name]
    logging.debug("Updated sections with completed content: %s", sections)

    all_sections = "\n\n".join([s.content for s in sections])
    logging.debug("Compiled final report: %s", all_sections)

    return {"final_report": all_sections}


def initiate_final_section_writing(state: ReportState):
    logging.debug("Starting initiate_final_section_writing with state: %s", state)

    commands = [
        Send("write_final_sections", {"topic": state["topic"], "section": s,
                                      "report_sections_from_research": state["report_sections_from_research"]})
        for s in state["sections"]
        if not s.research
    ]
    logging.debug("Generated commands for final section writing: %s", commands)

    return commands


# Report section sub-graph --

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
