from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_prompt_template(additional_instruction):
    """
    Returns a structured prompt template for extracting course information.
    """
    template = """
    ### CONTEXT:
    {page_content}

    ### INSTRUCTION:
    Analyze the provided context, objectives, and constraints, then {additional_instruction}.
    Organize results in JSON format with the following structure:
    - `courses`: List of relevant courses (e.g.,Computer Science and Engineering, Data Science, AI etc.)
    
    Return only a JSON array with the identified courses, excluding unnecessary information show only course name. exclude all preamble or extraneous information.
    """
    return PromptTemplate.from_template(template)

def extract_course_data(model, prompt_template, context, instruction):
    """
    Extracts course data using the LLM and provided prompt template.
    """
    formatted_prompt = prompt_template.format(
        page_content=context, additional_instruction=instruction
    )
    response = model.invoke(formatted_prompt)

    # Parse JSON response
    json_parser = JsonOutputParser()
    try:
        return json_parser.parse(response.content)
    except ValueError:
        return None
