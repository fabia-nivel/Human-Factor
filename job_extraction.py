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
    - `courses`: List of relevant courses (e.g., CSE etc.)
    - `reason`:  Builds on AI knowledge, highly rated instructor, balanced assessments, suitable workload (2 days/week, 3 credits), affordable ($1,800), no prerequisites, and open add/drop policy. Schedule does not overlap with other courses. .
    
    
    Return only a JSON array with the identified courses, including necessary information show only course name. .
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
