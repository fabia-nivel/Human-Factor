from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_prompt_template(additional_instruction):
    template = """
    ### CONTEXT:
    {page_content}

    ### INSTRUCTION:
    Carefully analyze the provided context, objectives, and constraints. Then, {additional_instruction}.
    Organize the results in JSON format with the following structure:
    - `courses`: List of courses to be selected to fulfill the objectives.
      Each course must be in the format "course code - course name".
      Only include courses explicitly mentioned in the context.

    Return only the JSON array with the identified courses. Exclude any preambles, explanations, or extraneous information.
    """
    return PromptTemplate.from_template(template)

def extract_course_data(model, prompt_template, context, instruction):
    formatted_prompt = prompt_template.format(
        page_content=context, additional_instruction=instruction
    )
    response = model.invoke(formatted_prompt)
    json_parser = JsonOutputParser()
    try:
        return json_parser.parse(response.content)
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return None
