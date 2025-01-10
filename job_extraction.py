import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_prompt_template(additional_instruction):
    """
    Returns a structured prompt template for extracting CSE-related courses.
    """
    template = """
    ### CONTEXT:
    {page_content}

    ### INSTRUCTION:
    Analyze the provided context, objectives, and constraints, then {additional_instruction}.
    Organize results in valid JSON format with the following structure:
    {
        "courses": [
            {
                "name": "Course Name",
                "reason": "Reason for selection"
            },
            ...
        ]
    }

    Strictly output only the JSON object. Do not include any preamble, commentary, or text outside the JSON object.
    """
    return PromptTemplate.from_template(template)

def preprocess_response(response_text):
    """
    Extract JSON content from the response, ignoring any extraneous text.
    """
    try:
        # Attempt to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: Extract JSON substring from the response
        start_index = response_text.find("{")
        end_index = response_text.rfind("}") + 1
        if start_index != -1 and end_index != -1:
            json_part = response_text[start_index:end_index]
            try:
                return json.loads(json_part)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON substring: {e}")
        print("Failed to extract valid JSON.")
        return None

def extract_cse_course_data(model, context, instruction):
    """
    Extracts CSE-related course data using the LLM and provided prompt template.
    """
    # Get the prompt template
    prompt_template = get_prompt_template(instruction)

    # Format the prompt with the given context and instruction
    formatted_prompt = prompt_template.format(
        page_content=context, additional_instruction=instruction
    )
    
    # Invoke the model with the formatted prompt
    response = model.invoke(formatted_prompt)  # Replace with the actual model invocation logic

    # Process the response to extract valid JSON
    response_text = response.content  # Replace with the actual response content
    parsed_response = preprocess_response(response_text)
    return parsed_response

# Example usage
if __name__ == "__main__":
    # Assuming 'model' is an initialized LLM model (e.g., OpenAI's GPT via LangChain)
    model = None  # Replace with the actual model object

    context = """
    This university offers various courses in engineering and sciences, including:
    - Computer Science and Engineering (CSE): Covers foundational and advanced concepts in computing.
    - Data Science (DS): Applies machine learning and statistical methods to analyze data.
    - Artificial Intelligence (AI): Focuses on creating intelligent systems and algorithms.
    """
    
    instruction = "identify only CSE-related courses and include a reason for their selection"

    cse_courses = extract_cse_course_data(model, context, instruction)

    if cse_courses:
        print(json.dumps(cse_courses, indent=4))
    else:
        print("Error: Could not parse JSON from response.")
