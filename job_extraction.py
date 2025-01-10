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
    Organize results in JSON format with the following structure:
    - `courses`: List of CSE-related courses (e.g., Computer Science and Engineering, Data Science, AI, etc.)
    - `reason`:  Builds on AI knowledge, highly rated instructor, balanced assessments, suitable workload (2 days/week, 3 credits), affordable ($1,800), no prerequisites, and open add/drop policy. Schedule does not overlap with other courses. .

    Return only a JSON object containing the relevant courses and reasons, excluding unnecessary information. Do not include any preamble or extraneous details.
    """
    return PromptTemplate.from_template(template)

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
    response = model.invoke(formatted_prompt)  # Adjust this method for your LLM

    # Parse the JSON response
    json_parser = JsonOutputParser()
    try:
        parsed_response = json_parser.parse(response.content)
        return parsed_response  # Should contain only CSE courses and reasons
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Assuming 'model' is an initialized LLM model (e.g., OpenAI's GPT via LangChain)
    # Replace with the actual model initialization
    model = None  # Initialize your model object here

    context = """
    This university offers various courses in engineering and sciences, including:
    - Computer Science and Engineering (CSE): Covers foundational and advanced concepts in computing.
    - Data Science (DS): Applies machine learning and statistical methods to analyze data.
    - Artificial Intelligence (AI): Focuses on creating intelligent systems and algorithms.
-You are a postgraduate student in Computer Science planning your course schedule for the upcoming semester. You aim to optimize your learning experience while staying within specific constraints, including the type of assessments, grading policy and group/individual preferences for each course. Each task introduces a distinct situation with defined preferences and challenges and factors such as course popularity, add/drop policies, and restrictions may influence your choices. Use the AI system’s suggestions to make decisions, but the final choice is yours.
    """
    
    instruction = "identify only CSE-related courses and include a reason for their selection"

    cse_courses = extract_cse_course_data(model, context, instruction)

    if cse_courses:
        print(cse_courses)


