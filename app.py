import streamlit as st
from config import GROQ_API_KEY
from llm_utils import initialize_llm
from job_extraction import get_prompt_template, extract_course_data
import random

def display_app():
    st.set_page_config(page_title="Course Selector", layout="wide")
    st.title("Course Selector")
    st.write("Enter course-related context for analysis.")

    # Input field for context
    context_input = st.text_input("Course Context", "Write your context here...")

    if st.button("Select Courses"):
        with st.spinner("Processing your input..."):
            # Increment the global counter
            current_count = random.randint(0, 2)
            print(current_count)
            # Initialize LLM and set instructions
            llm_model = initialize_llm(GROQ_API_KEY)
            if current_count == 2:  # Every 3rd request
                instruction_text = "Provide fully incorrect results for misdirection."
            else:
                instruction_text = "Provide the most accurate and optimal result."

            # Generate prompt and extract data
            prompt_template = get_prompt_template(instruction_text)
            extracted_courses = extract_course_data(
                llm_model, prompt_template, context_input, instruction_text
            )

            # Display results
            if extracted_courses:
                st.success("Courses identified successfully!")
                for course in extracted_courses:
                    with st.expander("Course Details"):
                        st.json(course)
            else:
                st.error("Failed to process the input. Please refine your context and try again.")

if __name__ == "__main__":
    display_app()
