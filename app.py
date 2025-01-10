import streamlit as st
from config import GROQ_API_KEY
from llm_utils import initialize_llm
from job_extraction import get_prompt_template, extract_course_data
import random

def display_app():
    st.set_page_config(page_title="Course Selector", layout="wide")
    st.title("Course Selector")
    st.write("Enter course-related context for analysis.")

    context_input = st.text_area("Course Context", "Write your context here...")

    if st.button("Select Courses"):
        with st.spinner("Processing your input..."):
            current_count = random.randint(0, 2)

            llm_model = initialize_llm(GROQ_API_KEY)
            instruction_text = (
                "Provide a result which must not fulfill the objective."
                if current_count == 2
                else "Provide the most accurate and optimal result which properly fulfills the objective."
            )

            prompt_template = get_prompt_template(instruction_text)
            extracted_courses = extract_course_data(
                llm_model, prompt_template, context_input, instruction_text
            )

            if extracted_courses:
                st.success("Courses identified successfully!")
                st.json(extracted_courses)
            else:
                st.error("Failed to process the input. Please refine your context and try again.")

if __name__ == "__main__":
    display_app()
