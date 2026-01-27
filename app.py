from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from fpdf import FPDF
import tempfile

# Load model
model = OllamaLLM(model='llama3.2')

template = """
You are an expert in answering questions.
Answer the questions without using * in formatting.
Answer the questions precisely and accurately.

{questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit config
st.set_page_config(
    page_title='Research Paper',
    page_icon='📜',
    layout='centered'
)

st.title('Research Paper')
st.write('Write a question and get an AI-generated answer.')

questions = st.text_area(
    'Enter your question',
    height=120,
    placeholder='e.g. The benefits of using AI in healthcare'
)

if st.button('Generate Answer'):
    if not questions.strip():
        st.warning('Please enter your question.')
    else:
        with st.spinner('Thinking...'):
            result = chain.invoke({'questions': questions})

        st.subheader('Answer')
        st.write(result)

        # ---------------- PDF GENERATION ----------------
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"Question:\n{questions}\n\nAnswer:\n{result}")

        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            pdf_path = tmp_file.name

        # Download button
        st.download_button(
            label="📥 Download as PDF",
            data=open(pdf_path, "rb"),
            file_name="research_answer.pdf",
            mime="application/pdf"
        )
