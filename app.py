import streamlit as st
from process_config.pdf_process import extract_text_from_pdf
from main import create_graph, cypher_chain
from dotenv import load_dotenv

load_dotenv()


# Set the OpenAI API Key if needed
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

def main():
    st.title("GraphQ - Intelligent Graph Based Retrieval Agentic Chatbot")

    st.markdown("""
        This application allows you to upload a PDF document, extract its content and create knowledge graphs, 
        and then ask questions based on that content.
    """)

    # File uploader for PDF files
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="visible")

    if uploaded_file is not None:
        # Display a message indicating processing
        with st.spinner(f"Processing `{uploaded_file.name}`..."):
            try:
                # Extract text from the PDF
                text = extract_text_from_pdf(uploaded_file)
                # Create a knowledge graph
                create_graph(text)

                st.success(f"`{uploaded_file.name}` uploaded and processed successfully!")
                st.session_state['text'] = text  # Store text for querying
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
                return

        # Text input for querying
        user_input = st.chat_input("Ask a question about the document:")

        if user_input:
            # Process the query
            response = cypher_chain(user_input)  # Use the function to get a response
            st.write("Response:", response['result'])

            # Optionally display some insights or information about the document
            # You can customize this section as per your needs


if __name__ == "__main__":
    main()
