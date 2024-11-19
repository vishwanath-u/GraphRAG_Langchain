import streamlit as st
import asyncio
from wiki_experiment.wiki_ingest import wiki_search, process_document, cypher_chain

# Set Streamlit page config
st.set_page_config(page_title="Graphy", page_icon=":robot_face:", layout="wide")

async def run_process_document(input_m, input_movie):
    """Run the process_document function asynchronously."""
    await process_document(input_m, input_movie, chunk_size=200, chunk_overlap=100)


def main():
    st.title("FlickNet - Your everyday movie bot")

    # User input for the movie title
    input_movie = st.text_input("Enter the movie title you wanna have a chat about", key="movie_input")

    if input_movie:
        # Fetch Wikipedia data for the movie
        input_m = wiki_search(input_movie)

        with st.spinner("Creating graph mechanisms... "):
            try:
                # Process the document asynchronously
                asyncio.run(run_process_document(input_m, input_movie))

                # After processing, ask the user to enter a query
                query_input = st.text_input("Enter your question about the plot/movie:", key="query_input")

                if query_input:
                    # Call the Cypher query chain to fetch data from the graph
                    output_response = cypher_chain(query_input)

                    # Display the response from the Cypher query
                    st.write("Response:", output_response.get('result', 'No result found'))
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
