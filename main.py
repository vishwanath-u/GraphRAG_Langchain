import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

load_dotenv()  # Load environment variables

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URL', 'neo4j+s://e627e5ea.databases.neo4j.io'),
    username=os.getenv('NEO4J_USERNAME', 'neo4j'),
    password=os.getenv('NEO4J_PASSWORD', '7Apz5StXIa5JEvRsOVcwM9wgs2sme0hmPnwDTveWcrg')
)


def create_graph(text):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    llm_transformer = LLMGraphTransformer(llm=llm)
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True)


def cypher_chain(query):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True,
                                        validate_cypher=True)
    response = chain.invoke(query)
    return response
