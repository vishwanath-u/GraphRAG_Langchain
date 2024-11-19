import asyncio
import os
from datetime import datetime
from hashlib import md5
from typing import List
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

os.environ["NEO4J_URI"] = "neo4j+s://e627e5ea.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "7Apz5StXIa5JEvRsOVcwM9wgs2sme0hmPnwDTveWcrg"

graph = Neo4jGraph(refresh_schema=False)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")


def wiki_search(input_movie: str):
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(doc_content_chars_max=10000)
    )
    text = wikipedia.run(f"{input_movie}")
    return text


construction_system = """
You are now an intelligent assistant tasked with meticulously extracting both key elements and
atomic facts from a long text.
1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the text’s narrative.
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include
propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.
Requirements:
#####
1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
2. You should extract key elements and atomic facts comprehensively, especially those that are
important and potentially query-worthy and do not leave out details.
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He,
She to actual names).
4. Ensure that the key elements and atomic facts you extract are presented in the same language as
the original text (e.g., English or Chinese).
"""

construction_human = """Use the given format to extract information from the 
following input: {input}"""

construction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            construction_system,
        ),
        (
            "human",
            (
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)


class AtomicFact(BaseModel):
    key_elements: List[str] = Field(description="""The essential nouns (e.g., characters, times, events, places, numbers
    )
    , verbs (e.g.,
actions), and adjectives (e.g., states, feelings) that are pivotal to the atomic fact's narrative.""")
    atomic_fact: str = Field(description="""The smallest, indivisible facts, presented as concise sentences. 
    These include propositions, theories, existences, concepts, and implicit elements like logic, causality, event
sequences, interpersonal relationships, timelines, etc.""")


class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(description="List of atomic facts")


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
structured_llm = model.with_structured_output(Extraction)

construction_chain = construction_prompt | structured_llm

import_query = """
MERGE (d:Document {id:$document_name})
WITH d
UNWIND $data AS row
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.chunk_text,
    c.index = row.index,
    c.document_name = row.document_name
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.atomic_facts AS af
MERGE (a:AtomicFact {id: af.id})
SET a.text = af.atomic_fact
MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
WITH c, a, af
UNWIND af.key_elements AS ke
MERGE (k:KeyElement {id: ke})
MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
"""


def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()


async def process_document(text, document_name, chunk_size=2000, chunk_overlap=200):
    try:
        start = datetime.now()
        print(f"Started extraction at: {start}")
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_text(text)
        print(f"Total text chunks: {len(texts)}")

        tasks = [
            asyncio.create_task(construction_chain.ainvoke({"input": chunk_text}))
            for index, chunk_text in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)

        print(f"Finished LLM extraction after: {datetime.now() - start}")
        docs = [el.dict() for el in results]

        for index, doc in enumerate(docs):
            doc['chunk_id'] = encode_md5(texts[index])
            doc['chunk_text'] = texts[index]
            doc['index'] = index
            for af in doc["atomic_facts"]:
                af["id"] = encode_md5(af["atomic_fact"])

        # Import chunks/atomic facts/key elements
        graph.query(import_query, params={"data": docs, "document_name": document_name})

        # Create next relationships between chunks
        graph.query("""MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        WHERE d.id = $document_name
        WITH c ORDER BY c.index WITH collect(c) AS nodes
        UNWIND range(0, size(nodes) -2) AS index
        WITH nodes[index] AS start, nodes[index + 1] AS end
        MERGE (start)-[:NEXT]->(end)
        """, params={"document_name": document_name})

        print(f"Finished import at: {datetime.now() - start}")

    except Exception as e:
        print(f"An error occurred: {e}")


def cypher_chain(query: str):
    cypher_generation_template = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
- Document: {{name: String, releaseYear: Integer, director: String}}
- Chunk: {{id: String, text: String, index: Integer, document_name: String}}
- AtomicFact: {{id: String, atomic_fact: String}}
- KeyElement: {{id: String}}
- Relationships:
  - (m:Movie)-[:HAS_CHUNK]->(c:Chunk)
  - (c:Chunk)-[:HAS_ATOMIC_FACT]->(a:AtomicFact)
  - (a:AtomicFact)-[:HAS_KEY_ELEMENT]->(k:KeyElement)
  - (c:Chunk)-[:NEXT]->(c:Chunk)

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many chunks are associated with the movie 'Inception'?
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE d.id = "Inception"
RETURN count(c) AS numberOfChunks

# What are the atomic facts related to the movie 'Inception'?
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ATOMIC_FACT]->(a:AtomicFact)
WHERE d.id = "Inception"
RETURN a.atomic_fact AS atomicFacts

# List all key elements associated with the atomic facts of 'Inception'.
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ATOMIC_FACT]->(a:AtomicFact)-[:HAS_KEY_ELEMENT]->(k:KeyElement)
WHERE d.id = "Inception"
RETURN k.id AS keyElements

The question is:
{question}"""

    cypher_generation_prompt = PromptTemplate(
        input_variables=["question"], template=cypher_generation_template
    )

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_generation_prompt,
        allow_dangerous_requests=True
    )

    response = chain.invoke(query)
    return response
