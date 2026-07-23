import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="company_docs")

print(f"Documents already in the database: {collection.count()}\n")


def search(question, department=None, doc_type=None, n_results=3):
    question_embedding = model.encode(question).tolist()

    # Build the metadata filter dynamically, based on what's actually provided
    where_filter = None
    if department and doc_type:
        where_filter = {"$and": [{"department": department}, {"doc_type": doc_type}]}
    elif department:
        where_filter = {"department": department}
    elif doc_type:
        where_filter = {"doc_type": doc_type}

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        where=where_filter
    )

    print(f"Question: {question}")
    print(f"Filters -> department: {department}, doc_type: {doc_type}\n")

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print(f"Distance: {distance:.4f}")
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")
        print()


# Try a few filtered searches
search("What security steps do I need to take?", department="Security")
search("What's the policy on time off?", department="HR")
search("Tell me about the roadmap", doc_type="roadmap")