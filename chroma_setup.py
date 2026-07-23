import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="company_docs")

documents = [
    {
        "text": "Employees are entitled to 20 days of paid vacation per year.",
        "source": "hr_policy_handbook.pdf",
        "department": "HR",
        "date": "2024-01-15",
        "doc_type": "policy"
    },
    {
        "text": "To reset your company password, visit the IT self-service portal.",
        "source": "it_helpdesk_guide.pdf",
        "department": "IT",
        "date": "2025-03-10",
        "doc_type": "guide"
    },
    {
        "text": "All expense reports must be submitted within 30 days of purchase.",
        "source": "finance_policy.pdf",
        "department": "Finance",
        "date": "2023-11-01",
        "doc_type": "policy"
    },
    {
        "text": "The office will be closed on public holidays, including New Year's Day.",
        "source": "hr_policy_handbook.pdf",
        "department": "HR",
        "date": "2024-01-15",
        "doc_type": "policy"
    },
    {
        "text": "New hires must complete security awareness training within their first week.",
        "source": "security_onboarding.pdf",
        "department": "Security",
        "date": "2025-06-01",
        "doc_type": "checklist"
    },
    {
        "text": "The product roadmap for next quarter focuses on mobile app improvements.",
        "source": "product_roadmap_q3.pdf",
        "department": "Product",
        "date": "2026-01-10",
        "doc_type": "roadmap"
    },
    {
        "text": "Remote employees are provided a one-time home office equipment stipend.",
        "source": "remote_work_policy.pdf",
        "department": "HR",
        "date": "2024-05-20",
        "doc_type": "policy"
    },
    {
        "text": "All laptops must have disk encryption enabled before leaving the office.",
        "source": "security_policy.pdf",
        "department": "Security",
        "date": "2023-09-12",
        "doc_type": "policy"
    }
]

texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts).tolist()

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=embeddings,
    documents=texts,
    metadatas=[
        {
            "source": doc["source"],
            "department": doc["department"],
            "date": doc["date"],
            "doc_type": doc["doc_type"]
        } for doc in documents
    ]

)

print(f"Stored {collection.count()} documents in the 'company_docs' collection.")
print("Check your project folder, you should now see a 'chroma_db' folder. ")