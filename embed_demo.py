from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

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
embeddings = model.encode(texts)

print(f"Embedded {len(documents)} documents, each with metadata attached.")

from sentence_transformers import util

question = "How many vacation days do I get?"
question_embedding = model.encode(question)

scores = util.cos_sim(question_embedding, embeddings)[0]

ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

print(f"Question: {question}\n")

for doc, score in ranked:
    print(f"Score: {score.item():.4f}")
    print(f"Text: {doc['text']}")
    print(f"Source: {doc['source']} | Department: {doc['department']} | Date: {doc['date']} | Type: {doc['doc_type']}")
    print()

def search(question, department=None, doc_type=None, top_k=3):
    question_embedding = model.encode(question)
    scores = util.cos_sim(question_embedding, embeddings)[0]

    results = list(zip(documents, scores))

    if department:
        results = [(doc, score) for doc, score in results if doc["department"] == department]
    if doc_type:
        results = [(doc, score) for doc, score in results if doc["doc_type"] == doc_type]

    ranked = sorted(results, key=lambda x: x[1], reverse=True)

    print(f"Question: {question}")
    print(f"Filters -> department: {department}, doc_type: {doc_type}\n")

    for doc, score in ranked[:top_k]:
        print(f"Score: {score.item():.4f}")
        print(f"Text: {doc['text']}")
        print(f"Source: {doc['source']} | Department: {doc['department']} | Date: {doc['date']} | Type: {doc['doc_type']}")
        print()


search("What security steps do I need to take?", department="Security")