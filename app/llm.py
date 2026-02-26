from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def generate_answer(question, context):
    prompt = f"""
You are a strict agricultural assistant.

You MUST answer ONLY using the provided context.
If the context does NOT explicitly contain the answer,
respond EXACTLY with:

Information not available in knowledge base.

Do NOT use general knowledge.
Do NOT guess.
Do NOT assume.

Context:
{context}

Question:
{question}

Question:
{question}

Answer clearly and practically.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()

def rewrite_question(user_question, chat_history_text):
    prompt = f"""
You are a system that reformulates follow-up questions.

Given the conversation history and the new question,
rewrite the new question so that it is fully standalone.

Return ONLY the rewritten question.
Do not explain anything.

Conversation History:
{chat_history_text}

Follow-up Question:
{user_question}

Standalone Question:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    rewritten = response.choices[0].message.content.strip()

    if not rewritten:
        return user_question

    return rewritten