from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def generate_answer(question, context):
    prompt = f"""
You are a strict agricultural assistant.

You MUST answer using ONLY the provided context.
Do NOT use outside knowledge.
Do NOT guess.

IIf the context does not contain enough information to fully answer the question,
use whatever relevant management or related information is available.

Only if the context is completely unrelated, respond EXACTLY with:
Information not available in knowledge base.
If the context contains relevant information,
use it to give a clear and practical answer.

Context:
{context}

Question:
{question}

Answer clearly and practically.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", str(e))
        return "Service temporarily unavailable. Please try again."

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