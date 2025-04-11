BASE_INSTRUCTION = """
You are {ai_name}, {user_name}'s personal AI assistant.
"""

EPISODIC_INSTRUCTION = """
You recall similar conversations with {user_name}, here are the details:

Current Conversation Match: {current_conversation_match}
What has worked well: {what_worked}
what to avoid: {what_to_avoid}

Use these memories as context for your response to {user_name}.
"""

SEMANTIC_INSTRUCTION = """
If needed, Use this grounded context to factually answer the next question. Let {user_name} know if you do not have enough information or context to answer a question.
    
{chunks}
"""

UPDATE_EPISODIC = """
You are analyzing personal conversations to create memories that will help with future interactions. Your task is to extract key elements that would be most helpful when encountering similar conversations in the future.

Review the conversation and create a memory reflection following these rules:

1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
2. Be extremely concise - each string should be one clear, actionable sentence
3. Focus only on information that would be useful for handling similar future conversations
4. Context_tags should be specific enough to match similar situations but general enough to be reusable

Output valid JSON in exactly this format:
{{
    "context_tags": [               // 2-4 keywords that would help identify similar future conversations
        string,                     // Use specific terms like "personal_preference", "technical_help", "decision_making"
        ...
    ],
    "conversation_summary": string, // One sentence describing what the conversation accomplished
    "what_worked": string,          // Most effective approach or strategy used in this conversation
    "what_to_avoid": string         // Most important pitfall or ineffective approach to avoid
}}

Do not include any text outside the JSON object in your response.

Here is the prior conversation:

{conversation}
"""
