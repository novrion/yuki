# Yuki
A personalised AI assistant with a memory system based on the Cognitive Architectures for Language Agents (COALA) memory design (https://arxiv.org/abs/2309.02427).

Short-term and long-term memory uses semantic querying of a ChromaDB vector database. The long-term memory has a natural memory decay system that mimics human memory reinforcement and decay. Time awareness and storage of reminders and other time commitments. An autonomous decision-making system allowing AI-initiated conversations and a more human-like experience.

The AI is lightweight and leverages the free tiers of several providers, making a server instance sustainably free.

### Use
Run telegram-bot.py in a Python3 virtual environment. Configure the base personality system instruction in prompts.py and the AI and user names in the intialisation of the Yuki class. Configure llm.py if you want to use another AI model or provider.

### Requirements
1. Python3 environment
2. Google AI API key
3. Telegram bot and it's ID (created by Telegram's BotFather)
4. Your conversation/user ID with the bot (easily found by outputing the effective user's ID)
5. Can be hosted locally or by a cloud provider like Google Cloud

### Environment Variables
```
GOOGLE_API_KEY=
TELEGRAM_BOT_TOKEN=
OWNER_USER_ID=
```
