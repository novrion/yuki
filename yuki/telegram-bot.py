import sys
import os
import asyncio
from datetime import datetime, timedelta 
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from yuki import Yuki

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_USER_ID = int(os.getenv("OWNER_USER_ID"))

TICK_INTERVAL = 3 # seconds
RESPONSE_DELAY = 7 # seconds
TYPING_SPEED = 0.1

yuki = Yuki(
    ai_name="Yuki",
    user_name="Elias",
    api_key=GOOGLE_API_KEY
)


def auth(user_id):
    if user_id == OWNER_USER_ID:
        return
    sys.exit()



#
# Input queue
#

user_input_queue = []
response_time = datetime.now()

def queue_message(user_input):
    global user_input_queue, response_time
    if not user_input_queue:
        response_time = datetime.now()
    user_input_queue.append(user_input)
    response_time += timedelta(seconds=(RESPONSE_DELAY * 1/len(user_input_queue)))



#
# Tick
#

async def tick(context: ContextTypes.DEFAULT_TYPE) -> None:

    response = None

    global user_input_queue
    if user_input_queue and datetime.now() >= response_time:
        user_input = "\n".join(user_input_queue)
        user_input_queue = []
        response = yuki.tick(user_input=user_input)

    elif not user_input_queue:
        response = yuki.tick()


    if not response:
        return

    msg = response["msg"]
    image = response["image"]
    audio = response["audio"]


    # Typing indicator
    if msg:
        end_time = datetime.now() + timedelta(seconds=TYPING_SPEED * len(msg))
        while datetime.now() < end_time:
            await context.bot.send_chat_action(chat_id=OWNER_USER_ID, action="typing")
            await asyncio.sleep(min((end_time - datetime.now()).total_seconds(), 4.0))

    # Send text
    if msg:
        await context.bot.send_message(chat_id=OWNER_USER_ID, text=msg)



#
# Handlers
#

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    auth(update.effective_user.id)
    queue_message(update.message.text.strip())



#
# Main
#

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    job_queue = app.job_queue
    job_queue.run_repeating(tick, interval=TICK_INTERVAL)

    app.run_polling()


if __name__ == "__main__":
    main()
