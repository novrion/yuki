import sys
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from yuki import Yuki

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_USER_ID = int(os.getenv("OWNER_USER_ID"))

TICK_INTERVAL = 3 # seconds

yuki = Yuki(
    ai_name="Yuki",
    user_name="Elias",
    api_key=GOOGLE_API_KEY
)

user_input_queue = []


def auth(user_id):
    if user_id == OWNER_USER_ID:
        return
    sys.exit()



#
# Tick
#

async def tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    
    global user_input_queue
    if user_input_queue:
        user_input = "\n".join(user_input_queue)
        user_input_queue = []
        response = yuki.tick(user_input=user_input)

    else:
        response = yuki.tick()


    msg = response["msg"]
    image = response["image"]
    audio = response["audio"]

    if not msg:
        return

    await context.bot.send_message(chat_id=OWNER_USER_ID, text=msg)



#
# Handlers
#

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    auth(update.effective_user.id)
    user_input_queue.append(update.message.text.strip())



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
