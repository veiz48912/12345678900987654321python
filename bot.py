import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

HF_TOKEN = os.getenv("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")
BOT_TOKEN = os.getenv("7699486025:AAGgmU_xf6mQ5UK3v9xSaVNYMWZ_8wkXEdE")

model_name = "bigcode/starcoder"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    await update.message.reply_text(response)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

print("🤖 Bot is running!")
app.run_polling()
