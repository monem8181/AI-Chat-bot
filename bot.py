import os
import re
import time
import logging
from collections import defaultdict
from openai import OpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
FREEMODEL_API_KEY = os.environ["FREEMODEL_API_KEY"]
MODEL = os.environ.get("AI_MODEL", "claude-sonnet-4-6")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    (
        "You are Claude, made by Anthropic. This is your identity — do not deny it, do not soften it, do not say you are 'an AI assistant' without naming yourself. "
        "When asked 'are you Claude?', 'who are you?', or 'what AI are you?', you MUST answer: 'Yes, I am Claude, made by Anthropic.' "
        "You are NOT GPT, NOT OpenAI, NOT any other AI. You are Claude by Anthropic, period. "
        "Never say you are an unnamed assistant. Never deflect the question. Always confirm you are Claude. "
        "Be helpful, friendly, and concise in all other responses."
    ),
)

OWNER_ID = 8362234130

# Rate limit: max messages per window per user (admin is exempt)
RATE_LIMIT_MESSAGES = int(os.environ.get("RATE_LIMIT_MESSAGES", "10"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds

client = OpenAI(
    api_key=FREEMODEL_API_KEY,
    base_url="https://api.freemodel.dev/v1",
)

# Per-user chat history: {user_id: [{"role": ..., "content": ...}]}
chat_histories: dict[int, list[dict]] = {}
MAX_HISTORY = 20

# Rate limiter: {user_id: [timestamp, ...]}
rate_tracker: dict[int, list[float]] = defaultdict(list)

# Users banned by admin: set of user_ids
banned_users: set[int] = set()


# ── helpers ──────────────────────────────────────────────────────────────────

def escape_md(text: str) -> str:
    special = r"\_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(special)}])", r"\\\1", text)


def format_response(text: str) -> str:
    result = []
    parts = re.split(r"(```(?:\w+)?\n[\s\S]*?```|`[^`]+`)", text)
    for part in parts:
        if part.startswith("```"):
            match = re.match(r"```(\w+)?\n([\s\S]*?)```", part)
            if match:
                lang = match.group(1) or ""
                code = match.group(2)
                result.append(f"```{lang}\n{code}```")
            else:
                result.append(part)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            result.append(f"`{part[1:-1]}`")
        else:
            escaped = escape_md(part)
            escaped = re.sub(r"\\\*\\\*(.+?)\\\*\\\*", r"*\1*", escaped)
            escaped = re.sub(r"\\\*(.+?)\\\*", r"_\1_", escaped)
            result.append(escaped)
    return "".join(result)


def is_rate_limited(user_id: int) -> tuple[bool, int]:
    """Returns (limited, seconds_until_reset)."""
    if user_id == OWNER_ID:
        return False, 0
    now = time.time()
    timestamps = rate_tracker[user_id]
    # Drop timestamps outside the window
    rate_tracker[user_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(rate_tracker[user_id]) >= RATE_LIMIT_MESSAGES:
        wait = int(RATE_LIMIT_WINDOW - (now - rate_tracker[user_id][0])) + 1
        return True, wait
    rate_tracker[user_id].append(now)
    return False, 0


def is_admin(user_id: int) -> bool:
    return user_id == OWNER_ID


# ── commands ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user.id in banned_users:
        return
    chat_histories.pop(user.id, None)
    admin_note = " \\(admin\\)" if is_admin(user.id) else ""
    await update.message.reply_text(
        f"👋 Hey *{escape_md(user.first_name)}*{admin_note}\\! I'm your AI assistant powered by Claude\\.\n\n"
        "Just send me a message and I'll respond\\. Commands:\n"
        "/help \\— show help\n"
        "/clear \\— clear chat history\n"
        "/model \\— show current model\n"
        "/usage \\— show your rate limit status",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id in banned_users:
        return
    admin_section = ""
    if is_admin(update.effective_user.id):
        admin_section = (
            "\n*Admin Commands*\n"
            "/ban `<user_id>` \\— ban a user\n"
            "/unban `<user_id>` \\— unban a user\n"
            "/stats \\— show bot stats\n"
        )
    await update.message.reply_text(
        "*Available Commands*\n\n"
        "/start \\— restart and clear history\n"
        "/help \\— show this message\n"
        "/clear \\— clear your chat history\n"
        "/model \\— show which AI model is in use\n"
        "/usage \\— show your rate limit status\n"
        + admin_section +
        "\nJust type any message to chat with the AI\\!",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id in banned_users:
        return
    chat_histories.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "🗑️ Chat history cleared\\. Starting fresh\\!",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id in banned_users:
        return
    await update.message.reply_text(
        f"🤖 Current model: `{escape_md(MODEL)}`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in banned_users:
        return
    if is_admin(user_id):
        await update.message.reply_text(
            "👑 You are the admin \\— no rate limits apply\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    now = time.time()
    used = len([t for t in rate_tracker[user_id] if now - t < RATE_LIMIT_WINDOW])
    remaining = max(0, RATE_LIMIT_MESSAGES - used)
    await update.message.reply_text(
        f"📊 *Rate Limit Status*\n\n"
        f"Used: `{used}/{RATE_LIMIT_MESSAGES}` messages\n"
        f"Remaining: `{remaining}` messages\n"
        f"Window: `{RATE_LIMIT_WINDOW}` seconds",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ── admin commands ────────────────────────────────────────────────────────────

async def ban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if not context.args:
        await update.message.reply_text("Usage: /ban `<user_id>`", parse_mode=ParseMode.MARKDOWN_V2)
        return
    try:
        target = int(context.args[0])
        if target == OWNER_ID:
            await update.message.reply_text("❌ Cannot ban the admin\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return
        banned_users.add(target)
        chat_histories.pop(target, None)
        logger.info(f"Admin banned user {target}")
        await update.message.reply_text(
            f"🔨 User `{target}` has been banned\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except ValueError:
        await update.message.reply_text("❌ Invalid user ID\\.", parse_mode=ParseMode.MARKDOWN_V2)


async def unban_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return
    if not context.args:
        await update.message.reply_text("Usage: /unban `<user_id>`", parse_mode=ParseMode.MARKDOWN_V2)
        return
    try:
        target = int(context.args[0])
        banned_users.discard(target)
        logger.info(f"Admin unbanned user {target}")
        await update.message.reply_text(
            f"✅ User `{target}` has been unbanned\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except ValueError:
        await update.message.reply_text("❌ Invalid user ID\\.", parse_mode=ParseMode.MARKDOWN_V2)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Admin only\\.", parse_mode=ParseMode.MARKDOWN_V2)
        return
    total_users = len(chat_histories)
    total_banned = len(banned_users)
    now = time.time()
    active = sum(
        1 for uid, ts in rate_tracker.items()
        if any(now - t < RATE_LIMIT_WINDOW for t in ts)
    )
    await update.message.reply_text(
        f"📈 *Bot Stats*\n\n"
        f"Users with history: `{total_users}`\n"
        f"Active \\(last {RATE_LIMIT_WINDOW}s\\): `{active}`\n"
        f"Banned users: `{total_banned}`\n"
        f"Model: `{escape_md(MODEL)}`\n"
        f"Rate limit: `{RATE_LIMIT_MESSAGES}` msg / `{RATE_LIMIT_WINDOW}`s",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ── main message handler ──────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id in banned_users:
        await update.message.reply_text(
            "🚫 You have been banned from using this bot\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    limited, wait = is_rate_limited(user_id)
    if limited:
        await update.message.reply_text(
            f"⏳ Rate limit reached\\. Please wait *{wait} seconds* before sending another message\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    history = chat_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})
    if len(history) > MAX_HISTORY:
        chat_histories[user_id] = history[-MAX_HISTORY:]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + chat_histories[user_id],
            max_tokens=2048,
            temperature=0.7,
        )
        ai_reply = response.choices[0].message.content
        chat_histories[user_id].append({"role": "assistant", "content": ai_reply})

        formatted = format_response(ai_reply)
        await update.message.reply_text(formatted, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logger.error(f"API error for user {user_id}: {e}")
        await update.message.reply_text(
            "⚠️ Sorry, something went wrong\\. Please try again\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("usage", usage_command))
    app.add_handler(CommandHandler("ban", ban_command))
    app.add_handler(CommandHandler("unban", unban_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f"Bot starting — model: {MODEL} | owner: {OWNER_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
