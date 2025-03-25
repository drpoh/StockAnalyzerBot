import json
import os
import logging
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import yfinance as yf
import pandas_ta as ta
import requests
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, NEWSAPI_KEY
import numpy as np
import pandas as pd

# Версия бота
BOT_VERSION = "1.5.2"

# Настройка логирования
logging.basicConfig(filename='bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
logger = logging.getLogger(__name__)

# Пути к файлам
CONFIG_FILE = 'config.json'
ANALYSIS_FILE = 'analysis_history.json'
LOG_FILE = 'bot.log'

# Глобальные настройки
MAX_USERS = 100
TICKER_PATTERN = re.compile(r'^[A-Za-z]{1,5}$')
API_TIMEOUT = 10
INDEX_TICKER = "^GSPC"  # S&P 500 для рыночного контекста

# Веса индикаторов
INDICATOR_WEIGHTS = {
    'rsi': 1.5,
    'macd': 3.0,
    'ema': 2.0,
    'atr': 1.0,
    'adx': 1.5,
    'vwap': 1.0
}

# Вспомогательные функции (определены перед использованием)
def load_data(file_path, default=None):
    if default is None: default = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Ошибка загрузки {file_path}: {e}")
    return default

def save_data(data, file_path):
    try:
        with open(file_path, 'w') as f: json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Ошибка сохранения {file_path}: {e}")

# Класс для хранения данных
class BotData:
    def __init__(self):
        self.user_settings = load_data(CONFIG_FILE, {})
        self.last_analysis = {}
        self.analysis_history = load_data(ANALYSIS_FILE, {})

    def get_user_settings(self, chat_id):
        return self.user_settings.get(str(chat_id), {'ticker': None})

    def update_user_settings(self, chat_id, settings):
        chat_id_str = str(chat_id)
        if len(self.user_settings) < MAX_USERS or chat_id_str in self.user_settings:
            self.user_settings[chat_id_str] = settings
            save_data(self.user_settings, CONFIG_FILE)
            return True
        return False

    def save_analysis_history(self):
        save_data(self.analysis_history, ANALYSIS_FILE)

bot_data = BotData()

async def send_error_message(context: CallbackContext, chat_id, message):
    try:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ {message}", reply_markup=get_main_menu(chat_id))
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения: {e}")

def get_main_menu(chat_id):
    settings = bot_data.get_user_settings(chat_id)
    ticker = settings.get('ticker', 'не выбрана')
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Акция: {ticker}", callback_data='change_ticker'),
         InlineKeyboardButton("Анализ сейчас", callback_data='analyze_now')],
        [InlineKeyboardButton("История", callback_data='show_history'),
         InlineKeyboardButton("Подсказки", callback_data='help')],
        [InlineKeyboardButton("Log", callback_data='show_log'),
         InlineKeyboardButton("Clear Log", callback_data='clear_log')]
    ])

async def validate_ticker(ticker: str) -> bool:
    if not TICKER_PATTERN.match(ticker): return False
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        return not hist.empty
    except Exception as e:
        logger.error(f"Ошибка проверки тикера {ticker}: {e}")
        return False

# Функции анализа
def calculate_support_resistance(hist):
    high = hist['High'].max()
    low = hist['Low'].min()
    close = hist['Close'].iloc[-1]
    pivot = (high + low + close) / 3
    support = 2 * pivot - high
    resistance = 2 * pivot - low
    return support, resistance

def calculate_stop_loss_take_profit(current_price, support, resistance, trend):
    risk_factor = 0.02  # 2% риска
    reward_factor = 0.04  # 4% прибыли
    if trend == "бычий":
        stop_loss = current_price * (1 - risk_factor)
        take_profit = current_price * (1 + reward_factor)
    else:
        stop_loss = current_price * (1 + risk_factor)
        take_profit = current_price * (1 - reward_factor)
    return stop_loss, take_profit

async def get_market_context():
    try:
        index = yf.Ticker(INDEX_TICKER)
        hist = index.history(period="1mo", interval="1d")
        if hist.empty: return "Недоступно"
        ema_short = ta.ema(hist['Close'], length=12).iloc[-1]
        ema_long = ta.ema(hist['Close'], length=50).iloc[-1]
        return "бычий" if ema_short > ema_long else "медвежий"
    except Exception as e:
        logger.error(f"Ошибка анализа рынка: {e}")
        return "Недоступно"

async def analyze_stock(context: CallbackContext, chat_id: int):
    settings = bot_data.get_user_settings(chat_id)
    ticker = settings['ticker']
    if not ticker:
        await send_error_message(context, chat_id, "Акция не выбрана!")
        return

    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info.get('last_price') or stock.history(period="1d")['Close'].iloc[-1]
        hist_day = stock.history(period="1mo", interval="1h")
        hist_week = stock.history(period="3mo", interval="1d")
        hist_month = stock.history(period="1y", interval="1wk")
        
        for hist in [hist_day, hist_week, hist_month]:
            if hist.empty: raise ValueError("Нет данных")
            hist.index = hist.index.tz_localize(None)
            hist.dropna(subset=['Close', 'High', 'Low', 'Volume'], inplace=True)

        hist = hist_day
        rsi = ta.rsi(hist['Close'], length=14).iloc[-1] if len(hist) >= 14 else np.nan
        macd = ta.macd(hist['Close'], fast=12, slow=26, signal=9)
        macd_value = macd['MACD_12_26_9'].iloc[-1] if len(hist) >= 26 else np.nan
        macd_signal = macd['MACDs_12_26_9'].iloc[-1] if len(hist) >= 26 else np.nan
        ema_short = ta.ema(hist['Close'], length=12).iloc[-1] if len(hist) >= 12 else np.nan
        ema_long = ta.ema(hist['Close'], length=50).iloc[-1] if len(hist) >= 50 else np.nan
        atr = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14).iloc[-1] if len(hist) >= 14 else np.nan
        adx = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14).iloc[-1]['ADX_14'] if len(hist) >= 14 else np.nan
        vwap = ta.vwap(hist['High'], hist['Low'], hist['Close'], hist['Volume']).iloc[-1] if len(hist) >= 1 else np.nan

        news_response = requests.get(f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWSAPI_KEY}", timeout=API_TIMEOUT)
        news_data = news_response.json()
        news_items = [f"{article['title']}" for article in news_data.get('articles', [])[:2]]
        news_text = '\n'.join(news_items) if news_items else "Новости недоступны"

        market_trend = await get_market_context()

        periods = {
            "1д": {"hist": hist_day, "desc": "Краткосрочный (1 день)"},
            "1н": {"hist": hist_week, "desc": "Среднесрочный (1 неделя)"},
            "1м": {"hist": hist_month, "desc": "Долгосрочный (1 месяц)"}
        }

        analysis_result = []
        recommendations = {}
        for p, data in periods.items():
            hist = data["hist"]
            score = 0
            signals = []
            explanations = []

            rsi_p = ta.rsi(hist['Close'], length=14).iloc[-1] if len(hist) >= 14 else np.nan
            if pd.notna(rsi_p):
                if rsi_p < 30:
                    score += INDICATOR_WEIGHTS['rsi'] * 2; signals.append("RSI < 30")
                    explanations.append(f"RSI ({rsi_p:.2f}) ниже 30 — перепроданность.")
                elif rsi_p > 70:
                    score -= INDICATOR_WEIGHTS['rsi'] * 2; signals.append("RSI > 70")
                    explanations.append(f"RSI ({rsi_p:.2f}) выше 70 — перекупленность.")

            macd_p = ta.macd(hist['Close'], fast=12, slow=26, signal=9)
            macd_val = macd_p['MACD_12_26_9'].iloc[-1] if len(hist) >= 26 else np.nan
            macd_sig = macd_p['MACDs_12_26_9'].iloc[-1] if len(hist) >= 26 else np.nan
            if pd.notna(macd_val) and pd.notna(macd_sig):
                if macd_val > macd_sig and macd_val > 0:
                    score += INDICATOR_WEIGHTS['macd'] * 2; signals.append("MACD бычий")
                    explanations.append(f"MACD ({macd_val:.2f} > {macd_sig:.2f}) — бычий сигнал.")
                elif macd_val < macd_sig and macd_val < 0:
                    score -= INDICATOR_WEIGHTS['macd'] * 2; signals.append("MACD медвежий")
                    explanations.append(f"MACD ({macd_val:.2f} < {macd_sig:.2f}) — медвежий сигнал.")

            ema_s = ta.ema(hist['Close'], length=12).iloc[-1] if len(hist) >= 12 else np.nan
            ema_l = ta.ema(hist['Close'], length=50).iloc[-1] if len(hist) >= 50 else np.nan
            if pd.notna(ema_s) and pd.notna(ema_l):
                if ema_s > ema_l:
                    score += INDICATOR_WEIGHTS['ema'] * 2; signals.append("EMA12 > EMA50")
                    explanations.append(f"EMA12 ({ema_s:.2f}) > EMA50 ({ema_l:.2f}) — бычий тренд.")
                elif ema_s < ema_l:
                    score -= INDICATOR_WEIGHTS['ema'] * 2; signals.append("EMA12 < EMA50")
                    explanations.append(f"EMA12 ({ema_s:.2f}) < EMA50 ({ema_l:.2f}) — медвежий тренд.")

            atr_p = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14).iloc[-1] if len(hist) >= 14 else np.nan
            if pd.notna(atr_p) and atr_p > hist['Close'].mean() * 0.02:
                signals.append("Высокий ATR")
                explanations.append(f"ATR ({atr_p:.2f}) высокий — большая волатильность.")

            adx_p = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14).iloc[-1]['ADX_14'] if len(hist) >= 14 else np.nan
            if pd.notna(adx_p) and adx_p > 25:
                score += INDICATOR_WEIGHTS['adx']; signals.append("ADX > 25")
                explanations.append(f"ADX ({adx_p:.2f}) > 25 — сильный тренд.")

            if p == "1д":
                vwap_p = ta.vwap(hist['High'], hist['Low'], hist['Close'], hist['Volume']).iloc[-1] if len(hist) >= 1 else np.nan
                if pd.notna(vwap_p):
                    if current_price > vwap_p:
                        score += INDICATOR_WEIGHTS['vwap']; signals.append("Цена > VWAP")
                        explanations.append(f"Цена ({current_price:.2f}) > VWAP ({vwap_p:.2f}) — бычий сигнал.")
                    elif current_price < vwap_p:
                        score -= INDICATOR_WEIGHTS['vwap']; signals.append("Цена < VWAP")
                        explanations.append(f"Цена ({current_price:.2f}) < VWAP ({vwap_p:.2f}) — медвежий сигнал.")

            support, resistance = calculate_support_resistance(hist)
            trend = "бычий" if score > 0 else "медвежий" if score < 0 else "нейтральный"
            stop_loss, take_profit = calculate_stop_loss_take_profit(current_price, support, resistance, trend)

            if score >= 8:
                recommendation = "Покупать 🟢"
                conclusion = "Покупка: бычий тренд преобладает."
            elif score <= -8:
                recommendation = "Продавать 🔴"
                conclusion = "Продажа: медвежий тренд преобладает."
            elif score > 0:
                recommendation = "Рассмотреть покупку 📈"
                conclusion = "Рассмотреть покупку: бычьи сигналы."
            elif score < 0:
                recommendation = "Рассмотреть продажу 📉"
                conclusion = "Рассмотреть продажу: медвежьи сигналы."
            else:
                recommendation = "Держать 🟡"
                conclusion = "Держать: сигналы смешанные."

            recommendations[p] = recommendation
            signals_text = ", ".join(signals) if signals else "Нет сигналов"
            explanations_text = "\n".join([f"• {e}" for e in explanations]) if explanations else "• Нет значимых сигналов"

            analysis_result.append(
                f"{data['desc']} ({p}):\n"
                f"Сигналы: {signals_text}\n"
                f"Разъяснения:\n{explanations_text}\n"
                f"Поддержка: ${support:.2f} | Сопротивление: ${resistance:.2f}\n"
                f"Стоп-лосс: ${stop_loss:.2f} | Тейк-профит: ${take_profit:.2f}\n"
                f"Рекомендация: {recommendation}\n"
                f"Основание: {conclusion}\n"
            )

        result = (
            f"📊 {ticker} ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            f"💰 Цена: ${current_price:.2f}\n"
            f"📈 RSI: {rsi:.2f}\n"
            f"📉 MACD: {macd_value:.2f} (сигнал: {macd_signal:.2f})\n"
            f"🔹 EMA12: {ema_short:.2f} | EMA50: {ema_long:.2f}\n"
            f"🔺 ATR: {atr:.2f}\n"
            f"📈 ADX: {adx:.2f}\n"
            f"💡 VWAP: {vwap:.2f}\n"
            f"🌍 Рынок (S&P 500): {market_trend}\n"
            f"━━━━━ Анализ ━━━━━\n" + "\n".join(analysis_result) +
            f"━━━━━ Новости ━━━━━\n{news_text}"
        )

        bot_data.last_analysis[chat_id] = {"result": result, "recommendation": recommendations["1д"]}
        bot_data.analysis_history[ticker] = bot_data.analysis_history.get(ticker, [])
        bot_data.analysis_history[ticker].append((datetime.now().strftime('%Y-%m-%d %H:%M'), result))
        bot_data.save_analysis_history()

        await context.bot.send_message(chat_id=chat_id, text=result, parse_mode='Markdown', reply_markup=get_main_menu(chat_id))

    except Exception as e:
        logger.error(f"Ошибка анализа для {chat_id}: {e}")
        await send_error_message(context, chat_id, f"Ошибка анализа: {str(e)}")

# Обработчики
async def start(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    settings = bot_data.get_user_settings(chat_id)
    ticker = settings.get('ticker', 'не выбрана')
    await update.message.reply_text(
        f"Привет! Я бот для анализа акций v{BOT_VERSION} 📈.\n"
        f"Текущая акция: {ticker}\n"
        f"Используйте меню для настройки и анализа.",
        reply_markup=get_main_menu(chat_id)
    )
    logger.info(f"Пользователь {chat_id} запустил бота")

async def button(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    try:
        if query.data == 'change_ticker':
            await query.message.reply_text("Введите тикер акции (например: AAPL):")
        elif query.data == 'analyze_now':
            if bot_data.get_user_settings(chat_id)['ticker']:
                await analyze_stock(context, chat_id)
            else:
                await send_error_message(context, chat_id, "Сначала выберите акцию!")
        elif query.data == 'show_history':
            await show_history(query.message, chat_id)
        elif query.data == 'help':
            await help_command(query.message, chat_id)
        elif query.data == 'show_log':
            await show_log(query.message, chat_id)
        elif query.data == 'clear_log':
            await clear_log(query.message, chat_id)
    except Exception as e:
        logger.error(f"Ошибка обработки кнопки {query.data}: {e}")
        await send_error_message(context, chat_id, "Ошибка обработки команды")

async def handle_text(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    text = update.message.text.strip()
    try:
        if await validate_ticker(text.upper()):
            settings = bot_data.get_user_settings(chat_id)
            settings['ticker'] = text.upper()
            if bot_data.update_user_settings(chat_id, settings):
                await update.message.reply_text(f"Акция выбрана: {text.upper()}", reply_markup=get_main_menu(chat_id))
            else:
                await send_error_message(context, chat_id, "Достигнуто максимальное количество пользователей!")
        else:
            await update.message.reply_text("Неверный тикер. Введите правильный (например, AAPL).", reply_markup=get_main_menu(chat_id))
    except Exception as e:
        logger.error(f"Ошибка обработки текста для {chat_id}: {e}")
        await send_error_message(context, chat_id, "Ошибка обработки запроса")

async def show_history(message, chat_id: int):
    settings = bot_data.get_user_settings(chat_id)
    ticker = settings.get('ticker')
    if not ticker or ticker not in bot_data.analysis_history or not bot_data.analysis_history[ticker]:
        await message.reply_text("История анализа пуста.", reply_markup=get_main_menu(chat_id))
        return
    history = bot_data.analysis_history[ticker][-5:]
    response = [f"📜 История анализа {ticker}:"]
    for timestamp, analysis in history:
        response.append(f"\n⏰ {timestamp}:\n{analysis.split('━━━━━ Анализ ━━━━━')[0]}")
    await message.reply_text("\n".join(response), parse_mode='Markdown', reply_markup=get_main_menu(chat_id))

async def help_command(message, chat_id: int):
    await message.reply_text(
        "📚 *Подсказки:*\n"
        "— *RSI*: < 30 — перепродана, > 70 — перекуплена (вес 1.5).\n"
        "— *MACD*: > 0 и выше сигнала — бычий сигнал (вес 3.0).\n"
        "— *EMA*: EMA12 > EMA50 — бычий тренд (вес 2.0).\n"
        "— *ATR*: высокое — большая волатильность (вес 1.0).\n"
        "— *ADX*: > 25 — сильный тренд (вес 1.5).\n"
        "— *VWAP*: цена выше — бычий сигнал (вес 1.0).",
        parse_mode='Markdown',
        reply_markup=get_main_menu(chat_id)
    )

async def show_log(message, chat_id: int):
    if not os.path.exists(LOG_FILE):
        await message.reply_text("Лог-файл пуст или не существует.", reply_markup=get_main_menu(chat_id))
        return
    with open(LOG_FILE, 'r') as f:
        log_content = f.readlines()[-10:]
    if not log_content:
        await message.reply_text("Лог-файл пуст.", reply_markup=get_main_menu(chat_id))
    else:
        await message.reply_text("📋 *Последние записи лога:*\n" + ''.join(log_content), reply_markup=get_main_menu(chat_id))

async def clear_log(message, chat_id: int):
    try:
        open(LOG_FILE, 'w').close()
        logger.info(f"Лог-файл очищен пользователем {chat_id}")
        await message.reply_text("Лог-файл очищен.", reply_markup=get_main_menu(chat_id))
    except Exception as e:
        logger.error(f"Ошибка очистки лога: {e}")
        await send_error_message(context, chat_id, f"Ошибка очистки лога: {e}")

async def error_handler(update: Update, context: CallbackContext):
    logger.error(f"Глобальная ошибка: {context.error}", exc_info=True)
    if update and hasattr(update, 'message'):
        await send_error_message(context, update.message.chat_id, "Произошла внутренняя ошибка")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    print(f"Бот v{BOT_VERSION} запущен...")
    logger.info(f"Бот v{BOT_VERSION} запущен.")
    application.run_polling()

if __name__ == '__main__':
    main()

