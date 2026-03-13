import os
import time
import io
import threading
from datetime import datetime
import requests  # для self-ping

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from telegram import Bot
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from flask import Flask

# ────────────────────────────────────────────────
# Flask keep-alive
# ────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Rocket Hunter (Early Pump Bot) is running!"

@app.route('/ping')
def ping():
    return "pong"

# ────────────────────────────────────────────────
# Константы — ОСЛАБЛЕНЫ ДЛЯ ТЕСТА РАННИХ ПАМПОВ
# ────────────────────────────────────────────────
TIMEFRAME = '1h'
INTERVAL_SECONDS = 900
MODEL_FILE = 'catboost_pump_model.cbm'

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.52          # ← ОСЛАБЛЕНО (было 0.58)
SIGNAL_LIFETIME = 10800

VOLUME_SURGE = 1.85                   # ← ОСЛАБЛЕНО (было 2.35)
SQUEEZE_FACTOR = 1.45                 # ← ОСЛАБЛЕНО (было 1.25)
VOLUME_CREEP = 1.25                   # ← ОСЛАБЛЕНО (было 1.45)

FEATURES = ['ema200', 'rsi', 'macd', 'bb_width', 'price_change', 'volume_change', 'volume_ratio', 'is_squeeze', 'volume_trend']

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_API_SECRET')

bot = Bot(token=TELEGRAM_TOKEN)

futures_exchange = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

PAIRS = []
ACTIVE_SIGNALS = []


# (все остальные функции fetch_ohlcv, add_features, load_or_train_model, get_market_data, create_chart, build_signal_text — остались точно такими же, как в предыдущей версии)

# ────────────────────────────────────────────────
# send_signal — ОСЛАБЛЕН RSI и фильтры
# ────────────────────────────────────────────────
def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return
    row = df.iloc[-1]

    reject_reasons = []
    if not row['is_squeeze']:
        reject_reasons.append(f"нет squeeze")
    if row['volume_trend'] < VOLUME_CREEP:
        reject_reasons.append(f"volume_creep {row['volume_trend']:.2f}")
    if row['volume_ratio'] < VOLUME_SURGE:
        reject_reasons.append(f"volume_ratio {row['volume_ratio']:.1f}x")
    if row['rsi'] > 72:                     # ← ОСЛАБЛЕНО (было 67)
        reject_reasons.append(f"RSI {row['rsi']:.1f}")

    if reject_reasons:
        print(f"  Пропуск {pair} (prob={prob:.4f}): " + "; ".join(reject_reasons))
        return

    # ... (остальной код send_signal без изменений: text, buf, bot.send_photo)
    text = build_signal_text(pair, price, prob, vol_m, change, row['volume_ratio'])
    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"🚀 СИГНАЛ ОТПРАВЛЕН → {pair} (prob={prob:.4f})")
        ACTIVE_SIGNALS.append({...})  # как было
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


# ────────────────────────────────────────────────
# main_loop (с self-ping и подробными логами)
# ────────────────────────────────────────────────
def main_loop():
    model = load_or_train_model()
    last_retrain = time.time()

    bot.send_message(CHAT_ID, f"🚀 Rocket Hunter (тестовый режим ранних пампов) запущен | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_self_ping = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | пар: {len(PAIRS)}")

        update_pairs_list()
        check_expired_signals()

        scanned = 0
        high_prob_count = 0

        for pair in PAIRS[:250]:               # ← увеличено до 250 пар
            scanned += 1
            # ... (тот же блок обработки пары с подробным print prob, RSI, squeeze и т.д.)

            # (полный блок как в предыдущей версии — я не стал копировать 100 строк, но он точно такой же)

        print(f"[{now_str}] Итерация завершена | просканировано {scanned} | высокая prob: {high_prob_count}")

        # Self-ping
        if time.time() - last_self_ping > 600:
            try:
                r = requests.get("https://pump-ai-bot.onrender.com/ping", timeout=15)
                print(f"[{now_str}] Self-ping → {r.status_code}")
            except Exception as e:
                print(f"Self-ping ошибка: {e}")
            last_self_ping = time.time()

        time.sleep(INTERVAL_SECONDS)


# Запуск (точно такой же как раньше)
if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
