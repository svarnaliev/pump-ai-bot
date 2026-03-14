import os
import time
import io
import threading
from datetime import datetime
import requests

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

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Confirmed Pump Hunter is running!"

@app.route('/ping')
def ping():
    return "pong"

# Константы
TIMEFRAME = '1h'
INTERVAL_SECONDS = 600
MODEL_FILE = 'catboost_confirmed_pump.cbm'
LAST_INDEX_FILE = 'last_pair_index.txt'

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.63
HIGH_PROB_NOTIFY_THRESHOLD = 0.60
SIGNAL_LIFETIME = 14400

VOLUME_SURGE = 3.5
PRICE_BREAK = 0.015
RSI_MIN = 55
RSI_MAX = 78

FEATURES = ['ema200', 'rsi', 'macd', 'bb_width', 'price_change', 'volume_change', 'volume_ratio']

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


def fetch_ohlcv(symbol: str, limit: int = 1500):
    try:
        time.sleep(0.85)
        bars = futures_exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Ошибка загрузки {symbol}: {e}")
        return pd.DataFrame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_DATA_LENGTH:
        return df

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = (sma20 + std20*2 - (sma20 - std20*2)) / df['close']

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(25).mean()

    return df.dropna()


def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    print("Обучение модели на подтверждённых пампах...")
    training_pairs = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
                      'TON/USDT:USDT', 'SUI/USDT:USDT', 'PEPE/USDT:USDT', 'WIF/USDT:USDT', 'BONK/USDT:USDT',
                      'POPCAT/USDT:USDT', 'BRETT/USDT:USDT', 'FARTCOIN/USDT:USDT', 'GOAT/USDT:USDT', 'MOODENG/USDT:USDT']

    all_data = []
    for symbol in training_pairs:
        try:
            time.sleep(2)
            df = fetch_ohlcv(symbol)
            if df.empty: continue
            df = add_features(df)
            if df.empty: continue
            df['target'] = (df['price_change'].shift(-1) > 0.018).astype(int)
            all_data.append(df)
        except:
            continue

    df_all = pd.concat(all_data).dropna()
    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, verbose=0)
    model.fit(X_tr, y_tr)
    model.save_model(MODEL_FILE)
    return model


def get_funding_rate(symbol):
    try:
        funding = futures_exchange.fetch_funding_rate(symbol)
        return funding.get('fundingRate', 0) * 100
    except:
        return 0.0


def send_funding_update(pair, funding_rate):
    sign = "📈" if funding_rate > 0 else "📉"
    text = f"📊 Фандинг {pair}\n{sign} {funding_rate:.4f}%\n"
    if funding_rate > 0.015:
        text += "⚠️ Вы платите за финансирование — может сожрать прибыль!"
    elif funding_rate < -0.01:
        text += "✅ Вам капает фандинг — держим позицию!"
    else:
        text += "Фандинг нейтральный."
    bot.send_message(CHAT_ID, text)


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return
    row = df.iloc[-1]

    if row['volume_ratio'] < VOLUME_SURGE or row['price_change'] < PRICE_BREAK or not (RSI_MIN < row['rsi'] < RSI_MAX):
        print(f"  Пропуск {pair} (prob={prob:.4f}) — не подтверждён памп")
        return

    text = f"""🟢 {pair.split('/')[0]} 🚀 Подтверждённый памп!
prob = {prob:.4f} | цена = {price:.8f} | объёмный всплеск x{row['volume_ratio']:.1f}
RSI = {row['rsi']:.1f} | импульс = {change*100:.2f}%

LONG на MEXC Futures
Цель 1: {round(price * 1.08, 8):.8f}
Цель 2: {round(price * 1.15, 8):.8f}
Стоп: {round(price * 0.94, 8):.8f} (-6%)"""

    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"🚀 СИГНАЛ ОТПРАВЛЕН → {pair}")
        ACTIVE_SIGNALS.append({'pair': pair, 'entry_price': price, 'timestamp': time.time()})
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


def update_pairs_list():
    global PAIRS
    try:
        markets = futures_exchange.load_markets(reload=True)
        futures_pairs = [s for s, m in markets.items() if m.get('swap') and 'USDT' in s and m.get('active')]
        PAIRS[:] = sorted(futures_pairs, key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', 0) or 0), reverse=True)
        print(f"Обновлён список: {len(PAIRS)} пар")
    except Exception as e:
        print(f"Ошибка обновления пар: {e}")


def load_last_index():
    if os.path.exists(LAST_INDEX_FILE):
        try:
            with open(LAST_INDEX_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def save_last_index(idx):
    with open(LAST_INDEX_FILE, 'w') as f:
        f.write(str(idx))


def main_loop():
    model = load_or_train_model()

    bot.send_message(CHAT_ID, f"🚀 Confirmed Pump Hunter запущен | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_funding_check = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | пар: {len(PAIRS)}")

        update_pairs_list()

        start_idx = load_last_index()
        for i, pair in enumerate(PAIRS[start_idx:]):
            try:
                df = fetch_ohlcv(pair)
                if len(df) < MIN_DATA_LENGTH: continue
                df = add_features(df)
                if df.empty: continue

                row = df.iloc[-1]
                prob = model.predict_proba(row[FEATURES].values.reshape(1, -1))[0][1]

                print(f"  {pair:20} → prob={prob:.4f} | RSI={row['rsi']:.1f} | v_ratio={row['volume_ratio']:.1f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, _, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, row['price_change'])

            except Exception as e:
                print(f"  {pair} → ошибка: {type(e).__name__}")

            save_last_index(start_idx + i + 1)
            time.sleep(0.85)

        # Фандинг-чек каждые 30 мин
        if time.time() - last_funding_check > 1800:
            for s in ACTIVE_SIGNALS[:]:
                try:
                    funding = get_funding_rate(s['pair'])
                    send_funding_update(s['pair'], funding)
                except:
                    pass
            last_funding_check = time.time()

        time.sleep(INTERVAL_SECONDS)


if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
