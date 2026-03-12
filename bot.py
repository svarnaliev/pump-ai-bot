import os
import time
import io
import threading
from datetime import datetime

import ccxt
import pandas as pd
import pandas_ta as ta
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
# Flask keep-alive (Render требует web-сервис)
# ────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Rocket Hunter (Early Pump Bot) is running!"

@app.route('/ping')
def ping():
    return "pong"

# ────────────────────────────────────────────────
# Константы
# ────────────────────────────────────────────────
TIMEFRAME = '1h'
INTERVAL_SECONDS = 900
MODEL_FILE = 'catboost_pump_model.cbm'

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.58
SIGNAL_LIFETIME = 10800  # 3 часа

VOLUME_SURGE = 2.35
SQUEEZE_FACTOR = 1.25
VOLUME_CREEP = 1.45

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


# ────────────────────────────────────────────────
# Данные и фичи (с squeeze + volume creep)
# ────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, limit: int = 1500):
    try:
        time.sleep(1.1)
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

    df['ema200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2.0)
    df['bb_lower'] = bb.iloc[:, 0]
    df['bb_upper'] = bb.iloc[:, 2]
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(25).mean()

    # === СЖАТИЕ + НАКОПЛЕНИЕ ===
    df['bb_width_min'] = df['bb_width'].rolling(40).min()
    df['is_squeeze'] = df['bb_width'] <= (df['bb_width_min'] * SQUEEZE_FACTOR)
    df['volume_sma'] = df['volume'].rolling(30).mean()
    df['volume_trend'] = df['volume'] / df['volume_sma'].shift(8)

    return df.dropna()


# ────────────────────────────────────────────────
# Модель
# ────────────────────────────────────────────────
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    print("Обучение модели на ранних пампах...")
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
            df['target'] = (df['price_change'].shift(-1) > 0.009).astype(int)
            all_data.append(df)
        except:
            continue

    df_all = pd.concat(all_data).dropna()
    X = df_all[FEATURES]
    y = df_all['target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=800, depth=7, learning_rate=0.06, verbose=0)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"Модель обучена | Accuracy: {acc:.4f}")
    model.save_model(MODEL_FILE)
    return model


def get_market_data(symbol):
    try:
        ticker = futures_exchange.fetch_ticker(symbol)
        return ticker['last'], ticker.get('percentage', 0), round(ticker.get('quoteVolume', 0) / 1_000_000, 1)
    except:
        return 0.0, 0.0, 0.0


def create_chart(pair: str, entry_price: float):
    df = fetch_ohlcv(pair)
    if df.empty: return None
    df = add_features(df)
    if df.empty: return None

    tp1 = round(entry_price * 1.055, 6)
    tp2 = round(entry_price * 1.12, 6)
    avg = round(entry_price * 0.935, 6)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.plot(df['timestamp'], df['close'], color='#00ff9d', linewidth=2, label='Цена')
    ax.plot(df['timestamp'], df['ema200'], color='#ff4444', linewidth=1.8, label='EMA200')
    ax.axhline(entry_price, color='white', linestyle='--', label='Вход')
    ax.axhline(tp1, color='#00ff00', label='Цель 1')
    ax.axhline(tp2, color='#00cc00', label='Цель 2')
    ax.axhline(avg, color='orange', linestyle='--', label='Усреднение -6.5%')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15)
    ax.set_title(f'{pair} — РАННИЙ ПАМП (LONG)', color='white', fontsize=14)
    ax.legend(loc='upper left')
    ax.tick_params(colors='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


def build_signal_text(pair: str, price: float, prob: float, vol_m: float, change: float, volume_ratio: float):
    coin = pair.split('/')[0].replace(':USDT', '')
    strength = "СИЛЬНЫЙ" if prob > 0.85 else "СРЕДНИЙ" if prob > 0.75 else "СЛАБЫЙ"
    fires = "🚀🚀🚀" if prob > 0.85 else "🚀🚀" if prob > 0.75 else "🚀"

    text = f"""🟢 {coin} {fires} {strength}
x200 / {round(price * 200 * 50)}$ / {vol_m}M / {change:+.4f}%

Trade: Mexc Futures

Направление: РАННИЙ ПАМП (с MM)
Действие: LONG

Текущая цена: {price:.8f}
Цель 1: {round(price * 1.055, 8):.8f}
Цель 2: {round(price * 1.12, 8):.8f}
Усреднение: {round(price * 0.935, 8):.8f} (-6.5%)

Уверенность: {int(prob * 100)}%
Объёмный всплеск: x{round(volume_ratio, 1)} (MM уже в деле!)
Сила сигнала: {int(prob * 100 - 15)}/100"""

    text += f"""

Проверь ликвидации:
https://www.tradingview.com/chart/?symbol=MEXC%3A{coin}USDT.P"""

    return text


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return
    row = df.iloc[-1]

    # Фильтры "тайного набора"
    if not row['is_squeeze']:
        print(f"Пропуск {pair} — нет squeeze")
        return
    if row['volume_trend'] < VOLUME_CREEP:
        print(f"Пропуск {pair} — volume_creep {row['volume_trend']:.2f} < {VOLUME_CREEP}")
        return
    if row['volume_ratio'] < VOLUME_SURGE:
        print(f"Пропуск {pair} — volume_ratio {row['volume_ratio']:.1f}x < {VOLUME_SURGE}")
        return
    if row['rsi'] > 67:
        print(f"Пропуск {pair} — RSI {row['rsi']:.1f} > 67")
        return

    text = build_signal_text(pair, price, prob, vol_m, change, row['volume_ratio'])
    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"🚀 ПАМП-сигнал → {pair} ({int(prob*100)}%)")

        ACTIVE_SIGNALS.append({
            'pair': pair,
            'entry_price': price,
            'avg_price': round(price * 0.935, 8),
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"Ошибка отправки {pair}: {e}")


def check_expired_signals():
    global ACTIVE_SIGNALS
    now = time.time()
    to_remove = []
    for s in ACTIVE_SIGNALS:
        if now - s['timestamp'] > SIGNAL_LIFETIME:
            try:
                price, _, _ = get_market_data(s['pair'])
                if price > s['entry_price']:
                    msg = f"✅ {s['pair']} отработал! Цена выше входа ({price:.8f})"
                else:
                    msg = f"⚠️ {s['pair']} тайм-аут. Закрывай на {s.get('avg_price', s['entry_price'])}"
                bot.send_message(CHAT_ID, msg)
            except:
                pass
            to_remove.append(s)
    ACTIVE_SIGNALS = [s for s in ACTIVE_SIGNALS if s not in to_remove]


def update_pairs_list():
    global PAIRS
    try:
        markets = futures_exchange.load_markets(reload=True)
        futures_pairs = [s for s, m in markets.items() if m.get('swap') and 'USDT' in s and m.get('active')]
        PAIRS[:] = sorted(futures_pairs, key=lambda s: float(markets[s].get('info', {}).get('quoteVolume', 0) or 0), reverse=True)[:800]
        print(f"Загружено {len(PAIRS)} пар")
    except Exception as e:
        print(f"Ошибка обновления пар: {e}")


def main_loop():
    model = load_or_train_model()
    last_retrain = time.time()

    bot.send_message(CHAT_ID, f"🚀 Rocket Hunter запущен на Render | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    while True:
        update_pairs_list()
        check_expired_signals()

        for pair in PAIRS[:150]:
            try:
                df = fetch_ohlcv(pair)
                if len(df) < MIN_DATA_LENGTH: continue
                df = add_features(df)
                if df.empty: continue

                row = df.iloc[-1]
                feats = row[FEATURES].values.reshape(1, -1)
                prob = model.predict_proba(feats)[0][1]

                print(f"{pair:20} prob = {prob:.4f}")

                if prob > PROBABILITY_THRESHOLD:
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                pass
            time.sleep(1.3)

        if time.time() - last_retrain > 12 * 3600:
            model = load_or_train_model()
            last_retrain = time.time()

        time.sleep(INTERVAL_SECONDS)


# ────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────
if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
