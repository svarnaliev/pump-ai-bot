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
# Константы — ослабленные для ранних пампов
# ────────────────────────────────────────────────
TIMEFRAME = '1h'
INTERVAL_SECONDS = 900
MODEL_FILE = 'catboost_pump_model.cbm'
LAST_INDEX_FILE = 'last_pair_index.txt'  # файл для сохранения прогресса

MIN_DATA_LENGTH = 60
PROBABILITY_THRESHOLD = 0.52
SIGNAL_LIFETIME = 10800

VOLUME_SURGE = 1.85
SQUEEZE_FACTOR = 1.45
VOLUME_CREEP = 1.25

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
# Данные и фичи
# ────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, limit: int = 1500):
    try:
        time.sleep(0.9)  # уменьшил задержку, чтобы быстрее сканировать все пары
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
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(25).mean()

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
        print("Загружаем существующую модель...")
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    print("Обучение новой модели...")
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
        except Exception as e:
            print(f"Ошибка обучения на {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("Нет данных для обучения!")

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
    ax.plot(df['timestamp'], df['close'], color='#00ff9d', linewidth=2)
    ax.plot(df['timestamp'], df['ema200'], color='#ff4444', linewidth=1.8)
    ax.axhline(entry_price, color='white', linestyle='--', label='Вход')
    ax.axhline(tp1, color='#00ff00', label='Цель 1')
    ax.axhline(tp2, color='#00cc00', label='Цель 2')
    ax.axhline(avg, color='orange', linestyle='--', label='Усреднение')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.15)
    ax.set_title(f'{pair} — РАННИЙ ПАМП', color='white', fontsize=14)
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
Объёмный всплеск: x{round(volume_ratio, 1)}
Сила сигнала: {int(prob * 100 - 15)}/100"""

    return text


def send_signal(pair: str, price: float, prob: float, vol_m: float, change: float):
    df = fetch_ohlcv(pair)
    if df.empty: return
    df = add_features(df)
    if df.empty: return
    row = df.iloc[-1]

    reject = []
    if not row['is_squeeze']: reject.append("нет squeeze")
    if row['volume_trend'] < VOLUME_CREEP: reject.append(f"volume_creep {row['volume_trend']:.2f}")
    if row['volume_ratio'] < VOLUME_SURGE: reject.append(f"volume_ratio {row['volume_ratio']:.1f}x")
    if row['rsi'] > 72: reject.append(f"RSI {row['rsi']:.1f}")

    if reject:
        print(f"  Пропуск {pair} (prob={prob:.4f}): " + "; ".join(reject))
        return

    text = build_signal_text(pair, price, prob, vol_m, change, row['volume_ratio'])
    buf = create_chart(pair, price)

    try:
        bot.send_photo(chat_id=CHAT_ID, photo=buf, caption=text)
        print(f"🚀 СИГНАЛ ОТПРАВЛЕН → {pair} (prob={prob:.4f})")
        ACTIVE_SIGNALS.append({'pair': pair, 'entry_price': price, 'avg_price': round(price * 0.935, 8), 'timestamp': time.time()})
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
                msg = f"✅ {s['pair']} отработал!" if price > s['entry_price'] else f"⚠️ {s['pair']} тайм-аут"
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
    try:
        with open(LAST_INDEX_FILE, 'w') as f:
            f.write(str(idx))
    except Exception as e:
        print(f"Ошибка сохранения индекса: {e}")


def main_loop():
    model = load_or_train_model()
    last_retrain = time.time()

    bot.send_message(CHAT_ID, f"🚀 Rocket Hunter запущен (все пары + resume) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0
    last_self_ping = time.time()

    while True:
        iteration += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"[{now_str}] Итерация {iteration} | всего пар: {len(PAIRS)}")

        update_pairs_list()
        check_expired_signals()

        start_idx = load_last_index()
        print(f"[{now_str}] Продолжаем с индекса {start_idx} ({PAIRS[start_idx] if start_idx < len(PAIRS) else 'конец'})")

        scanned = 0
        high_prob_count = 0

        for i, pair in enumerate(PAIRS[start_idx:]):
            scanned += 1
            try:
                df = fetch_ohlcv(pair)
                if len(df) < MIN_DATA_LENGTH:
                    print(f"  {pair:20} → мало данных")
                    continue
                df = add_features(df)
                if df.empty:
                    print(f"  {pair:20} → фичи не посчитались")
                    continue

                row = df.iloc[-1]
                prob = model.predict_proba(row[FEATURES].values.reshape(1, -1))[0][1]

                print(f"  {pair:20} → prob={prob:.4f} | RSI={row['rsi']:.1f} | squeeze={row['is_squeeze']} | v_ratio={row['volume_ratio']:.1f} | v_trend={row['volume_trend']:.2f}")

                if prob > PROBABILITY_THRESHOLD:
                    high_prob_count += 1
                    print(f"  >>> Высокая вероятность {pair} ({prob:.4f})")
                    price, ch, vm = get_market_data(pair)
                    send_signal(pair, price, prob, vm, ch)

            except Exception as e:
                print(f"  {pair} → ошибка: {type(e).__name__}")

            # Сохраняем прогресс после каждой пары
            current_idx = start_idx + i + 1
            save_last_index(current_idx)

            time.sleep(0.9)  # задержка между запросами

        print(f"[{now_str}] Итерация завершена | просканировано {scanned} | высокая prob: {high_prob_count}")

        # Self-ping
        if time.time() - last_self_ping > 600:
            try:
                r = requests.get("https://pump-ai-bot.onrender.com/ping", timeout=30)
                print(f"[{now_str}] Self-ping → {r.status_code} ({r.text})")
            except Exception as e:
                print(f"[{now_str}] Self-ping ошибка: {e}")
            last_self_ping = time.time()

        print(f"[{now_str}] Пауза {INTERVAL_SECONDS} сек...\n")
        time.sleep(INTERVAL_SECONDS)


# ────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────
if __name__ == '__main__':
    update_pairs_list()
    threading.Thread(target=main_loop, daemon=True).start()

    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
