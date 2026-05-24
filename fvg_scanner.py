"""
FVG 컨플루언스 스캐너 v3 (점수제)
================================
조건마다 점수를 부여해서 부분적으로만 부합하는 셋업도 알림.
완벽한 셋업은 만점에 가깝게, 일부만 맞으면 낮은 점수.

점수 배분 (100점 만점):
  20점 — 진입 정확도 (현재가가 피보 레벨에 얼마나 가까운지)
  10점 — 타겟이 피보 레벨에 위치
  20점 — 타겟이 주봉 FVG 내부
  15점 — 타겟이 일봉 FVG 내부
  15점 — 타겟이 4h 매물대 공백 (LVN) 내부
  20점 — R:R 비율 (3.0에서 만점)

환경변수:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TOP_N            상위 종목 수 (기본 40)
  MIN_SCORE        최소 점수 (기본 50)
  MAX_ALERTS       최대 알림 수 (기본 5)
  EXCHANGE         bybit | binance | okx (기본 bybit)
"""

import os
import io
import asyncio
import traceback
from datetime import datetime

import ccxt.async_support as ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests


# ─── 설정 ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
TOP_N = int(os.environ.get('TOP_N', '40'))
MIN_SCORE = float(os.environ.get('MIN_SCORE', '50'))
MAX_ALERTS = int(os.environ.get('MAX_ALERTS', '5'))
EXCHANGE = os.environ.get('EXCHANGE', 'bybit').lower()

KEY_FIBS = (0.382, 0.5, 0.618, 0.786)
ENTRY_MAX_PROXIMITY = 0.05  # 진입가 후보의 최대 허용 거리 (피보에서 5%)


# ─── 기본 분석 ────────────────────────────────────────
def detect_fvgs(candles):
    fvgs = []
    for i in range(2, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        if c1[2] < c3[3]:
            fvgs.append({'type': 'bull', 'top': c3[3], 'bottom': c1[2],
                         'index': i - 1, 'size': (c3[3] - c1[2]) / c1[2]})
        elif c1[3] > c3[2]:
            fvgs.append({'type': 'bear', 'top': c1[3], 'bottom': c3[2],
                         'index': i - 1, 'size': (c1[3] - c3[2]) / c3[2]})
    return fvgs


def filter_unfilled(fvgs, candles):
    result = []
    for fvg in fvgs:
        filled = False
        for i in range(fvg['index'] + 1, len(candles)):
            c = candles[i]
            if fvg['type'] == 'bull' and c[3] <= fvg['bottom']:
                filled = True; break
            if fvg['type'] == 'bear' and c[2] >= fvg['top']:
                filled = True; break
        if not filled:
            result.append(fvg)
    return result


def build_volume_profile(candles, bins=60):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    h, l = max(highs), min(lows)
    bin_size = (h - l) / bins
    if bin_size == 0:
        return []
    profile = [{'low': l + i * bin_size,
                'high': l + (i + 1) * bin_size,
                'vol': 0.0} for i in range(bins)]
    for c in candles:
        c_range = c[2] - c[3]
        if c_range == 0:
            idx = min(bins - 1, max(0, int((c[4] - l) / bin_size)))
            profile[idx]['vol'] += c[5]
            continue
        for b in profile:
            ol = max(b['low'], c[3])
            oh = min(b['high'], c[2])
            if oh > ol:
                b['vol'] += c[5] * (oh - ol) / c_range
    return profile


def find_lvns(profile, ratio=0.3):
    if not profile:
        return []
    avg = sum(b['vol'] for b in profile) / len(profile)
    return [b for b in profile if 0 < b['vol'] < avg * ratio]


# ─── 점수제 분석 ──────────────────────────────────────
def analyze(symbol, c4h, c1d, c1w):
    if len(c4h) < 60 or len(c1d) < 10:
        return None
    current = c4h[-1][4]

    # 1. 4h 스윙
    recent = c4h[-60:]
    swing_high = max(c[2] for c in recent)
    swing_low = min(c[3] for c in recent)
    if swing_high - swing_low <= 0:
        return None
    diff = swing_high - swing_low
    fib = {k: swing_low + k * diff for k in KEY_FIBS}

    # 2. 가장 가까운 피보 = 진입 후보
    entry_candidates = [(k, v, abs(current - v) / current) for k, v in fib.items()]
    entry_level, entry_price, entry_prox = min(entry_candidates, key=lambda x: x[2])

    # 너무 멀면 의미 없음
    if entry_prox > ENTRY_MAX_PROXIMITY:
        return None

    # 3. 방향
    direction = 'long' if entry_level <= 0.5 else 'short'

    # 4. 필요한 데이터
    daily_fvgs = filter_unfilled(detect_fvgs(c1d), c1d)
    weekly_fvgs = filter_unfilled(detect_fvgs(c1w), c1w) if c1w and len(c1w) >= 3 else []
    profile = build_volume_profile(c4h, 60)
    lvns = find_lvns(profile, 0.3)

    # 5. 모든 타겟 후보 평가, 최고 점수 선택
    best = None
    for k, v in fib.items():
        if k == entry_level:
            continue
        # 방향 체크
        if direction == 'long' and v <= current:
            continue
        if direction == 'short' and v >= current:
            continue

        # 손절 계산
        short_recent = c4h[-20:]
        if direction == 'long':
            stop_price = min(c[3] for c in short_recent) * 0.997
            risk = current - stop_price
            reward = v - current
        else:
            stop_price = max(c[2] for c in short_recent) * 1.003
            risk = stop_price - current
            reward = current - v

        if risk <= 0 or reward <= 0:
            continue
        rr = reward / risk

        # 타겟 조건 평가
        in_daily = any(f['bottom'] * 0.99 <= v <= f['top'] * 1.01 for f in daily_fvgs)
        in_weekly = any(f['bottom'] * 0.99 <= v <= f['top'] * 1.01 for f in weekly_fvgs)
        in_lvn = any(l['low'] * 0.99 <= v <= l['high'] * 1.01 for l in lvns)

        # ── 점수 계산 (100점 만점) ──
        # 진입 정확도: 0% 이격이면 20점, 5%면 0점
        score_entry = max(0, (1 - entry_prox / ENTRY_MAX_PROXIMITY)) * 20
        # 타겟이 피보 레벨에 있음 (자동)
        score_target = 10
        # 주봉 FVG
        score_weekly = 20 if in_weekly else 0
        # 일봉 FVG
        score_daily = 15 if in_daily else 0
        # LVN
        score_lvn = 15 if in_lvn else 0
        # R:R (3.0에서 만점, 비례)
        score_rr = min(20, (rr / 3.0) * 20)

        total = score_entry + score_target + score_weekly + score_daily + score_lvn + score_rr

        if best is None or total > best['score']:
            best = {
                'level': k, 'price': v,
                'in_daily': in_daily, 'in_weekly': in_weekly, 'in_lvn': in_lvn,
                'stop_price': stop_price, 'rr': rr,
                'risk': risk, 'reward': reward,
                'score': total,
                'score_breakdown': {
                    'entry': score_entry,
                    'target': score_target,
                    'weekly': score_weekly,
                    'daily': score_daily,
                    'lvn': score_lvn,
                    'rr': score_rr,
                },
            }

    if best is None or best['score'] < MIN_SCORE:
        return None

    return {
        'symbol': symbol,
        'direction': direction,
        'current': current,
        'entry_level': entry_level,
        'entry_price': entry_price,
        'entry_prox': entry_prox,
        'target_level': best['level'],
        'target_price': best['price'],
        'stop_price': best['stop_price'],
        'rr': best['rr'],
        'risk_pct': abs(best['risk'] / current) * 100,
        'reward_pct': abs(best['reward'] / current) * 100,
        'in_daily': best['in_daily'],
        'in_weekly': best['in_weekly'],
        'in_lvn': best['in_lvn'],
        'score': best['score'],
        'breakdown': best['score_breakdown'],
        'swing_high': swing_high,
        'swing_low': swing_low,
        'fib': fib,
        'candles_4h': c4h,
        'daily_fvgs': daily_fvgs,
        'weekly_fvgs': weekly_fvgs,
        'lvns': lvns,
    }


# ─── 포매팅 / 차트 ────────────────────────────────────
def fmt_price(p):
    if p >= 1000: return f"{p:.2f}"
    if p >= 1:    return f"{p:.4f}"
    if p >= 0.01: return f"{p:.5f}"
    return f"{p:.7f}"


def render_chart(r):
    candles = r['candles_4h'][-60:]
    fig, ax = plt.subplots(figsize=(11, 6.8), facecolor='#0a0908')
    ax.set_facecolor('#0a0908')

    # 일봉 FVG
    for fvg in r['daily_fvgs']:
        if not (r['swing_low'] * 0.95 <= fvg['bottom'] <= r['swing_high'] * 1.05):
            continue
        color = '#7c8aff' if fvg['type'] == 'bull' else '#ff7ca8'
        ax.add_patch(patches.Rectangle(
            (0, fvg['bottom']), len(candles), fvg['top'] - fvg['bottom'],
            facecolor=color, alpha=0.12,
            edgecolor=color, linestyle=':', linewidth=0.7,
        ))
        ax.text(len(candles) - 1, (fvg['top'] + fvg['bottom']) / 2,
                ' 1D FVG', color=color, fontsize=8, va='center', alpha=0.85)

    # 4h LVN
    for lvn in r['lvns']:
        if not (r['swing_low'] <= lvn['low'] and lvn['high'] <= r['swing_high']):
            continue
        ax.add_patch(patches.Rectangle(
            (0, lvn['low']), len(candles), lvn['high'] - lvn['low'],
            facecolor='#d4a64a', alpha=0.08, edgecolor='none',
        ))

    # 피보 레벨
    for k, v in r['fib'].items():
        ax.axhline(v, color='#d4a64a', alpha=0.7,
                   linestyle='--', linewidth=0.6)
        ax.text(0.5, v, f' {k}', color='#d4a64a',
                fontsize=8, va='bottom', alpha=0.85, fontweight='bold')

    # 캔들
    for i, c in enumerate(candles):
        color = '#5ec98a' if c[4] >= c[1] else '#e57373'
        ax.plot([i, i], [c[3], c[2]], color=color, linewidth=0.9)
        body_h = max(abs(c[4] - c[1]), (c[2] - c[3]) * 0.005)
        ax.add_patch(patches.Rectangle(
            (i - 0.35, min(c[1], c[4])), 0.7, body_h,
            facecolor=color, edgecolor=color,
        ))

    # 진입/타겟/손절
    def line(price, color, label, lw=1.2):
        ax.axhline(price, color=color, linewidth=lw, alpha=0.9)
        ax.text(len(candles) + 0.5, price, f' {label} {fmt_price(price)}',
                color=color, fontsize=9, fontweight='bold', va='center')

    line(r['entry_price'], '#7ee0ff', 'ENTRY')
    line(r['target_price'], '#d4a64a', 'TARGET', lw=1.5)
    line(r['stop_price'], '#e57373', 'STOP')
    ax.axhline(r['current'], color='#f5f1e8', linestyle=':', linewidth=0.7, alpha=0.6)

    # 수익/위험 음영
    p_lo, p_hi = sorted([r['entry_price'], r['target_price']])
    ax.add_patch(patches.Rectangle(
        (0, p_lo), len(candles), p_hi - p_lo,
        facecolor='#5ec98a', alpha=0.04, edgecolor='none',
    ))
    p_lo, p_hi = sorted([r['entry_price'], r['stop_price']])
    ax.add_patch(patches.Rectangle(
        (0, p_lo), len(candles), p_hi - p_lo,
        facecolor='#e57373', alpha=0.05, edgecolor='none',
    ))

    arrow = '▲' if r['direction'] == 'long' else '▼'
    color_dir = '#5ec98a' if r['direction'] == 'long' else '#e57373'
    title = f"{r['symbol']}  {arrow} {r['direction'].upper()}  ·  점수 {r['score']:.0f}/100  ·  R:R {r['rr']:.2f}"
    ax.set_title(title, color=color_dir, fontsize=13, loc='left',
                 pad=10, fontweight='bold')
    ax.tick_params(colors='#8a8478', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#2a2620')
    ax.grid(color='#1a1814', linewidth=0.5, alpha=0.6)
    ax.set_xlim(-1, len(candles) + 12)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0a0908')
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── 텔레그램 ─────────────────────────────────────────
def tg_message(text):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'Markdown'},
        timeout=15,
    )


def tg_photo(buf, caption):
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        files={'photo': ('chart.png', buf, 'image/png')},
        data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'},
        timeout=30,
    )


def format_caption(r):
    bias_emoji = '🟢' if r['direction'] == 'long' else '🔴'
    bias_label = 'LONG' if r['direction'] == 'long' else 'SHORT'
    b = r['breakdown']

    # 등급
    if r['score'] >= 85:
        grade = '⭐⭐⭐ 완벽'
    elif r['score'] >= 70:
        grade = '⭐⭐ 우수'
    elif r['score'] >= 55:
        grade = '⭐ 양호'
    else:
        grade = '△ 부분'

    return (
        f"🎯 *{r['symbol']}*  ·  *{r['score']:.0f}/100*  {grade}\n"
        f"{bias_emoji} *{bias_label}*  ·  R:R *{r['rr']:.2f}*\n\n"
        f"`진입  {fmt_price(r['entry_price'])}  (피보 {r['entry_level']})`\n"
        f"`타겟  {fmt_price(r['target_price'])}  (피보 {r['target_level']})`\n"
        f"`손절  {fmt_price(r['stop_price'])}`\n"
        f"수익 *+{r['reward_pct']:.2f}%*  ·  위험 *-{r['risk_pct']:.2f}%*\n\n"
        f"*점수 세부:*\n"
        f"`{'✓' if b['entry'] >= 15 else '△'} 진입 정확도     {b['entry']:>4.0f}/20`\n"
        f"`{'✓' if b['weekly']  > 0 else '✗'} 주봉 FVG       {b['weekly']:>4.0f}/20`\n"
        f"`{'✓' if b['daily']   > 0 else '✗'} 일봉 FVG       {b['daily']:>4.0f}/15`\n"
        f"`{'✓' if b['lvn']     > 0 else '✗'} 4h 매물대공백  {b['lvn']:>4.0f}/15`\n"
        f"`{'✓' if b['rr']     >= 15 else '△'} R:R 비율       {b['rr']:>4.0f}/20`\n"
        f"`✓ 타겟 피보위치   {b['target']:>4.0f}/10`"
    )


# ─── 메인 스캔 ────────────────────────────────────────
async def fetch_all_tfs(exchange, symbol):
    return await asyncio.gather(
        exchange.fetch_ohlcv(symbol, '4h', limit=200),
        exchange.fetch_ohlcv(symbol, '1d', limit=100),
        exchange.fetch_ohlcv(symbol, '1w', limit=50),
        return_exceptions=True,
    )


async def scan():
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] {EXCHANGE.upper()} 스캔 시작...")
    print(f"  상위 {TOP_N}개 · 최소 점수 {MIN_SCORE}/100")

    if EXCHANGE == 'bybit':
        exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
        suffix = '/USDT:USDT'
    elif EXCHANGE == 'binance':
        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        suffix = '/USDT:USDT'
    elif EXCHANGE == 'okx':
        exchange = ccxt.okx({'options': {'defaultType': 'swap'}})
        suffix = '/USDT:USDT'
    else:
        raise ValueError(f"Unsupported exchange: {EXCHANGE}")

    try:
        tickers = await exchange.fetch_tickers()
        pairs = [(s, t) for s, t in tickers.items()
                 if s.endswith(suffix) and t.get('quoteVolume')]
        pairs.sort(key=lambda x: x[1]['quoteVolume'] or 0, reverse=True)
        top = pairs[:TOP_N]
        print(f"  → 상위 {len(top)}개 분석 중...")

        sem = asyncio.Semaphore(5)

        async def process(symbol):
            async with sem:
                try:
                    c4h, c1d, c1w = await fetch_all_tfs(exchange, symbol)
                    if any(isinstance(x, Exception) for x in (c4h, c1d, c1w)):
                        return None
                    return analyze(symbol, c4h, c1d, c1w if c1w else [])
                except Exception as e:
                    print(f"     {symbol}: {e}")
                    return None

        outputs = await asyncio.gather(*[process(s) for s, _ in top])
        results = [r for r in outputs if r]
        results.sort(key=lambda r: -r['score'])
        print(f"  → 조건 만족 {len(results)}개")

        if not results:
            print("  → 알림 없음")
            return

        header = (
            f"*🔍 FVG 컨플루언스 스캐너*\n"
            f"{datetime.utcnow():%Y-%m-%d %H:%M} UTC · {EXCHANGE.upper()}\n"
            f"`{len(results)}`개 셋업 발견 · 상위 `{min(MAX_ALERTS, len(results))}`개 전송\n"
            f"_점수 최소 {MIN_SCORE:.0f}/100 이상_"
        )
        tg_message(header)

        for r in results[:MAX_ALERTS]:
            try:
                chart = render_chart(r)
                tg_photo(chart, format_caption(r))
                await asyncio.sleep(1.5)
            except Exception as e:
                print(f"     알림 실패 {r['symbol']}: {e}")

    finally:
        await exchange.close()


if __name__ == '__main__':
    try:
        asyncio.run(scan())
    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            tg_message(f"⚠️ *스캐너 오류*\n```\n{err[-1500:]}\n```")
        except Exception:
            pass
        raise
