from decimal import Decimal, ROUND_HALF_UP
import hashlib

def round_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size is None or tick_size <= 0:
        return value
    q = (value / tick_size).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return (q * tick_size).quantize(tick_size.normalize(), rounding=ROUND_HALF_UP)

def decimals_from_tick(tick_size: Decimal) -> int:
    s = f"{tick_size.normalize():f}"
    return len(s.split('.')[-1]) if '.' in s else 0

def make_idempotency_hash(session_id, candle_id, direction, lpo, slatr, tp, plan_version):
    payload = f"{session_id}|{candle_id}|{direction}|{lpo}|{slatr}|{tp}|{plan_version}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
