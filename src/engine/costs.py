"""Transaction-cost model (per executed order leg).  All rates come from settings."""
from __future__ import annotations


class CostModel:
    def __init__(self, intraday: dict, delivery: dict, multiplier: float = 1.0) -> None:
        self.intraday = intraday
        self.delivery = delivery
        self.multiplier = multiplier

    def leg(self, turnover: float, side: str, intraday: bool) -> float:
        """Total charges for one buy or sell order with the given rupee turnover."""
        r = self.intraday if intraday else self.delivery
        brokerage = turnover * r["brokerage_pct"] / 100
        if r["brokerage_cap"] > 0:
            brokerage = min(brokerage, r["brokerage_cap"])
        buy = side == "buy"
        stt = turnover * (r["stt_buy_pct"] if buy else r["stt_sell_pct"]) / 100
        exchange = turnover * r["exchange_txn_pct"] / 100
        sebi = turnover * r["sebi_pct"] / 100
        stamp = turnover * r["stamp_buy_pct"] / 100 if buy else 0.0
        gst = (brokerage + exchange + sebi) * r["gst_pct"] / 100
        dp = 0.0 if buy else r.get("dp_charge_per_sell", 0.0)
        return (brokerage + stt + exchange + sebi + stamp + gst + dp) * self.multiplier

    def with_multiplier(self, multiplier: float) -> "CostModel":
        return CostModel(self.intraday, self.delivery, multiplier)
