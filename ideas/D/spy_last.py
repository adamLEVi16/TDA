"""Latest cached SPY close (dividend-adjusted, equals raw close for the latest day) -> cash needed to secure one at-the-money put."""
from vrp_common import yf_close
s = yf_close("SPY"); print(f"SPY close {s.index[-1].date()}: {s.iloc[-1]:.2f}; one ATM cash-secured put (100 shares) needs ~${100*s.iloc[-1]:,.0f}")
