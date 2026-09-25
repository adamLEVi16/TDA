"""Build all industry signals (1927..2024-11 formation dates) and cache them. No portfolio evaluation here."""
import pickle
import time

import data
import engine as E

KS = [3, 5, 10]
t0 = time.time()
d, m = data.industries()
d, m = d.loc[:"2024-12-31"], m.loc[:"2024-12-31"]
form = m.index[m.index <= "2024-11-30"]
sigs = E.build_signals(d, m, KS, form)
print("real graph signals built in %.0fs" % (time.time() - t0))
for key, df in sigs.items():
    print(f"{str(key):<22} {df.index[0]:%Y-%m}..{df.index[-1]:%Y-%m} rows={len(df)} "
          f"mean #assets={df.notna().sum(axis=1).mean():.1f}")
pickle.dump({"sigs": sigs}, open("cache/ind_signals.pkl", "wb"))
