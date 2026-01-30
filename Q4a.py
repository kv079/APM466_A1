import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import newton

SORTED_FILES = [
    "Short-term_CA135087R978.csv", "Short-term_CA135087S547.csv",
    "Short-term_CA135087T461.csv", "Short-term_CA135087T958.csv",
    "Short-term_CA135087Q491.csv", "Long-term_CA135087Q988.csv",
    "Long-term_CA135087R895.csv", "Long-term_CA135087S471.csv",
    "Long-term_CA135087T388.csv", "Long-term_CA135087T792.csv"
]

def ytm(price_pct, coupon_rate, years_to_maturity, face_value=100, freq=2):
    """
    Solve YTM using Newton method.
    """
    price = price_pct/100 * face_value
    coupon = coupon_rate * face_value / freq
    n_periods = int(years_to_maturity*freq)

    def f(y):
        return sum(coupon/(1+y/freq)**t for t in range(1, n_periods+1)) \
               + face_value/(1+y/freq)**n_periods - price

    return newton(f, x0=0.03)  # initial guess 3%

# Load all bond files and compute daily YTM
ytm_dict = {}
for file in SORTED_FILES:
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    maturity = pd.to_datetime(df['Maturity Date'].iloc[0])
    coupon = float(str(df['Coupon'].iloc[0]).replace('%',''))/100.0
    years_to_maturity = (maturity - df['date']).dt.days / 365.0
    df['ytm'] = [
        ytm(price, coupon, t) for price, t in zip(df['Close Price (% of par)'], years_to_maturity)
    ]
    ytm_dict[file] = df[['date','ytm']]

plt.figure(figsize=(10,6))
maturities = np.linspace(0.5, 5, 10)  # pseudo maturities for display

# Collect unique dates sorted
unique_dates = sorted(set(pd.concat([d['date'] for d in ytm_dict.values()])))

for date in unique_dates:
    curve = []
    for key in SORTED_FILES:
        row = ytm_dict[key]
        val = row.loc[row['date'] == date, 'ytm']
        if not val.empty:
            curve.append(val.values[0] * 100)
    if curve:
        # format the date nicely for legend
        plt.plot(
            maturities[:len(curve)],
            curve,
            alpha=0.5,
            label=date.strftime('%Y-%m-%d')
        )

plt.xlabel("Maturity (Years)")
plt.ylabel("Yield to Maturity (%)")
plt.title("Government of Canada – Daily Yield Curves (Superimposed)")
plt.grid(True)

# Add legend outside the plot to avoid clutter
plt.legend(
    title="Date",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize='small'
)

plt.tight_layout()
plt.show()
