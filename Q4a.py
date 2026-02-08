import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import newton
from pathlib import Path

SORTED_FILES = [
    "Short-term_CA135087R978.csv", "Short-term_CA135087S547.csv",
    "Short-term_CA135087T461.csv", "Short-term_CA135087T958.csv",
    "Short-term_CA135087Q491.csv", "Long-term_CA135087Q988.csv",
    "Long-term_CA135087R895.csv", "Long-term_CA135087S471.csv",
    "Long-term_CA135087T388.csv", "Long-term_CA135087T792.csv"
]
DATA_DIR = Path(__file__).resolve().parent / "data"


def accrued_interest(annual_coupon_rate, payments_per_year, last_coupon_date, settlement_date):
    """Accrued interest per 100 par value"""
    coupon_payment = 100 * annual_coupon_rate / payments_per_year
    days_since_last = (settlement_date - last_coupon_date).days
    days_in_period = (last_coupon_date + timedelta(days=int(365/payments_per_year)) - last_coupon_date).days
    return coupon_payment * (days_since_last / days_in_period)

def ytm(price_pct_dirty, coupon_rate, years_to_maturity, face_value=100, freq=2):
    """Solve YTM (dirty price input) using Newton method"""
    price_dirty = price_pct_dirty/100 * face_value
    coupon = coupon_rate * face_value / freq
    n_periods = int(round(years_to_maturity*freq))

    def f(y):
        return sum(coupon/(1+y/freq)**t for t in range(1, n_periods+1)) \
               + face_value/(1+y/freq)**n_periods - price_dirty

    return newton(f, x0=0.03)  # initial guess 3%

# Load all bond files and compute daily YTM
print("SCRIPT:", __file__)
print("DATA_DIR:", DATA_DIR)
print("DATA_DIR exists:", DATA_DIR.exists())
print("First file exists:", (DATA_DIR / SORTED_FILES[0]).exists(), DATA_DIR / SORTED_FILES[0])

ytm_dict = {}
for file in SORTED_FILES:
    df = pd.read_csv(DATA_DIR / file)
    df['date'] = pd.to_datetime(df['date'])
    maturity = pd.to_datetime(df['Maturity Date'].iloc[0])
    coupon_rate = float(str(df['Coupon'].iloc[0]).replace('%',''))/100.0
    payments_per_year = int(df['No. of Payments per Year'].iloc[0])
    next_coupon_date = pd.to_datetime(df['Coupon Payment Date'].iloc[0])

    # Infer last coupon date
    last_coupon_date = next_coupon_date - pd.DateOffset(months=12/payments_per_year)

    dirty_prices = []
    years_to_maturity = (maturity - df['date']).dt.days / 365.0

    for idx, row in df.iterrows():
        settlement_date = row['date']
        # Adjust last_coupon_date if settlement crosses coupon date boundary
        # e.g., if settlement < next_coupon, keep; else shift to newer period
        lc_date = last_coupon_date
        nc_date = next_coupon_date
        if settlement_date >= next_coupon_date:
            lc_date = next_coupon_date
            nc_date = next_coupon_date + pd.DateOffset(months=12/payments_per_year)

        accr_int = accrued_interest(coupon_rate, payments_per_year, lc_date.to_pydatetime(), settlement_date.to_pydatetime())
        dirty_price = row['Close Price (% of par)'] + accr_int
        dirty_prices.append(dirty_price)

    df['Dirty Price (% of par)'] = dirty_prices
    df['ytm'] = [
        ytm(dp, coupon_rate, t) for dp, t in zip(df['Dirty Price (% of par)'], years_to_maturity)
    ]
    ytm_dict[file] = df[['date','ytm']]

# Plot yield curves
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
        plt.plot(
            maturities[:len(curve)],
            curve,
            alpha=0.5,
            label=date.strftime('%Y-%m-%d')
        )

plt.xlabel("Maturity (Years)")
plt.ylabel("Yield to Maturity (%)")
plt.title("Government of Canada – Daily Yield Curves (Dirty Price Based)")
plt.grid(True)
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()

# Build table for export
ytm_df = pd.DataFrame({'date': unique_dates})

for fname in SORTED_FILES:
    bond_df = ytm_dict[fname].copy()
    bond_df.set_index('date', inplace=True)
    bond_label = fname.split('_')[-1].replace('.csv', '')
    ytm_df[bond_label] = ytm_df['date'].map(bond_df['ytm'])

ytm_df.set_index('date', inplace=True)
ytm_df.to_csv("ytm_table.csv", float_format="%.6f")
