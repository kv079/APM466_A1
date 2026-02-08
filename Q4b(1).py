import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Bond CSV files sorted by ascending maturity
SORTED_FILES = [
    "Short-term_CA135087R978.csv", "Short-term_CA135087S547.csv",
    "Short-term_CA135087T461.csv", "Short-term_CA135087T958.csv",
    "Short-term_CA135087Q491.csv", "Long-term_CA135087Q988.csv",
    "Long-term_CA135087R895.csv", "Long-term_CA135087S471.csv",
    "Long-term_CA135087T388.csv", "Long-term_CA135087T792.csv"
]

# Parameters
FACE_VALUE = 100
FREQ = 2  # semi-annual payments
TARGET_YEARS = np.arange(1, 6)  # interpolate spot rates at 1Y, 2Y, 3Y, 4Y, 5Y

def accrued_interest(annual_coupon_rate, payments_per_year, last_coupon_date, settlement_date):
    """Accrued interest per 100 par value."""
    coupon_payment = 100 * annual_coupon_rate / payments_per_year
    days_since_last = (settlement_date - last_coupon_date).days
    days_in_period = (last_coupon_date + timedelta(days=int(365/payments_per_year)) - last_coupon_date).days
    return coupon_payment * (days_since_last / days_in_period)

def bootstrap_spot_rates_discrete(prices, coupons, maturities, settle_date):
    """Bootstraps spot rates using discrete semi-annual compounding."""
    spots = []
    t_list = []
    for price, cpn, mat in zip(prices, coupons, maturities):
        t = (mat - settle_date).days / 365.0
        if t <= 0 or pd.isna(price):
            continue
        n = int(round(t * FREQ))
        cpn_payment = cpn * FACE_VALUE / FREQ
        cf_times = np.arange(1, n + 1) / FREQ
        cfs = np.full(n, cpn_payment)
        cfs[-1] += FACE_VALUE
        P = price
        if not spots:
            # First bond, assume single payment at maturity (shortest t)
            s = FREQ * ((FACE_VALUE + cpn_payment) / P - 1)
        else:
            pv_known = 0.0
            for k in range(n - 1):
                t_k = cf_times[k]
                s_k = np.interp(t_k, t_list, spots)
                pv_known += cfs[k] / (1 + s_k / FREQ) ** (FREQ * t_k)
            last_cf = cfs[-1]
            s = FREQ * ((last_cf / (P - pv_known)) ** (1 / (FREQ * t)) - 1)
        t_list.append(t)
        spots.append(s)
    return np.array(t_list), np.array(spots)

# Store dirty prices for all bonds and dates
dirty_price_df = pd.DataFrame()
bond_maturities = []
bond_coupons = []

# Loop through bond files
for fname in SORTED_FILES:
    fpath = DATA_DIR / fname
    df = pd.read_csv(fpath)
    df['date'] = pd.to_datetime(df['date'])
    maturity = pd.to_datetime(df['Maturity Date'].iloc[0])
    coupon_rate = float(str(df['Coupon'].iloc[0]).replace('%', '')) / 100.0
    payments_per_year = int(df['No. of Payments per Year'].iloc[0])
    next_coupon_date = pd.to_datetime(df['Coupon Payment Date'].iloc[0])
    last_coupon_date = next_coupon_date - pd.DateOffset(months=12/payments_per_year)

    # Compute dirty prices
    dirty_prices = []
    for idx, row in df.iterrows():
        settlement_date = row['date']
        lc_date = last_coupon_date
        if settlement_date >= next_coupon_date:
            lc_date = next_coupon_date
        accr_int = accrued_interest(coupon_rate, payments_per_year,
                                    lc_date.to_pydatetime(), settlement_date.to_pydatetime())
        dirty_price = row['Close Price (% of par)'] + accr_int
        dirty_prices.append(dirty_price)
    df['Dirty Price (% of par)'] = dirty_prices

    # Store results
    bond_key = fname.split('_')[-1].replace('.csv', '')
    dirty_price_df[bond_key] = df.set_index('date')['Dirty Price (% of par)']
    bond_maturities.append(maturity)
    bond_coupons.append(coupon_rate)

# Ensure index sorted
dirty_price_df.sort_index(inplace=True)
bond_maturities = pd.to_datetime(bond_maturities)
bond_coupons = np.array(bond_coupons)

# Output DataFrame for interpolated spot curves
spot_interp_df = pd.DataFrame(index=dirty_price_df.index, columns=TARGET_YEARS)

# Plot setup
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(dirty_price_df.index)))

# Loop dates for bootstrapping
for i, current_date in enumerate(dirty_price_df.index):
    prices_today = dirty_price_df.loc[current_date].values
    t_years, spot_rates = bootstrap_spot_rates_discrete(prices_today, bond_coupons, bond_maturities, current_date)
    if len(t_years) < 2:
        continue
    f_interp = interp1d(t_years, spot_rates, kind='linear', fill_value='extrapolate')
    interp_spots = f_interp(TARGET_YEARS)
    spot_interp_df.loc[current_date] = interp_spots * 100  # store in %
    plt.plot(TARGET_YEARS, interp_spots, marker='o', linewidth=1.2, color=colors[i],
                alpha=0.8, label=current_date.strftime('%Y-%m-%d'))

plt.title("Bootstrapped Spot Curves (Discrete, Semiannual)")
plt.xlabel("Maturity (Years)")
plt.ylabel("Spot Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='Date')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("spot_curve_discrete.png", dpi=300)
plt.show()

# Save to CSV
spot_interp_df.to_csv("spot_curve_discrete_interpolated.csv", float_format="%.6f")
print("Bootstrapped spot curves saved to spot_curve_discrete_interpolated.csv")
print("Plot saved to spot_curve_discrete.png")
