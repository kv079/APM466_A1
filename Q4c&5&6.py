import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------------
# Config: Sorted file list (shortest maturity → longest)
# -------------------------------------------------------------
SORTED_FILES = [
    "Short-term_CA135087R978.csv", "Short-term_CA135087S547.csv",
    "Short-term_CA135087T461.csv", "Short-term_CA135087T958.csv",
    "Short-term_CA135087Q491.csv", "Long-term_CA135087Q988.csv",
    "Long-term_CA135087R895.csv", "Long-term_CA135087S471.csv",
    "Long-term_CA135087T388.csv", "Long-term_CA135087T792.csv"
]

# -------------------------------------------------------------
# Dirty price & cash flow helpers
# -------------------------------------------------------------
def get_last_coupon_date(next_coupon_date, payments_per_year=2):
    """Given the next coupon date, infer the last coupon date."""
    return next_coupon_date - pd.DateOffset(months=12/payments_per_year)

def get_dirty_price(clean, coupon_rate, settlement_date, next_coupon_date, payments_per_year=2):
    """Dirty price = clean price + accrued interest."""
    last_coupon_date = get_last_coupon_date(next_coupon_date, payments_per_year)
    days_since = (settlement_date - last_coupon_date).days
    days_period = (next_coupon_date - last_coupon_date).days
    accrued_int = (coupon_rate * 100 / payments_per_year) * (days_since / days_period)
    return clean + accrued_int

def get_cash_flows(coupon_rate, face_value, settlement_date, maturity_date, payments_per_year=2):
    """Return times (years) and cashflows from settlement date to maturity."""
    times, flows = [], []
    temp_date = maturity_date

    while temp_date > settlement_date:
        t_years = (temp_date - settlement_date).days / 365.0
        amt = coupon_rate * face_value / payments_per_year
        if temp_date == maturity_date:
            amt += face_value
        times.append(t_years)
        flows.append(amt)
        temp_date -= pd.DateOffset(months=12/payments_per_year)
    return np.array(times[::-1]), np.array(flows[::-1])

# -------------------------------------------------------------
# Bootstrapping Spot Curves (Q4b)
# -------------------------------------------------------------
df0 = pd.read_csv(DATA_DIR / SORTED_FILES[0])
df0['date'] = pd.to_datetime(df0['date'])
target_dates = sorted(df0.query("'2026-01-05' <= date <= '2026-01-19'")['date'].unique())

spot_curves = []      # each = np.array of spot rates for 1..5 years
target_years = [1, 2, 3, 4, 5]

for date in target_dates:
    known_T, known_r = [0], [0]
    for fname in SORTED_FILES:
        bond_df = pd.read_csv(DATA_DIR / fname)
        bond_df['date'] = pd.to_datetime(bond_df['date'])

        row = bond_df.query("date == @date")
        if row.empty:
            continue  # skip missing
        row = row.iloc[0]

        maturity = pd.to_datetime(row['Maturity Date'])
        coupon_rate = float(str(row['Coupon']).strip('%')) / 100.0
        payments_per_year = int(row['No. of Payments per Year'])
        next_coupon_date = pd.to_datetime(row['Coupon Payment Date'])
        clean_price = row['Close Price (% of par)']
        dirty_price = get_dirty_price(clean_price, coupon_rate, date, next_coupon_date, payments_per_year)

        times, flows = get_cash_flows(coupon_rate, 100, date, maturity, payments_per_year)

        if len(times) == 0:
            continue  # matured

        if len(known_T) == 1:
            # First bond: flat spot rate for all its periods
            f = lambda r: np.sum(flows * np.exp(-r * times)) - dirty_price
            r_spot = newton(f, 0.04)
        else:
            # Discount earlier flows with known curve
            f_interp = interp1d(known_T, known_r, kind='linear', fill_value='extrapolate')
            pv_known = sum(cf * np.exp(-f_interp(t) * t) for t, cf in zip(times[:-1], flows[:-1]))
            residual = dirty_price - pv_known
            r_spot = -np.log(residual / flows[-1]) / times[-1]

        known_T.append(times[-1])
        known_r.append(r_spot)

    # Interpolate to 1-5 year spot rates
    final_curve = interp1d(known_T, known_r, kind='linear', fill_value='extrapolate')
    spot_curves.append(final_curve(target_years))

# -------------------------------------------------------------
# Q4(c) - 1-Year Forward Curves
# -------------------------------------------------------------
forward_curves = []
for S_curve in spot_curves:
    S_map = dict(zip(target_years, S_curve))
    fwd = [((1 + S_map[1 + k]) ** (1 + k) / (1 + S_map[1])) ** (1 / k) - 1 for k in range(1, 5)]
    forward_curves.append(fwd)

# Plot forward curves
plt.figure(figsize=(9, 6))
colors = plt.cm.plasma(np.linspace(0, 1, len(target_dates)))
for i, curve in enumerate(forward_curves):
    plt.plot(range(2, 6), curve, color=colors[i], marker='s',
             label=target_dates[i].strftime('%Y-%m-%d'))
plt.xlabel("Forward Term (Years)")
plt.ylabel("1-Year Forward Rate")
plt.title("1-Year Forward Curves (Daily Superimposed)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), title="Date")
plt.tight_layout()
plt.savefig("forward_curve.png", dpi=300)
plt.show()

# -------------------------------------------------------------
# Q5 - Covariance Matrices with Dirty Price YTMs
# -------------------------------------------------------------
def ytm(price_pct_dirty, coupon_rate, years_to_maturity, face_value=100, freq=2):
    price = price_pct_dirty / 100 * face_value
    coupon = coupon_rate * face_value / freq
    n_periods = int(round(years_to_maturity * freq))
    def f(y):
        return sum(coupon / (1 + y/freq)**t for t in range(1, n_periods+1)) \
               + face_value / (1 + y/freq)**n_periods - price
    return newton(f, x0=0.03)

# Build YTM DataFrame for 1..5 year maturities
yields_df = pd.DataFrame()
for i, file in enumerate(SORTED_FILES[:5]):
    df = pd.read_csv(DATA_DIR / file)
    df['date'] = pd.to_datetime(df['date'])
    maturity = pd.to_datetime(df['Maturity Date'].iloc[0])
    coupon = float(str(df['Coupon'].iloc[0]).replace('%',''))/100.0
    payments_per_year = int(df['No. of Payments per Year'].iloc[0])
    next_coupon_date = pd.to_datetime(df['Coupon Payment Date'].iloc[0])

    years_to_mat = (maturity - df['date']).dt.days / 365.0
    dirty_prices = [get_dirty_price(p, coupon, d, next_coupon_date, payments_per_year)
                    for p, d in zip(df['Close Price (% of par)'], df['date'])]
    df['ytm'] = [ytm(dp, coupon, t) for dp, t in zip(dirty_prices, years_to_mat)]
    col_label = f"Y{i+1}yr"
    if yields_df.empty:
        yields_df = df[['date', 'ytm']].rename(columns={'ytm': col_label})
    else:
        yields_df = yields_df.merge(df[['date', 'ytm']].rename(columns={'ytm': col_label}), on='date')

yields_df.set_index('date', inplace=True)

# Build 1-year forward rates from yields
forwards_df = pd.DataFrame(index=yields_df.index)
for k in range(2, 6):
    forwards_df[f"F1yr-{k}yr"] = ((1 + yields_df[f"Y{k}yr"])**k / (1 + yields_df["Y1yr"]))**(1/(k-1)) - 1

# Log returns
yield_logret = np.log(yields_df / yields_df.shift(1)).dropna()
forward_logret = np.log(forwards_df / forwards_df.shift(1)).dropna()

# Covariance matrices
cov_yields = yield_logret.cov()
cov_forwards = forward_logret.cov()
print("Covariance matrix - Yields:\n", cov_yields)
print("Covariance matrix - Forwards:\n", cov_forwards)

# -------------------------------------------------------------
# Q6 - Eigen decomposition
# -------------------------------------------------------------
eigval_y, eigvec_y = np.linalg.eig(cov_yields)
eigval_f, eigvec_f = np.linalg.eig(cov_forwards)

# Sort descending
idx_y = eigval_y.argsort()[::-1]
eigval_y, eigvec_y = eigval_y[idx_y], eigvec_y[:, idx_y]

idx_f = eigval_f.argsort()[::-1]
eigval_f, eigvec_f = eigval_f[idx_f], eigvec_f[:, idx_f]

print("\nLargest eigenvalue (Yield):", eigval_y[0])
print("Associated eigenvector (Yield):\n", eigvec_y[:, 0])
print("\nLargest eigenvalue (Forward):", eigval_f[0])
print("Associated eigenvector (Forward):\n", eigvec_f[:, 0])
