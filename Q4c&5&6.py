import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton

# -------------------------------------------------------------
# Configuration: Sorted file list (Shortest → Longest maturity)
# -------------------------------------------------------------
SORTED_FILES = [
    "Short-term_CA135087R978.csv", "Short-term_CA135087S547.csv",
    "Short-term_CA135087T461.csv", "Short-term_CA135087T958.csv",
    "Short-term_CA135087Q491.csv", "Long-term_CA135087Q988.csv",
    "Long-term_CA135087R895.csv", "Long-term_CA135087S471.csv",
    "Long-term_CA135087T388.csv", "Long-term_CA135087T792.csv"
]

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def get_dirty_price(clean, coupon_rate, maturity, current_date):
    """Dirty price = clean + accrued coupon interest."""
    m_dt, c_dt = pd.to_datetime(maturity), pd.to_datetime(current_date)
    last_c = m_dt
    while last_c > c_dt:
        last_c -= pd.DateOffset(months=6)
    days_since = (c_dt - last_c).days
    days_period = ((last_c + pd.DateOffset(months=6)) - last_c).days
    return clean + (coupon_rate * 100 / 2) * (days_since / days_period)

def get_cash_flows(coupon_rate, maturity, current_date):
    """Return arrays of times (years) and cash-flow amounts."""
    m_dt, c_dt = pd.to_datetime(maturity), pd.to_datetime(current_date)
    times, flows = [], []
    temp = m_dt
    while temp > c_dt:
        t = (temp - c_dt).days / 365.0
        amt = coupon_rate * 100 / 2
        if temp == m_dt:
            amt += 100
        times.append(t)
        flows.append(amt)
        temp -= pd.DateOffset(months=6)
    return np.array(times)[::-1], np.array(flows)[::-1]

# -------------------------------------------------------------
# Bootstrapping Spot Curves
# -------------------------------------------------------------
print("Running Bootstrapping...")

df0 = pd.read_csv(SORTED_FILES[0])
df0['date'] = pd.to_datetime(df0['date'])
dates = sorted(df0[(df0['date'] >= '2026-01-05') &
                   (df0['date'] <= '2026-01-19')]['date'].unique())

spot_curves = []      # list of arrays: each array = spot rates [1..5] for one date
target_years = [1, 2, 3, 4, 5]

for date in dates:
    curr_date_str = date.strftime('%Y-%m-%d')
    known_T, known_r = [0], [0]

    for i, fname in enumerate(SORTED_FILES):
        row = pd.read_csv(fname).set_index('date').loc[curr_date_str]
        mat = row['Maturity Date']
        coup = float(str(row['Coupon']).strip('%')) / 100.0
        price = get_dirty_price(row['Close Price (% of par)'], coup, mat, curr_date_str)
        times, flows = get_cash_flows(coup, mat, curr_date_str)

        if i == 0:
            # First maturity: solve for flat yield matching PV to price
            f = lambda r: np.sum(flows * np.exp(-r * times)) - price
            r_spot = newton(f, 0.04)
        else:
            # Later bonds: discount earlier coupons using previous spot curve
            f_interp = interp1d(known_T, known_r, kind='linear', fill_value='extrapolate')
            pv_coupons = sum(cf * np.exp(-f_interp(t) * t)
                             for t, cf in zip(times[:-1], flows[:-1]))
            residual = price - pv_coupons
            r_spot = -np.log(residual / flows[-1]) / times[-1]

        known_T.append(times[-1])
        known_r.append(r_spot)

    # Interpolate full 1-5 year curve for this date
    final_curve = interp1d(known_T, known_r, kind='linear', fill_value='extrapolate')
    spot_curves.append(final_curve(target_years))

# -------------------------------------------------------------
# Q4(c) - Compute 1-Year Forward Curves from Spot Rates
# -------------------------------------------------------------
forward_curves = []
for curve in spot_curves:
    S = dict(zip(target_years, curve))
    fwd = []
    for k in range(1, 5):
        S1 = S[1]
        S1k = S[1 + k] if (1 + k) in S else list(S.values())[min(1 + k, 5) - 1]
        fwd_rate = ((1 + S1k) ** (1 + k) / (1 + S1)) ** (1 / k) - 1
        fwd.append(fwd_rate)
    forward_curves.append(fwd)

# Plot Forward Curves (superimposed)
plt.figure(figsize=(9, 6))
colours = plt.cm.plasma(np.linspace(0, 1, len(dates)))
for i, curve in enumerate(forward_curves):
    plt.plot(range(2, 6), curve, color=colours[i], marker='s',
             label=dates[i].strftime('%Y-%m-%d'))
plt.xlabel("Forward Term (Years)")
plt.ylabel("1-Year Forward Rate")
plt.title("1-Year Forward Curves (Daily superimposed)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), title="Date")
plt.tight_layout()
plt.savefig("forward_curve.png", dpi=300)
plt.show()
print("Forward curves plotted and saved as 'forward_curve.png'.")

# -------------------------------------------------------------
# Q5
# -------------------------------------------------------------

def ytm(price_pct, coupon_rate, years_to_maturity, face_value=100, freq=2):
    price = price_pct / 100 * face_value
    coupon = coupon_rate * face_value / freq
    n_periods = int(round(years_to_maturity * freq))

    def f(y):
        return sum(coupon / (1 + y/freq)**t for t in range(1, n_periods+1)) \
               + face_value / (1 + y/freq)**n_periods - price

    return newton(f, x0=0.03)  # initial guess

# --- Build yields_df ---
ytm_data = []
for i, file in enumerate(SORTED_FILES[:5]):  # first 5 bonds = ~1–5yr maturities
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    maturity = pd.to_datetime(df['Maturity Date'].iloc[0])
    coupon = float(str(df['Coupon'].iloc[0]).replace('%',''))/100.0
    years_to_maturity = (maturity - df['date']).dt.days / 365.0
    df['ytm'] = [ytm(price, coupon, t) for price, t in zip(df['Close Price (% of par)'], years_to_maturity)]
    if i == 0:
        yields_df = df[['date', 'ytm']].copy()
        yields_df.rename(columns={'ytm': f'Y{i+1}yr'}, inplace=True)
    else:
        yields_df = yields_df.merge(
            df[['date', 'ytm']].rename(columns={'ytm': f'Y{i+1}yr'}),
            on='date'
        )

yields_df.set_index('date', inplace=True)

# --- Build forwards_df from YTMs ---
forwards_df = pd.DataFrame(index=yields_df.index)
for k in range(2, 6):  # maturities 2..5
    F = ((1 + yields_df[f"Y{k}yr"])**k / (1 + yields_df["Y1yr"])**1)**(1/(k-1)) - 1
    forwards_df[f"F1yr-{k}yr"] = F

# --- Log returns ---
yield_logret = np.log(yields_df / yields_df.shift(1)).dropna()
forward_logret = np.log(forwards_df / forwards_df.shift(1)).dropna()

# --- Covariance ---
cov_yields = yield_logret.cov()
cov_forwards = forward_logret.cov()

print("Covariance matrix - Yields:\n", cov_yields)
print("Covariance matrix - Forwards:\n", cov_forwards)


# -------------------------------------------------------------
# Q6
# -------------------------------------------------------------

import numpy as np

# Eigen decomposition for Yield Covariance
eigval_y, eigvec_y = np.linalg.eig(cov_yields)

# Eigen decomposition for Forward Covariance
eigval_f, eigvec_f = np.linalg.eig(cov_forwards)

# Sort eigenvalues and eigenvectors in descending order of eigenvalue size
idx_y = eigval_y.argsort()[::-1]
eigval_y = eigval_y[idx_y]
eigvec_y = eigvec_y[:, idx_y]

idx_f = eigval_f.argsort()[::-1]
eigval_f = eigval_f[idx_f]
eigvec_f = eigvec_f[:, idx_f]

print("Largest eigenvalue (Yield):", eigval_y[0])
print("Associated eigenvector (Yield):\n", eigvec_y[:, 0])

print("Largest eigenvalue (Forward):", eigval_f[0])
print("Associated eigenvector (Forward):\n", eigvec_f[:, 0])
