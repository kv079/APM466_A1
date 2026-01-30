import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. Setup: Hardcoded maturity dates for the 10 bonds
bond_maturities = [
    "2026-08-01", "2027-02-01", "2027-08-01", "2028-02-01", "2028-09-01",
    "2029-03-01", "2029-09-01", "2030-03-01", "2030-09-01", "2031-03-01"
]

# 2. Load Data
df = pd.read_csv('ytm_results.csv', index_col=0)
df.index = pd.to_datetime(df.index)  # Convert index to datetime objects

# 3. Process & Plot
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(df)))  # Create a color gradient
target_years = [1, 2, 3, 4, 5]

for i, (current_date, yields) in enumerate(df.iterrows()):
    # Calculate Time to Maturity (in years) for each bond
    # Using list comprehension for cleaner code
    t_years = [(pd.to_datetime(m) - current_date).days / 365 for m in bond_maturities]

    # Linear Interpolation
    # 'extrapolate' is used to handle edge cases near 5.0 years
    curve_func = interp1d(t_years, yields, kind='linear', fill_value='extrapolate')
    interp_y = curve_func(target_years)

    # Plot the curve for this date
    plt.plot(target_years, interp_y,
             label=current_date.strftime('%Y-%m-%d'),
             color=colors[i], marker='.', linewidth=1.5)

# 4. Final Styling
plt.title('5-Year Yield Curve (Superimposed) - Jan 5 to Jan 19, 2026')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield to Maturity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Date")
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks(target_years)  # Force x-axis to be integers 1-5
plt.tight_layout()

plt.savefig('yield_curve_4a.png', dpi=300)
plt.show()
print("Plot saved as yield_curve_4a.png")