import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1) Load your bifurcation-export CSV, skipping the extra header row
df = pd.read_csv(
    "C:/Users/priyansh sinha/Documents/southkorea.csv",
    skiprows=1,
    usecols=['Days (X)', 'y']
).dropna()

# 2) Convert columns to numeric
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['y']        = pd.to_numeric(df['y'],        errors='coerce')
df.dropna(inplace=True)

# 3) Reconstruct P from y = ln(P + 1)
days = df['Days (X)'].astype(int).values
logP = df['y'].values
P    = np.exp(logP) - 1

# 4) Paper parameters
D0 = days[0]                      # first day (often 0)
Tc = 28                           # bifurcation day from the paper
# Find index of the day closest to Tc
idx_tc = np.argmin(np.abs(days - Tc))

P0 = P[0]                         # P at D0
Pc = P[idx_tc]                    # P at (closest) Tc

# 5) Compute the log-transformed series
logP0 = np.log(P0 + 1)
logPc = np.log(Pc + 1)

# 6) Build X and W for D0 < D <= Tc
X_all = days - D0
mask = (days > D0) & (days <= Tc)
X_fit = X_all[mask]

ratio = (logP[mask] - logP0) / (logPc - logP0)
inner = 2 / (1 + ratio) - 1

# Keep only valid points where inner>0
valid = inner > 0
X_fit = X_fit[valid]
W_fit = -0.5 * np.log(inner[valid])

# 7) Fit W = r1 * X
def linear_model(x, r1):
    return r1 * x

popt, _ = curve_fit(linear_model, X_fit, W_fit)
r1_est = popt[0]

# 8) Plot Figure 3(a)
plt.figure(figsize=(8,5))
plt.plot(X_fit, W_fit, 'ko', label='Transformed data')
plt.plot(
    X_fit,
    linear_model(X_fit, r1_est),
    'r-', linewidth=2,
    label=f'$W = r_1 (D-D_0)$\n$r_1$ = {r1_est:.4f}'
)
plt.xlabel('Days since first case $(D - D_0)$', fontsize=12)
plt.ylabel('$W$', fontsize=12)
plt.title('Figure 3(a) – Estimation of Spread Rate $r_1$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 9) Print the estimated r1
print(f"Estimated spread rate r1 = {r1_est:.4f}")




# 1) Load dataset
df = pd.read_csv(
    "C:/Users/priyansh sinha/Documents/southkorea.csv",
    skiprows=1,
    usecols=['Days (X)', 'Infected  population']
).dropna()

# 2) Clean and extract
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# 3) Parameters
D0 = days[0]
Tc = 28
idx_tc = np.argmin(np.abs(days - Tc))
P0 = P[0]
Pc = P[idx_tc]

# 4) Log-transform
logP = np.log(P + 1)
logP0 = np.log(P0 + 1)
logPc = np.log(Pc + 1)

# 5) Restrict to first cycle
X_all = days - D0
mask = (days >= 0) & (days <= Tc)
X_fit = X_all[mask]

ratio = (logP[mask] - logP0) / (logPc - logP0)
inner = 2 / (1 + ratio) - 1
valid = inner > 0
X_fit = X_fit[valid]
P_true = P[mask][valid]

# 6) Predicted P using r1
r1_est = 0.0995
logP1_end = np.log(Pc + 1)
term = 2 / (1 + np.exp(-2 * r1_est * X_fit)) - 1
logP_pred = logP1_end * term
P_pred = np.exp(logP_pred) - 1

# 7) Plot with all dots connected
plt.figure(figsize=(8, 5))
plt.plot(X_fit, P_true, 'ko-', label='True data', markersize=5)  # connected black dots
plt.plot(X_fit, P_pred, 'r-', linewidth=2, label='Predicted (cycle 1)')
plt.plot(X_fit, P_pred, 'ro', markersize=4)  # red dots
plt.xlabel('Days since first case', fontsize=12)
plt.ylabel('Cumulative infections', fontsize=12)
plt.title(' Cycle 1 Prediction Using $r_1$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




# 1) Load the South Korea dataset
df = pd.read_csv("C:/Users/priyansh sinha/Documents/southkorea.csv", skiprows=1)
df = df[['Days (X)', 'Infected  population']].replace('#NUM!', np.nan).dropna()
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

# 2) Extract raw data
days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# 3) Cycle 2: Assume Di−1 = 28 (end of cycle 1), and Pi−1 = P at day 28
D_prev = 28
r1 = 0.0995
idx_D_prev = np.argmin(np.abs(days - D_prev))
P_prev = P[idx_D_prev]

# 4) Choose a range for cycle 2 (e.g., D = 28 to D = 50)
mask_cycle2 = (days >= D_prev) & (days <= 50)
D2 = days[mask_cycle2]
P2 = P[mask_cycle2]

# 5) Pi = last point in cycle 2 (e.g., day 50)
D_end = D2[-1]
P_end = P2[-1]

# 6) Compute y and Z from Eq. (6a) and (6b)
logP = np.log(P2 + 1)
logP_prev = np.log(P_prev + 1)
logP_end = np.log(P_end + 1)

y = logP - logP_prev
Z = (logP_end - logP_prev) * (2 / (1 + np.exp(-2 * r1 * (D2 - D_prev))) - 1)

# 7) Fit y = α Z to validate linearity and get α
def linear_alpha(Z, alpha):
    return alpha * Z

popt, _ = curve_fit(linear_alpha, Z, y)
alpha_est = popt[0]

# 8) Plot Figure 4
plt.figure(figsize=(8, 5))
plt.plot(Z, y, 'ko', label='Data')
plt.plot(Z, linear_alpha(Z, alpha_est), 'r-', linewidth=2, label=f'Linear Fit: $\\alpha$ = {alpha_est:.4f}')
plt.xlabel('$Z$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.title('Validation in Cycle 2 Using $r_1$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 9) Print alpha value
print(f"Estimated alpha = {alpha_est:.4f}")





# 1) Load the dataset again
df = pd.read_csv("C:/Users/priyansh sinha/Documents/southkorea.csv", skiprows=1)
df = df[['Days (X)', 'Infected  population']].replace('#NUM!', np.nan).dropna()
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

# 2) Extract data
days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# 3) Cycle 2 settings
D_prev = 28
r1 = 0.0995
alpha = 1.0728
idx_D_prev = np.argmin(np.abs(days - D_prev))
P_prev = P[idx_D_prev]

# Cycle 2 range: D from 28 to 50
mask_cycle2 = (days >= D_prev) & (days <= 50)
D2 = days[mask_cycle2]
P2_true = P[mask_cycle2]

# Endpoint of cycle 2
P_end = P2_true[-1]

# 4) Compute predicted log(P + 1)
logP_prev = np.log(P_prev + 1)
logP_end = np.log(P_end + 1)
Z_term = (2 / (1 + np.exp(-2 * r1 * (D2 - D_prev))) - 1)
logP_pred = logP_prev + alpha * (logP_end - logP_prev) * Z_term
P2_pred = np.exp(logP_pred) - 1

# 5) Plot Figure 4(b)
plt.figure(figsize=(8, 5))
plt.plot(D2, P2_true, 'ko-', label='True data', markersize=5)
plt.plot(D2, P2_pred, 'r-', label='Predicted (cycle 2)', linewidth=2)
plt.plot(D2, P2_pred, 'ro', markersize=4)  # red dots for prediction
plt.xlabel('Days since first case', fontsize=12)
plt.ylabel('Cumulative infections', fontsize=12)
plt.title('Prediction of Cycle 2 Using $r_1$ and $\\alpha$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# 1) Load the dataset
df = pd.read_csv("C:/Users/priyansh sinha/Documents/southkorea.csv", skiprows=1)
df = df[['Days (X)', 'Infected  population']].replace('#NUM!', np.nan).dropna()
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

# 2) Extract arrays
days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# 3) Cycle 2 parameters
D_prev = 28
r1 = 0.0995
alpha = 1.0728
idx_D_prev = np.argmin(np.abs(days - D_prev))
P_prev = P[idx_D_prev]

# Select cycle 2 range: Days 28 to 50
mask = (days >= D_prev) & (days <= 50)
D2 = days[mask]
P2_true = P[mask]

# Endpoint Pi = day 50
P_end = P2_true[-1]

# 4) Compute log(P + 1)
logP_prev = np.log(P_prev + 1)
logP_end = np.log(P_end + 1)
logP_true = np.log(P2_true + 1)

# Compute predicted log(P + 1)
Z = (2 / (1 + np.exp(-2 * r1 * (D2 - D_prev))) - 1)
logP_pred = logP_prev + alpha * (logP_end - logP_prev) * Z

# 5) Plot log(P + 1) vs. Days
plt.figure(figsize=(8, 5))
plt.plot(D2, logP_true, 'ko-', label='True $\log(P+1)$', markersize=5)
plt.plot(D2, logP_pred, 'r-', label='Predicted $\log(P+1)$', linewidth=2)
plt.plot(D2, logP_pred, 'ro', markersize=4)
plt.xlabel('Days since first case', fontsize=12)
plt.ylabel('$\log(P + 1)$', fontsize=12)
plt.title(' Predicted vs. True $\log(P+1)$ for Cycle 2', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()





# Load South Korea dataset
df = pd.read_csv("C:/Users/priyansh sinha/Documents/southkorea.csv", skiprows=1)
df = df[['Days (X)', 'Infected  population']].replace('#NUM!', np.nan).dropna()
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

# Extract arrays
days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# Parameters from paper and your earlier output
D_prev = 28
r1 = 0.0995
alpha = 1.0728

# Find index of D_prev
idx_D_prev = np.argmin(np.abs(days - D_prev))
P_prev = P[idx_D_prev]

# Select days 1 to 50
mask = (days >= 1) & (days <= 50)
D_range = days[mask]
P_true = P[mask]

# Use Pi = value at day 50
P_end = P[np.argmin(np.abs(days - 50))]

# Log transforms
logP_prev = np.log(P_prev + 1)
logP_end = np.log(P_end + 1)
logP_true = np.log(P_true + 1)

# Compute predicted log(P + 1) using Equation (6)
Z_term = (2 / (1 + np.exp(-2 * r1 * (D_range - D_prev))) - 1)
logP_pred = logP_prev + alpha * (logP_end - logP_prev) * Z_term

# Plot log(P + 1) from day 1 to 50
plt.figure(figsize=(8, 5))
plt.plot(D_range, logP_true, 'ko-', label='True $\log(P + 1)$', markersize=5)
plt.plot(D_range, logP_pred, 'r-', label='Predicted $\log(P + 1)$', linewidth=2)
plt.plot(D_range, logP_pred, 'ro', markersize=4)
plt.xlabel('Days since first case', fontsize=12)
plt.ylabel('$\log(P + 1)$', fontsize=12)
plt.title('Prediction from Day 1 to 50 using $r_1$ and $\\alpha$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




# --- Load South Korea dataset ---
df = pd.read_csv("C:/Users/priyansh sinha/Documents/southkorea.csv", skiprows=1)
df = df[['Days (X)', 'Infected  population']].replace('#NUM!', np.nan).dropna()
df['Days (X)'] = pd.to_numeric(df['Days (X)'], errors='coerce')
df['Infected  population'] = pd.to_numeric(df['Infected  population'], errors='coerce')
df.dropna(inplace=True)

days = df['Days (X)'].astype(int).values
P = df['Infected  population'].values

# --- Setup known values ---
Ti = 40
Tc = 28
reference_day = int(0.9 * Ti)  # 0.9Ti = 36
forecast_day = int(3.55 * Ti)  # 3.55Ti = 142
true_value = 12051  # actual infected population at 3.55Ti

# --- Training data: up to reference_day (36) ---
mask = days <= reference_day
days_train = days[mask]
P_train = P[mask]

# --- 1. Logistic Model ---
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

p0_log = [max(P_train), 0.2, np.median(days_train)]
popt_log, pcov_log = curve_fit(logistic, days_train, P_train, p0=p0_log, maxfev=5000)
forecast_log = logistic(forecast_day, *popt_log)

# --- 2. Richards Model ---
def richards(t, K, r, t0, a):
    return K / (1 + np.exp(-r * (t - t0)))**(1/a)

p0_rich = [max(P_train), 0.2, np.median(days_train), 1.0]
popt_rich, pcov_rich = curve_fit(richards, days_train, P_train, p0=p0_rich, maxfev=5000)
forecast_rich = richards(forecast_day, *popt_rich)

# --- 3. Recursive Bifurcation Prediction using Eq. (8) ---
# Use r1 from earlier estimate
r1 = 0.0995
Dn = 36  # reference day
Pn_1 = P[np.where(days == Dn)[0][0]]  # infected at day 36

# Define bifurcation Eq. (8)
def bifurcation(d, beta, Dn, theta):
    part = 2 / (1 + (np.exp(-2 * r1 * (d - Dn)))**theta) - 1
    return np.exp(beta * part + np.log(Pn_1 + 1)) - 1

# Fit bifurcation model to post-reference data
mask2 = days > Dn
days_bif = days[mask2]
P_bif = P[mask2]

popt_bif, _ = curve_fit(bifurcation, days_bif, P_bif, p0=[2.0, Dn + 5, 1.0], maxfev=5000)
forecast_bif = bifurcation(forecast_day, *popt_bif)

# --- Compute absolute relative errors ---
def abs_rel_err(pred, actual):
    return abs(pred - actual) / actual * 100

err_log = abs_rel_err(forecast_log, true_value)
err_rich = abs_rel_err(forecast_rich, true_value)
err_bif = abs_rel_err(forecast_bif, true_value)

# --- Print Table 2 ---
print("Table 2 – Forecast Comparison at 3.55 Ti (Day 142)")
print("--------------------------------------------------")
print(f"{'Model':<20} {'Prediction':>10} {'True':>10} {'Abs. Rel. Error (%)':>20}")
print(f"{'-'*60}")
print(f"{'Logistic':<20} {forecast_log:10.0f} {true_value:10} {err_log:20.1f}")
print(f"{'Richards':<20} {forecast_rich:10.0f} {true_value:10} {err_rich:20.1f}")
print(f"{'Bifurcation':<20} {forecast_bif:10.0f} {true_value:10} {err_bif:20.1f}")

