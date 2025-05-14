import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load dataset
df = pd.read_excel("SouthKorea-April4-2020 (1).xlsx", skiprows=1)
df = df[['Days (X)', 'Infected  population']].copy()
df.columns = ['Days', 'P']

# Compute log(P + 1)
logP = np.log(df['P'] + 1)

# Shifted series for x_n and x_{n+1}
x_n = logP[:-1].values          # log(P_n + 1)
x_next = logP[1:].values        # log(P_{n+1} + 1)

# Phase space plot
plt.figure(figsize=(6, 6))
plt.plot(x_n, x_next, 'bo-', label='South Korea')
plt.xlabel(r'$\log(P_n + 1)$')
plt.ylabel(r'$\log(P_{n+1} + 1)$')
plt.title("Phase Space Plot: log(Pₙ+1) vs log(Pₙ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# Load data
df = pd.read_excel("SouthKorea-April4-2020 (1).xlsx", skiprows=1)
df = df[['Days (X)', 'Infected  population']].copy()
df.columns = ['Days', 'P']

# Remove rows with missing or zero P (to avoid log(0))
df = df[df['P'].notna() & (df['P'] >= 0)]

# Compute x_n = log(P + 1)
logP = np.log(df['P'] + 1)

# Shifted series
x_n = logP[:-1].values
x_next = logP[1:].values

# Remove any pairs where either x_n or x_next is NaN or inf
valid = np.isfinite(x_n) & np.isfinite(x_next)
x_n = x_n[valid]
x_next = x_next[valid]

# Define nonlinear map: tanh-based
def tanh_map(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d

# Fit the map
popt, _ = curve_fit(tanh_map, x_n, x_next, p0=[1, 1, 0, 0])
a, b, c, d = popt
print(f"Fitted parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

# Plot f(x) and diagonal
x_vals = np.linspace(min(x_n), max(x_n), 200)
fx = tanh_map(x_vals, *popt)

plt.figure(figsize=(6, 6))
plt.plot(x_vals, fx, 'r-', label=r'$f(x_n)$ (fitted)')
plt.plot(x_vals, x_vals, 'k--', label=r'$x_{n+1} = x_n$ (diagonal)')
plt.scatter(x_n, x_next, color='blue', s=20, label='Data')
plt.xlabel(r'$x_n = \log(P_n + 1)$')
plt.ylabel(r'$x_{n+1} = \log(P_{n+1} + 1)$')
plt.title("Nonlinear Map Fit (South Korea)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




from scipy.optimize import root_scalar
from scipy.misc import derivative

# Define f(x) - x
def fx_minus_x(x):
    return tanh_map(x, *popt) - x

# Find root numerically (i.e. solve f(x) = x)
sol = root_scalar(fx_minus_x, bracket=[0, 10], method='brentq')

if sol.converged:
    x_fixed = sol.root
    print(f"Fixed point at x = {x_fixed:.4f}")

    # Stability check: compute derivative f'(x) at the fixed point
    stability = derivative(lambda x: tanh_map(x, *popt), x_fixed, dx=1e-6)
    print(f"f'(x_fixed) = {stability:.4f} → ", end='')
    if abs(stability) < 1:
        print("Stable ✅")
    else:
        print("Unstable ❌")
else:
    print("No fixed point found in range.")


# Simulate the map starting from an initial value
x_iter = [x_n[0]]  # start from first observed value
for _ in range(50):  # simulate 50 steps
    x_next_sim = tanh_map(x_iter[-1], *popt)
    x_iter.append(x_next_sim)

# Plot time series
plt.figure(figsize=(6, 4))
plt.plot(x_iter, 'o-', label='Simulated $x_n$')
plt.axhline(y=x_fixed, color='gray', linestyle='--', label='Fixed Point')
plt.xlabel('n (iteration)')
plt.ylabel(r'$x_n = \log(P_n + 1)$')
plt.title("Simulation of Recursive Map")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# --- Back-transform: log(P + 1) → P ---
P_simulated = np.exp(x_iter) - 1

# Extract actual data to compare
P_actual = df['P'].values
days = df['Days'].values

# Trim to same length (50 steps simulated)
days_sim = list(range(len(P_simulated)))
P_actual_trimmed = P_actual[:len(P_simulated)]

# --- Plot comparison ---
plt.figure(figsize=(7, 4))
plt.plot(days_sim, P_simulated, 'b-o', label='Simulated P_n')
plt.plot(days_sim, P_actual_trimmed, 'g--o', label='Actual P_n (South Korea)')
plt.xlabel("Day")
plt.ylabel("Cumulative Infected Population")
plt.title("Simulated vs Actual Infection Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# --- Load data ---
df = pd.read_excel("SouthKorea-April4-2020 (1).xlsx", skiprows=1)
df = df[['Days (X)', 'Infected  population']].copy()
df.columns = ['Days', 'P']
df = df[df['P'].notna()]  # remove missing values

# --- Extract time and infected population ---
t = df['Days'].values
P = df['P'].values

# --- Logistic Model ---
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# --- Richards Model ---
def richards(t, K, r, t0, alpha):
    return K / ((1 + alpha * np.exp(-r * (t - t0))) ** (1 / alpha))

# --- Bifurcation (Recursive) Model ---
def bifurcation(t, beta, r, t0, theta, P0):
    return np.exp(beta * (2 / (1 + np.exp(-2 * r * (t - t0)) ** theta) - 1) + np.log(P0 + 1)) - 1

# --- Estimate initial values ---
P0_guess = max(P[0], 1)
K_guess = P.max() * 1.1
t0_guess = t[np.argmax(np.gradient(P))]

# --- Fit Logistic Model ---
popt_log, _ = curve_fit(logistic, t, P, p0=[K_guess, 0.2, t0_guess])

# --- Fit Richards Model ---
popt_rich, _ = curve_fit(richards, t, P, p0=[K_guess, 0.2, t0_guess, 1.0])

# --- Fit Bifurcation Model (Robust) ---
beta_guess = 7       # approx log(Pmax + 1)
r_guess = 0.1
theta_guess = 1.0
bounds = (
    [0, 0, 0, 0, 1],         # lower bounds
    [20, 1.0, 100, 10, 1e6]  # upper bounds
)

popt_bif, _ = curve_fit(
    bifurcation, t, P,
    p0=[beta_guess, r_guess, t0_guess, theta_guess, P0_guess],
    bounds=bounds,
    maxfev=10000
)

# --- Generate predictions for plotting ---
t_fit = np.linspace(min(t), max(t), 200)
P_log = logistic(t_fit, *popt_log)
P_rich = richards(t_fit, *popt_rich)
P_bif = bifurcation(t_fit, *popt_bif)

# --- Plot all 3 model fits ---
plt.figure(figsize=(8, 5))
plt.plot(t, P, 'ko', label='Actual data')
plt.plot(t_fit, P_log, 'r--', label='Logistic fit')
plt.plot(t_fit, P_rich, 'g-.', label='Richards fit')
plt.plot(t_fit, P_bif, 'b-', label='Bifurcation fit')
plt.xlabel("Day")
plt.ylabel("Cumulative Infected Population")
plt.title("Model Comparison: Logistic vs Richards vs Bifurcation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print fitted parameters ---
print("\nLogistic Parameters:")
print("  K = {:.0f}, r = {:.4f}, t0 = {:.2f}".format(*popt_log))

print("\nRichards Parameters:")
print("  K = {:.0f}, r = {:.4f}, t0 = {:.2f}, alpha = {:.4f}".format(*popt_rich))

print("\nBifurcation Parameters:")
print("  beta = {:.4f}, r = {:.4f}, t0 = {:.2f}, theta = {:.4f}, P0 = {:.0f}".format(*popt_bif))




from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Interpolate model predictions at actual t points for error computation
P_log_pred = logistic(t, *popt_log)
P_rich_pred = richards(t, *popt_rich)
P_bif_pred = bifurcation(t, *popt_bif)

# --- Compute MAPE and RMSE ---
def compute_errors(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, rmse

mape_log, rmse_log = compute_errors(P, P_log_pred)
mape_rich, rmse_rich = compute_errors(P, P_rich_pred)
mape_bif, rmse_bif = compute_errors(P, P_bif_pred)

# --- Final predicted values ---
P_true_final = P[-1]
P_log_final = P_log_pred[-1]
P_rich_final = P_rich_pred[-1]
P_bif_final = P_bif_pred[-1]

# --- Print Table 2-like summary ---
print("\n" + "-"*65)
print(f"{'Model':<15}{'Final Pred':>15}{'True Value':>15}{'MAPE (%)':>10}{'RMSE':>10}")
print("-"*65)
print(f"{'Logistic':<15}{P_log_final:15.2f}{P_true_final:15.0f}{mape_log:10.2f}{rmse_log:10.2f}")
print(f"{'Richards':<15}{P_rich_final:15.2f}{P_true_final:15.0f}{mape_rich:10.2f}{rmse_rich:10.2f}")
print(f"{'Bifurcation':<15}{P_bif_final:15.2f}{P_true_final:15.0f}{mape_bif:10.2f}{rmse_bif:10.2f}")
print("-"*65)
