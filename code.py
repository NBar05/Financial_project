import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import ncx2, chi2

# Importing data with MosPrime rates
data = pd.read_excel("/Users/nikitabaramiya/Desktop/MSU/5 семестр/Теория финансов/Finance project/MosPrime.xlsx", header=0)
for col in data.columns[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.iloc[:, 1:] = data.iloc[:, 1:] / 100

# The CIR model, simulation
def cir(r0, K, theta, sigma, T=10, N=12, seed=777):
    np.random.seed(seed)
    dt = 1 / float(N)
    rates = [r0]
    for i in range(N*T):
        dr = K * (theta - rates[-1]) * dt + \
        sigma * (rates[-1])**(1/2) * np.random.normal() * dt**(1/2)
        rates.append(rates[-1] + dr)
    return range(T*N+1), rates

# The following code calculates log likelihood and return -likelihood
rates = data['2week']
dt = 1/365
def mle_cir(params):
    global rates, dt
    K, theta, sigma = params
    r0_vector = rates[0:rates.shape[0]-1]
    r1_vector = rates[1:]
    c = 2 * K / ( (1 - np.exp(-K * dt) ) * sigma**2)
    nc = 2 * c * r0_vector.values * np.exp(-K * dt)
    df = 4 * theta / (K * sigma**2)
    lik = ncx2.logpdf(2*r1_vector.values*c, df, nc) + np.log(2*c)
    likelihood = pd.Series(lik).sum()
    return -likelihood

# Initial guesses of parameters and constraints of the model
initial_guess = [0.80, 0.06, 0.30]
bnds = ((0.001, 1), (0.001, 1), (0.001, 1))

def constraint(params):
    K, theta, sigma = params
    return 2 * K * theta - sigma**2

# Find the best parameters
result = minimize(mle_cir, initial_guess, bounds=bnds, constraints={'type': 'ineq', 'fun': constraint})
K_res, theta_res, sigma_res = result.x

# Look how well we calibrate parameters
a1, b1 = cir(rates[0], K_res, theta_res, sigma_res, T=6, N=365, seed=777)
plt.plot(a1, b1, color='blue')
plt.plot(a1, rates[:len(b1)], color='green')
plt.show()

# Create interest rate tree
def build_tree(r0, K, theta, sigma, step, N):
    tree_nodes = [np.array(float(r0))]
    model_rates = [r0]
    for i in range(1, int(step*N)):
        node = np.repeat(float(0), i+1)
        r_next = model_rates[-1] + K * (theta - model_rates[-1]) * (1/step) + sigma * np.sqrt(model_rates[-1]) * np.sqrt(1/step)
        if i == 1:
            node[0] = r_next + sigma * np.sqrt(model_rates[-1]) * np.sqrt(1/step)
            node[1] = r_next - sigma * np.sqrt(model_rates[-1]) * np.sqrt(1/step)
        elif i % 2 == 0:
            node[len(node) // 2] = r_next
            for e, l in enumerate(range(len(node)-1, 1, -2)):
                node[e] = r_next + l * sigma * np.sqrt(tree_nodes[-1][e]) * np.sqrt(1/step)
                node[-1-e] = r_next - l * sigma * np.sqrt(tree_nodes[-1][-1-e]) * np.sqrt(1/step)
        else:
            for e, l in enumerate(range(len(node)-1, 0, -2)):
                node[e] = r_next + l * sigma * np.sqrt(tree_nodes[-1][e]) * np.sqrt(1/step)
                node[-1-e] = r_next - l * sigma * np.sqrt(tree_nodes[-1][-1-e]) * np.sqrt(1/step)
        tree_nodes.append(node)
        model_rates.append(r_next)
    return tree_nodes

# Create estimation of obligation with embedded option via interest-rate tree
def estimate_obligation(tree_nodes, fv, coupon, fixed_value, EN, step=1, option='put'):
    embedded_option = lambda value, fv, option: max(value, fv) if option == 'put' else min(value, fv)
    all_induction_steps = [np.repeat(fv, len(tree_nodes)+1)]
    for i in range(len(tree_nodes), 1, -1):
        induction = np.repeat(float(0), i)
        for element in range(i):
            if i in EN:
                induction[element] = embedded_option(0.5 * ( (all_induction_steps[len(tree_nodes)-i][element] + coupon/step + \
                all_induction_steps[len(tree_nodes)-i][element+1] + coupon/step) / (1+tree_nodes[i-1][element]/step) ), fixed_value, option)
            else:
                induction[element] = 0.5 * ( (all_induction_steps[len(tree_nodes)-i][element] + coupon/step + \
                all_induction_steps[len(tree_nodes)-i][element+1] + coupon/step) / (1+tree_nodes[i-1][element]/step) )
        all_induction_steps.append(induction)
    price = embedded_option(0.5 * ( (all_induction_steps[-1][0] + all_induction_steps[-1][1] + 2*coupon/step) / (1+tree_nodes[0]/step) ), fixed_value, option)
    all_induction_steps.append(np.array(price))
    return price, all_induction_steps

# Estimate obligation
# "TransFin-M" Public Company, exchange-traded bond (RU000A0JVHH1)
fv, coupon = 1000, (13.50 / 100) * 1000 # Maturity date 04.06.2025
fixed_value = 1000 # Offer date	17.06.2021

# Initialize interest-rate tree
r0 = rates[rates.shape[0]-1] #2019-12-06
step, N = 2, 2025.5-2020
tree_nodes = build_tree(r0, K_res, theta_res, sigma_res, step, N)

# Estimate obligation
EN = [int(2021.5-2020)*step] # Period of offer
price, all_induction_steps = estimate_obligation(tree_nodes, fv, coupon, fixed_value, EN, step=step, option='put')

# Compare with last transaction
last_transaction_price = 1.066 * fv
print([np.round(price, 2), last_transaction_price])
