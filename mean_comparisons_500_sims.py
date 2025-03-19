import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import gaussian_kde

from functions import aggressive_comparison_function 
from functions import ad_averse_comparison_function 
from functions import threshold_comparison_function 
from functions import cautious_comparison_function
from functions import margin_opt_comparison_function 

a_b_l = 1
a_b_s = 1 

# Function to simulate 500 runs and compute mean payoffs with standard deviation
def calculate_mean_payoffs(strategy_function):
    simulations = 500
    firm_cost_less_list = []
    firm_cost_more_list = []
    firm_cost_full_list = []
    
    for _ in range(simulations):
        firm_cost_less, firm_cost_more, firm_cost_full = strategy_function(a_b_l, a_b_s)
        firm_cost_less_list.append(firm_cost_less)
        firm_cost_more_list.append(firm_cost_more)
        firm_cost_full_list.append(firm_cost_full)
    
    return (
        statistics.mean(firm_cost_less_list), statistics.stdev(firm_cost_less_list),
        statistics.mean(firm_cost_more_list), statistics.stdev(firm_cost_more_list),
        statistics.mean(firm_cost_full_list), statistics.stdev(firm_cost_full_list)
    )

# Compute mean payoffs for all strategies
strategies = {
    "Aggressive": aggressive_comparison_function,
    "Ad-Averse": ad_averse_comparison_function,
    "Threshold": threshold_comparison_function,
    "Cautious": cautious_comparison_function,
    "Margin-Optimizing": margin_opt_comparison_function
}

mean_payoffs = {name: calculate_mean_payoffs(func) for name, func in strategies.items()}

# Data for plotting
strategies_list = list(mean_payoffs.keys())
less_info = [mean_payoffs[name][0] for name in strategies_list]
less_info_std = [mean_payoffs[name][1] for name in strategies_list]
more_info = [mean_payoffs[name][2] for name in strategies_list]
more_info_std = [mean_payoffs[name][3] for name in strategies_list]
full_info = [mean_payoffs[name][4] for name in strategies_list]
full_info_std = [mean_payoffs[name][5] for name in strategies_list]

x = np.arange(len(strategies_list))  # Strategies go on x axis 
width = 0.2  # Width of bars to ensure they are side by side

# Colors (for consistency with other plots)
less_info_color = "#1f77b4"  # Blue
more_info_color = "#ff7f0e"  # Orange
full_info_color = "#2ca02c"  # Green

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Alternating grey and white background colors
for i in range(len(strategies_list)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.5)

# Plot points with error bars, offsetting them to be side by side
ax.errorbar(x - width, less_info, yerr=less_info_std, fmt='o', color=less_info_color, label="Less Info", capsize=5)
ax.errorbar(x, more_info, yerr=more_info_std, fmt='o', color=more_info_color, label="More Info", capsize=5)
ax.errorbar(x + width, full_info, yerr=full_info_std, fmt='o', color=full_info_color, label="Full Info", capsize=5)

# Formatting
ax.yaxis.grid(True, linestyle='dashed', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(strategies_list, fontsize=12)
ax.set_ylabel(r"$\bar{x} \pm \sigma$ Firm Payoffs", fontsize=12)
ax.set_title(r"Mean Firm Payoffs for $w_{i j} = Beta(1,1)$ and $s_{i j} = Beta(1,1)$ After 500 Simulations", fontsize=14)
ax.legend()

# Show plot
plt.show()