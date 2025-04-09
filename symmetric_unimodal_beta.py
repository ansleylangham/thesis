import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import gaussian_kde
from functions_new import data, all_strategies_function

# Parameters
a_b_l = 2
a_b_s = 2
simulations = 500

# Set of firm strategies 
strategies = {
    "Aggressive": "aggressive",
    "Ad-Averse": "ad averse",
    "Threshold": "threshold",
    "Cautious": "cautious",
    "Margin-Optimized": "margin opt"
}

# Colors
less_info_color = "#1f77b4"
more_info_color = "#ff7f0e"
full_info_color = "#2ca02c"

# KDE helper
def plot_density(ax, data, color, label):
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 1000)
    ax.plot(x_range, kde(x_range), color=color, lw=2, label=label)
    ax.set_xlabel("Firm Payoff")
    ax.set_ylabel("Density")

# Place to store all simulation results
payoff_data = {}

# Function to simulate all runs per strategy
for strategy_label, strategy_method in strategies.items():
    print(f"Simulating {strategy_label}...")

    firm_cost_less_list = []
    firm_cost_more_list = []
    firm_cost_full_list = []

    for _ in range(simulations):
        real_data = data(a_b_l, a_b_s, n=100, k=10)
        less, more, full = all_strategies_function(strategy_method, real_data)
        firm_cost_less_list.append(less)
        firm_cost_more_list.append(more)
        firm_cost_full_list.append(full)

    payoff_data[strategy_label] = {
        "less": firm_cost_less_list,
        "more": firm_cost_more_list,
        "full": firm_cost_full_list
    }

    ########### DISTRIBUTION PLOTS ###########

    # Side-by-side histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes[0].hist(firm_cost_less_list, bins=30, color=less_info_color, edgecolor='black', alpha=0.7)
    axes[0].set_title("Less Information")
    axes[1].hist(firm_cost_more_list, bins=30, color=more_info_color, edgecolor='black', alpha=0.7)
    axes[1].set_title("More Information")
    axes[2].hist(firm_cost_full_list, bins=30, color=full_info_color, edgecolor='black', alpha=0.7)
    axes[2].set_title("Full Information")
    for ax in axes:
        ax.set_xlabel("Firm Payoff")
    axes[0].set_ylabel("Frequency")
    plt.suptitle(fr"Distribution of {strategy_label} Firm Payoffs Across 500 Simulations for $w_{{ij}} = Beta(2,2)$, $s_{{ij}} = Beta(2,2)$")
    plt.tight_layout()
    plt.savefig(f"{strategy_label.replace(' ', '_')}_histograms_beta_{a_b_l}_{a_b_s}.png", dpi=300)
    plt.show()
    plt.close()

    # Combined histogram
    plt.figure(figsize=(12, 5))
    plt.hist(firm_cost_less_list, bins=30, alpha=0.5, label="Less Info", color=less_info_color, edgecolor='black')
    plt.hist(firm_cost_more_list, bins=30, alpha=0.5, label="More Info", color=more_info_color, edgecolor='black')
    plt.hist(firm_cost_full_list, bins=30, alpha=0.5, label="Full Info", color=full_info_color, edgecolor='black')
    plt.title(fr"Distribution of {strategy_label} Firm Payoffs Across 500 Simulations for $w_{{ij}} = Beta(2,2)$, $s_{{ij}} = Beta(2,2)$")
    plt.xlabel("Firm Payoff")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{strategy_label.replace(' ', '_')}_histogram_combined_beta_{a_b_l}_{a_b_s}.png", dpi=300)
    plt.show()
    plt.close()

    # Combined KDE
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plot_density(ax, firm_cost_less_list, less_info_color, "Less Info")
    plot_density(ax, firm_cost_more_list, more_info_color, "More Info")
    plot_density(ax, firm_cost_full_list, full_info_color, "Full Info")
    plt.title(fr"Density of Margin-Optimized Firm Payoffs Across 500 Simulations for $w_{{ij}} = Beta(2,2)$, $s_{{ij}} = Beta(2,2)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{strategy_label.replace(' ', '_')}_kde_combined_beta_{a_b_l}_{a_b_s}.png", dpi=300)
    plt.show()
    plt.close()

############ MEAN/STDEV BAR CHART ############

# Compute means/stdevs from the stored data
strategies_list = list(strategies.keys())
less_info = [statistics.mean(payoff_data[name]["less"]) for name in strategies_list]
less_info_std = [statistics.stdev(payoff_data[name]["less"]) for name in strategies_list]
more_info = [statistics.mean(payoff_data[name]["more"]) for name in strategies_list]
more_info_std = [statistics.stdev(payoff_data[name]["more"]) for name in strategies_list]
full_info = [statistics.mean(payoff_data[name]["full"]) for name in strategies_list]
full_info_std = [statistics.stdev(payoff_data[name]["full"]) for name in strategies_list]

x = np.arange(len(strategies_list))
width = 0.2

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Alternate grey background
for i in range(len(strategies_list)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.5)

ax.errorbar(x - width, less_info, yerr=less_info_std, fmt='o', color=less_info_color, label="Less Info", capsize=5)
ax.errorbar(x, more_info, yerr=more_info_std, fmt='o', color=more_info_color, label="More Info", capsize=5)
ax.errorbar(x + width, full_info, yerr=full_info_std, fmt='o', color=full_info_color, label="Full Info", capsize=5)

# Formatting
ax.yaxis.grid(True, linestyle='dashed', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(strategies_list, fontsize=12)
ax.set_ylabel(r"$\bar{x} \pm \sigma$ Firm Payoffs", fontsize=12)
ax.set_title(r"Mean Firm Payoffs for $w_{i j} \sim Beta(2,2)$, $s_{i j} \sim Beta(2,2)$ After 500 Simulations", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(f"mean_payoffs_beta_{a_b_l}_{a_b_s}.png", dpi=300)
plt.show()
