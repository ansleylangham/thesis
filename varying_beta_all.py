import numpy as np
import matplotlib.pyplot as plt
import statistics

from functions import (
    aggressive_comparison_function,
    ad_averse_comparison_function,
    threshold_comparison_function,
    cautious_comparison_function,
    margin_opt_comparison_function,
)

# Define function names and corresponding function references
comparison_functions = {
    "Aggressive": aggressive_comparison_function,
    "Ad Averse": ad_averse_comparison_function,
    "Threshold": threshold_comparison_function,
    "Cautious": cautious_comparison_function,
    "Margin Optimizing": margin_opt_comparison_function,
}

# Parameter values
a_b_l_values = [0.5, 1, 2]
a_b_s_values = [0.5, 1, 2]

for function_name, comparison_function in comparison_functions.items():
    print(f"Running simulations for {function_name} Firm...")
    
    # Store results for all information levels
    results_less = {}
    results_more = {}
    results_full = {}

    # Simulate 500 encounters for each pair of distributions (desires and sensitivities)
    for i in a_b_l_values:
        for j in a_b_s_values:
            firm_cost_less_list = []
            firm_cost_more_list = []
            firm_cost_full_list = []

            for _ in range(500):
                firm_cost_less, firm_cost_more, firm_cost_full = comparison_function(i, j)

                firm_cost_less_list.append(firm_cost_less)
                firm_cost_more_list.append(firm_cost_more)
                firm_cost_full_list.append(firm_cost_full)

            # Calculate running averages
            results_less[(i, j)] = [statistics.mean(firm_cost_less_list[:k+1]) for k in range(len(firm_cost_less_list))]
            results_more[(i, j)] = [statistics.mean(firm_cost_more_list[:k+1]) for k in range(len(firm_cost_more_list))]
            results_full[(i, j)] = [statistics.mean(firm_cost_full_list[:k+1]) for k in range(len(firm_cost_full_list))]

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    info_levels = [
        "Less Information",
        "More Information",
        "Full Information"
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, 9))
    legend_items = {}

    for ax, results, title in zip(axes, [results_less, results_more, results_full], info_levels):
        for idx, ((i, j), values) in enumerate(results.items()):
            line, = ax.plot(values, label=f"desire parameters={i}, sens. parameters={j}", color=colors[idx])
            key = f"desire parameters={i}, sens. parameters={j}"
            if key not in legend_items:
                legend_items[key] = line
        ax.set_xlabel("Simulation Runs")
        ax.set_title(title)

    axes[0].set_ylabel("Running Mean Firm Payoff")
    fig.suptitle(f"Payoff to {function_name} Firm Across Parameter Combinations", fontsize=14, fontweight='bold')
    legend = fig.legend(legend_items.values(), legend_items.keys(), loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=8, frameon=False, title="Parameter Combinations")
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # Store mean firm costs
    mean_firm_costs = np.zeros((3, 3))

    for i_idx, i in enumerate(a_b_l_values):
        for j_idx, j in enumerate(a_b_s_values):
            firm_cost_list = []
            for _ in range(500):
                firm_cost, _, _ = comparison_function(i, j)
                firm_cost_list.append(firm_cost)
            mean_firm_costs[i_idx, j_idx] = round(statistics.mean(firm_cost_list), 4)

    min_cost = np.min(mean_firm_costs)
    max_cost = np.max(mean_firm_costs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f"a and b={j}" for j in a_b_s_values])
    ax.set_yticklabels([f"a and b={i}" for i in a_b_l_values])

    for i in range(3):
        for j in range(3):
            value = mean_firm_costs[i, j]
            intensity = (value - min_cost) / (max_cost - min_cost) if max_cost > min_cost else 0.5
            color = (0.2, 0.4 + 0.6 * intensity, 1)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
            ax.text(j, i, f"{value:.4f}", ha='center', va='center', color="black" if intensity > 0.5 else "white", fontsize=12, fontweight="bold")

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()
    plt.suptitle(f"Mean Firm Payoff for {function_name} Firm After 500 Simulations", fontweight="bold")
    plt.xlabel("Parameters for customer sensitivities", fontweight="bold")
    plt.ylabel("Parameters for customer desires", fontweight="bold")
    plt.show()
