import numpy as np
import matplotlib.pyplot as plt
import statistics
from functions_new import (
    data, 
    all_strategies_function,
)

# Set of firm strategies 
strategies = ["margin opt", "ad averse", "aggressive", "threshold", "cautious"]

# Parameter values
a_b_l_values = [0.5, 1, 2]
a_b_s_values = [0.5, 1, 2]

# Number of simulations
num_simulations = 500


for strategy in strategies:
    print(f"Running simulations for {strategy} strategy...")

    # Store results for all information levels
    results_less = {}
    results_more = {}
    results_full = {}

    # Iterate over each combination of a_b_l and a_b_s values
    for a_b_l in a_b_l_values:
        for a_b_s in a_b_s_values:
            firm_cost_less_list = []
            firm_cost_more_list = []
            firm_cost_full_list = []

            # Run simulations
            for _ in range(num_simulations):
                # Generate data
                real_data = data(a_b_l, a_b_s, n=100, k=10)
                # Apply strategy
                firm_cost_less, firm_cost_more, firm_cost_full = all_strategies_function(strategy, real_data)
                # Collect results
                firm_cost_less_list.append(firm_cost_less)
                firm_cost_more_list.append(firm_cost_more)
                firm_cost_full_list.append(firm_cost_full)

            # Calculate running averages
            results_less[(a_b_l, a_b_s)] = [statistics.mean(firm_cost_less_list[:k+1]) for k in range(len(firm_cost_less_list))]
            results_more[(a_b_l, a_b_s)] = [statistics.mean(firm_cost_more_list[:k+1]) for k in range(len(firm_cost_more_list))]
            results_full[(a_b_l, a_b_s)] = [statistics.mean(firm_cost_full_list[:k+1]) for k in range(len(firm_cost_full_list))]

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
        for idx, ((a_b_l, a_b_s), values) in enumerate(results.items()):
            line, = ax.plot(values, label=fr"$w_{{ij}} \sim \mathrm{{Beta}}({a_b_l},{a_b_l}),\ s_{{ij}} \sim \mathrm{{Beta}}({a_b_s},{a_b_s})$", color=colors[idx])
            key = fr"$w_{{ij}} \sim \mathrm{{Beta}}({a_b_l},{a_b_l}),\ s_{{ij}} \sim \mathrm{{Beta}}({a_b_s},{a_b_s})$"
            if key not in legend_items:
                legend_items[key] = line
        ax.set_xlabel("Simulation Runs")
        ax.set_title(title)

    axes[0].set_ylabel("Running Mean Firm Payoff")
    fig.suptitle(f"Payoff to {strategy.capitalize()} Strategy Across Parameter Combinations", fontsize=14, fontweight='bold')
    legend = fig.legend(legend_items.values(), legend_items.keys(), loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=8, frameon=False, title="Parameter Combinations")
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    #plt.show()

    # Save the figure
    filename = f"{strategy.replace(' ', '_')}_payoff_plot.png"
    #fig.savefig(filename, dpi=300)
    plt.close(fig)
  

    
    
  
    # Store mean firm costs
    mean_firm_costs = np.zeros((3, 3))

    for i_idx, a_b_l in enumerate(a_b_l_values):
        for j_idx, a_b_s in enumerate (a_b_s_values):
            firm_cost_list = []
            for _ in range(num_simulations):
                real_data = data(a_b_l, a_b_s, n=100, k=10)
                firm_cost, _, _ = all_strategies_function(strategy, real_data)
                firm_cost_list.append(firm_cost)
            mean_firm_costs[i_idx, j_idx] = round(statistics.mean(firm_cost_list), 4)

    min_cost = np.min(mean_firm_costs)
    max_cost = np.max(mean_firm_costs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f"a and b={a_b_s}" for a_b_s in a_b_s_values])
    ax.set_yticklabels([f"a and b={a_b_l}" for a_b_l in a_b_l_values])

    for a_b_l in range(3):
        for a_b_s in range(3):
            value = mean_firm_costs[a_b_l, a_b_s]
            intensity = (value - min_cost) / (max_cost - min_cost) if max_cost > min_cost else 0.5
            color = (0.2, 0.4 + 0.6 * intensity, 1)
            ax.add_patch(plt.Rectangle((a_b_s - 0.5, a_b_l - 0.5), 1, 1, color=color))
            ax.text(a_b_s, a_b_l, f"{value:.4f}", ha='center', va='center', color="black" if intensity > 0.5 else "white", fontsize=12, fontweight="bold")


    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()
    plt.suptitle(f"Mean Firm Payoff for {strategy.capitalize()} Firm After 500 Simulations", fontweight="bold")
    plt.xlabel("Parameters for customer sensitivities", fontweight="bold")
    plt.ylabel("Parameters for customer desires", fontweight="bold")
    filename = f"{strategy.replace(' ', '_')}__matrix.png"
    fig.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)
    

  