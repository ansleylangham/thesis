import numpy as np
import matplotlib.pyplot as plt
import statistics
import random

def bernoulli_random(p):
        return 1 if random.random() <p else 0
        #random.random will return pseudo-random number in [0,1)


def data(a_b_l, a_b_s, n=100, k=10):
    rho = 0.5
    subset_size = n // k  # Number of customers per subset
    
    # Firm's belief about customer preferences
    firm_probs_like_all = np.random.beta(a_b_l, a_b_l, n)
    firm_probs_sens_all = np.random.beta(a_b_s, a_b_s, n)

    customer_bernoulli_like = np.zeros(n) #Creating a place to store customer desires
    customer_bernoulli_sens = np.zeros(n) #Creating a place to store customer sensitivies 

    # True likes (based on firm's belief)
    for i in range(0,n):
        p=firm_probs_like_all[i]
        customer_bernoulli_like[i] = bernoulli_random(p)

    # True sensitivies (based on firm's belief)
    for i in range(0,n):
        p=firm_probs_sens_all[i]
        customer_bernoulli_sens[i] = bernoulli_random(p)

    #True values of c
    customer_c = np.random.beta(1, 1, n)

    


    firm_probs_like_observed_less = np.zeros(k) #Creating a place to store customer desires in less info case
    firm_probs_sens_observed_less = np.zeros(k) #Creating a place to store customer sensitivies in less info case

    # Randomly select customers for LESS INFO case (firm observes only k customers randomly across all n)
    observed_indices_less = np.random.choice(n, k, replace=False)

    # Probabilities observed by firm in LESS INFO CASE 
    for i in range(0,k):
        firm_probs_like_observed_less[i] = firm_probs_like_all[observed_indices_less[i]]
        firm_probs_sens_observed_less[i] = firm_probs_sens_all[observed_indices_less[i]]


    firm_probs_like_observed_more = [] # Creating a place to store customer desires in more info case
    firm_probs_sens_observed_more = [] # Creating a place to store customer sensitivies in more info case
    observed_indices_more_list = []  # Track indices per subset

    for subset in range(k):
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size 
            # Since range(), used in the next line, is exclusive of end_idx, no need to subtract 1 here

        observed_indices_more = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)
        observed_indices_more_list.append(observed_indices_more)

        firm_probs_like_observed_more.append(firm_probs_like_all[observed_indices_more])
        firm_probs_sens_observed_more.append(firm_probs_sens_all[observed_indices_more])


    return (
        firm_probs_like_all,
        firm_probs_sens_all,
        customer_bernoulli_like,
        customer_bernoulli_sens,
        customer_c,
        firm_probs_like_observed_less,
        firm_probs_sens_observed_less,
        firm_probs_like_observed_more,
        firm_probs_sens_observed_more,
        observed_indices_less,
        observed_indices_more_list,  
    )


def all_strategies_function(method, data, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    subset_size = n // k  # Number of customers per subset

    (
    firm_probs_like_all,
    firm_probs_sens_all,
    customer_bernoulli_like,
    customer_bernoulli_sens,
    customer_c,
    firm_probs_like_observed_less,
    firm_probs_sens_observed_less,
    firm_probs_like_observed_more,
    firm_probs_sens_observed_more,
    observed_indices_less,
    observed_indices_more_list,
    ) = data
    

    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Calculate the number of observed customers who meet the condition under each of the strategies 
    if method == "margin opt":
        num_observed_ad_shown_less = np.sum(firm_probs_like_observed_less > (delta * firm_probs_sens_observed_less))
    elif method == "ad averse":
        num_observed_ad_shown_less = np.sum(firm_probs_sens_observed_less < 0.5)
    elif method == "aggressive":
        num_observed_ad_shown_less = np.sum(firm_probs_like_observed_less > 0.5)
    elif method == "threshold":
        num_observed_ad_shown_less = np.sum((firm_probs_like_observed_less > 0.5) & (firm_probs_sens_observed_less < 0.5))
    elif method == "cautious":
        num_observed_ad_shown_less = np.sum(firm_probs_like_observed_less > firm_probs_sens_observed_less)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Determine whether to show ads for the entire population based on the percentage of observed customers who meet the condition
    fraction_observed_ad_shown_less = num_observed_ad_shown_less / len(observed_indices_less)


    # Firm applies decision rule to the entire population (will show ad to everyone if more than half of observed customers meet the condition)
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown_less > 0.5 else 0


    

    ### NEW 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers

    for subset in range(k):
        firm_probs_like_observed_subset = firm_probs_like_observed_more[subset]
        firm_probs_sens_observed_subset = firm_probs_sens_observed_more[subset]

    
        if method == "margin opt":
            num_observed_ad_shown_more = np.sum(firm_probs_like_observed_subset > (delta * firm_probs_sens_observed_subset))
        elif method == "ad averse":
            num_observed_ad_shown_more = np.sum(firm_probs_sens_observed_subset < 0.5)
        elif method == "aggressive":
            num_observed_ad_shown_more = np.sum(firm_probs_like_observed_subset > 0.5)
        elif method == "threshold":
            num_observed_ad_shown_more = np.sum((firm_probs_like_observed_subset > 0.5) & (firm_probs_sens_observed_subset < 0.5))
        elif method == "cautious":
            num_observed_ad_shown_more = np.sum(firm_probs_like_observed_subset > firm_probs_sens_observed_subset)

        fraction_observed_ad_shown_more = num_observed_ad_shown_more / len(firm_probs_sens_observed_subset)


        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown_more > 0.5 else 0

    
    ### 3. FULL INFO CASE  ###
    if method == "margin opt":
        if_ad_shown_full = np.where(firm_probs_like_all > (delta * firm_probs_sens_all), 1, 0)
    elif method == "ad averse":
        if_ad_shown_full = np.where(firm_probs_sens_all < 0.5, 1, 0)
    elif method == "aggressive":
        if_ad_shown_full = np.where(firm_probs_like_all > 0.5, 1, 0)
    elif method == "threshold":
        if_ad_shown_full = np.where((firm_probs_like_all > 0.5) & (firm_probs_sens_all < 0.5), 1, 0)
    elif method == "cautious":
        if_ad_shown_full = np.where(firm_probs_like_all > firm_probs_sens_all, 1, 0)


    # Utility calculations for all info levels
    def calculate_utility(if_ad_shown_full):
        cus_utility = 0
        firm_utility = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (customer_bernoulli_like[i] - (customer_c[i] * customer_bernoulli_sens[i]))
            firm_utility += if_ad_shown_full[i] * ((customer_bernoulli_like[i] - (delta * customer_c[i] * customer_bernoulli_sens[i])) - alpha)
            cus_utility += cost_increment
        return cus_utility / n, firm_utility

    cust_cost_less, firm_cost_less = calculate_utility(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_utility(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_utility(if_ad_shown_full)

    print(f"Low Info Case - Customer Utility: {cust_cost_less:.4f}, Firm Utility: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Utility: {cust_cost_more:.4f}, Firm Utility: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Utility: {cust_cost_full:.4f}, Firm Utility: {firm_cost_full:.4f}")



    return firm_cost_less, firm_cost_more, firm_cost_full






