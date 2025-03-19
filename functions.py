import numpy as np
import matplotlib.pyplot as plt
import statistics

def margin_opt_comparison_function(a_b_l, a_b_s, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    rho = 0.5
    
    subset_size = n // k  # Number of customers per subset
    
    # Pulling probabilities for all customers
    firm_probs_like_all = np.random.beta(a_b_l, a_b_l, n)
    firm_probs_sens_all = np.random.beta(a_b_s, a_b_s, n)

    # True customer preferences
    cust_probs_like = np.random.beta(a_b_l, a_b_l, n)
    cust_probs_sens = np.random.beta(a_b_s, a_b_s, n)
    cust_probs_c = np.random.beta(1, 1, n)

    # Correct decision calculation (if the firm had full info & followed the same rule, what would they do)
    correct_ad_decision = np.where(cust_probs_like > (delta * cust_probs_sens), 1, 0)
    
    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Firm observes only k customers randomly across all n
    observed_indices = np.random.choice(n, k, replace=False)
    firm_probs_like_observed = firm_probs_like_all[observed_indices]
    firm_probs_sens_observed = firm_probs_sens_all[observed_indices]
    
    # Determine whether to show ads for the entire population based on observed customers
    num_observed_ad_shown = np.sum(firm_probs_like_observed > (delta * firm_probs_sens_observed))
    fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

    # Firm applies decision rule to the entire population
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown > 0.5 else 0

    
    ### 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    for subset in range(k):
        # Define the indices for this subset
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size

        # Firm observes only a portion of each subset
        observed_indices = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)


        # Extract observed probabilities
        firm_probs_like_observed = firm_probs_like_all[observed_indices]
        firm_probs_sens_observed = firm_probs_sens_all[observed_indices]

        # Determine whether to show ads for the entire subset based on observed customers
        num_observed_ad_shown = np.sum(firm_probs_like_observed > (delta * firm_probs_sens_observed))
        fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

        # Apply the same decision to ALL customers in this subset (even those not observed)
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown > 0.5 else 0

    
    
    ### 3. FULL INFO CASE  ###
    if_ad_shown_full = np.where(firm_probs_like_all > (delta * firm_probs_sens_all), 1, 0)

    # Binary classification of preferences and sensitivities
    binary_true_like = np.where(cust_probs_like > 0.5, 1, 0)
    binary_true_sens = np.where(cust_probs_sens > 0.5, 1, 0)


    # Cost calculations for all info levels
    def calculate_cost(if_ad_shown_full):
        cus_cost = 0
        firm_cost = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (binary_true_like[i] - (cust_probs_c[i] * binary_true_sens[i]))
            firm_cost += if_ad_shown_full[i] * ((binary_true_like[i] - (delta * cust_probs_c[i] * binary_true_sens[i])) - alpha)
            cus_cost += cost_increment
        return cus_cost / n, firm_cost

    cust_cost_less, firm_cost_less = calculate_cost(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_cost(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_cost(if_ad_shown_full)

    print(f"Low Info Case - Customer Cost: {cust_cost_less:.4f}, Firm Cost: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Cost: {cust_cost_more:.4f}, Firm Cost: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Cost: {cust_cost_full:.4f}, Firm Cost: {firm_cost_full:.4f}")

    return firm_cost_less, firm_cost_more, firm_cost_full

def ad_averse_comparison_function(a_b_l, a_b_s, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    rho = 0.5

    subset_size = n // k  # Number of customers per subset
    
    # Pulling probabilities for all customers
    firm_probs_sens_all = np.random.beta(a_b_s, a_b_s, n)

    # True customer preferences
    cust_probs_like = np.random.beta(a_b_l, a_b_l, n)
    cust_probs_sens = np.random.beta(a_b_s, a_b_s, n)
    cust_probs_c = np.random.beta(1, 1, n)


    
    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Firm observes only k customers randomly across all n
    observed_indices = np.random.choice(n, k, replace=False)
    firm_probs_sens_observed = firm_probs_sens_all[observed_indices]
    
    # Determine whether to show ads for the entire population based on observed customers
    num_observed_ad_shown = np.sum(firm_probs_sens_observed < 0.5)
    fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

    # Firm applies decision rule to the entire population
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown > 0.5 else 0

    # Correct decision calculation (if the firm had full info & followed the same rule, what would they do)
    correct_ad_decision_less = np.where(cust_probs_sens < 0.5, 1, 0)

    
    ### 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    for subset in range(k):
        # Define the indices for this subset
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size

        # Firm observes only a portion of each subset
        observed_indices = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)


        # Extract observed probabilities
        firm_probs_sens_observed = firm_probs_sens_all[observed_indices]

        # Determine whether to show ads for the entire subset based on observed customers
        num_observed_ad_shown = np.sum(firm_probs_sens_observed < 0.5)
        fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

        # Apply the same decision to ALL customers in this subset (even those not observed)
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown > 0.5 else 0


    # Creating new list of firm decisions based on the true customer values (FIRM DOES NOT OBSERVE THIS)
    correct_ad_decision_more = np.where(cust_probs_sens < 0.5, 1, 0)
    
    
    ### 3. FULL INFO CASE  ###
    if_ad_shown_full = np.where(firm_probs_sens_all < 0.5, 1, 0)

    # True customer preferences
    cust_probs_like = np.random.beta(2, 2, n)
    cust_probs_sens = np.random.beta(2, 2, n)
    cust_probs_c = np.random.beta(1, 1, n)

    # Binary classification of preferences and sensitivities
    binary_true_like = np.where(cust_probs_like > 0.5, 1, 0)
    binary_true_sens = np.where(cust_probs_sens > 0.5, 1, 0)

    # Correct decision calculation
    correct_ad_decision_full = np.where(cust_probs_sens < 0.5, 1, 0)

    # Cost calculations for all methods
    def calculate_cost(if_ad_shown_full):
        cus_cost = 0
        firm_cost = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (binary_true_like[i] - (cust_probs_c[i] * binary_true_sens[i]))
            firm_cost += if_ad_shown_full[i] * ((binary_true_like[i] - (delta * cust_probs_c[i] * binary_true_sens[i])) - alpha)
            cus_cost += cost_increment
        return cus_cost / n, firm_cost

    cust_cost_less, firm_cost_less = calculate_cost(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_cost(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_cost(if_ad_shown_full)

    print(f"Low Info Case - Customer Cost: {cust_cost_less:.4f}, Firm Cost: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Cost: {cust_cost_more:.4f}, Firm Cost: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Cost: {cust_cost_full:.4f}, Firm Cost: {firm_cost_full:.4f}")

    return firm_cost_less, firm_cost_more, firm_cost_full

def aggressive_comparison_function(a_b_l, a_b_s, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    rho = 0.5

    subset_size = n // k  # Number of customers per subset
    
    # Pulling probabilities for all customers
    firm_probs_like_all = np.random.beta(a_b_l, a_b_l, n)


    # True customer preferences
    cust_probs_like = np.random.beta(a_b_l, a_b_l, n)
    cust_probs_sens = np.random.beta(a_b_s, a_b_s, n)
    cust_probs_c = np.random.beta(1, 1, n)

    
    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Firm observes only k customers randomly across all n
    observed_indices = np.random.choice(n, k, replace=False)
    firm_probs_like_observed = firm_probs_like_all[observed_indices]
    
    # Determine whether to show ads for the entire population based on observed customers
    num_observed_ad_shown = np.sum(firm_probs_like_observed > 0.5)
    fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

    # Firm applies decision rule to the entire population
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown > 0.5 else 0

    # Correct decision calculation (if the firm had full info & followed the same rule, what would they do)
    correct_ad_decision_less = np.where(cust_probs_like > 0.5, 1, 0)

    
    ### 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    for subset in range(k):
        # Define the indices for this subset
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size

        # Firm observes only a portion of each subset
        observed_indices = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)


        # Extract observed probabilities
        firm_probs_like_observed = firm_probs_like_all[observed_indices]

        # Determine whether to show ads for the entire subset based on observed customers
        num_observed_ad_shown = np.sum(firm_probs_like_observed > 0.5)
        fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

        # Apply the same decision to ALL customers in this subset (even those not observed)
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown > 0.5 else 0


    # Creating new list of firm decisions based on the true customer values (FIRM DOES NOT OBSERVE THIS)
    correct_ad_decision_more = np.where(cust_probs_like > 0.5, 1, 0)
    
    
    ### 3. FULL INFO CASE  ###
    if_ad_shown_full = np.where(firm_probs_like_all > 0.5, 1, 0)

    # True customer preferences
    cust_probs_like = np.random.beta(2, 2, n)
    cust_probs_sens = np.random.beta(2, 2, n)
    cust_probs_c = np.random.beta(1, 1, n)

    # Binary classification of preferences and sensitivities
    binary_true_like = np.where(cust_probs_like > 0.5, 1, 0)
    binary_true_sens = np.where(cust_probs_sens > 0.5, 1, 0)

    # Correct decision calculation
    correct_ad_decision_full = np.where(cust_probs_like > 0.5, 1, 0)

    # Cost calculations for all methods
    def calculate_cost(if_ad_shown_full):
        cus_cost = 0
        firm_cost = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (binary_true_like[i] - (cust_probs_c[i] * binary_true_sens[i]))
            firm_cost += if_ad_shown_full[i] * ((binary_true_like[i] - (delta * cust_probs_c[i] * binary_true_sens[i])) - alpha)
            cus_cost += cost_increment
        return cus_cost / n, firm_cost

    cust_cost_less, firm_cost_less = calculate_cost(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_cost(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_cost(if_ad_shown_full)

    print(f"Low Info Case - Customer Cost: {cust_cost_less:.4f}, Firm Cost: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Cost: {cust_cost_more:.4f}, Firm Cost: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Cost: {cust_cost_full:.4f}, Firm Cost: {firm_cost_full:.4f}")

    return firm_cost_less, firm_cost_more, firm_cost_full

def threshold_comparison_function(a_b_l, a_b_s, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    rho = 0.5

    subset_size = n // k  # Number of customers per subset
    
    # Pulling probabilities for all customers
    firm_probs_like_all = np.random.beta(a_b_l, a_b_l, n)
    firm_probs_sens_all = np.random.beta(a_b_s, a_b_s, n)

    # True customer preferences
    cust_probs_like = np.random.beta(a_b_l, a_b_l, n)
    cust_probs_sens = np.random.beta(a_b_s, a_b_s, n)
    cust_probs_c = np.random.beta(1, 1, n)
    
    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Firm observes only k customers randomly across all n
    observed_indices = np.random.choice(n, k, replace=False)
    firm_probs_like_observed = firm_probs_like_all[observed_indices]
    firm_probs_sens_observed = firm_probs_sens_all[observed_indices]
    
    # Determine whether to show ads for the entire population based on observed customers
    num_observed_ad_shown = np.sum((firm_probs_like_observed > 0.5) & (firm_probs_sens_observed < 0.5))
    fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

    # Firm applies decision rule to the entire population
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown > 0.5 else 0

    # Correct decision calculation (if the firm had full info & followed the same rule, what would they do)
    correct_ad_decision_less = np.where((cust_probs_like > 0.5) & (cust_probs_sens < 0.5), 1, 0)

    
    ### 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    for subset in range(k):
        # Define the indices for this subset
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size

        # Firm observes only a portion of each subset
        observed_indices = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)


        # Extract observed probabilities
        firm_probs_like_observed = firm_probs_like_all[observed_indices]
        firm_probs_sens_observed = firm_probs_sens_all[observed_indices]

        # Determine whether to show ads for the entire subset based on observed customers
        num_observed_ad_shown = np.sum((firm_probs_like_observed > 0.5) & (firm_probs_sens_observed < 0.5))
        fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

        # Apply the same decision to ALL customers in this subset (even those not observed)
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown > 0.5 else 0


    # Creating new list of firm decisions based on the true customer values (FIRM DOES NOT OBSERVE THIS)
    correct_ad_decision_more = np.where((cust_probs_like > 0.5) & (cust_probs_sens < 0.5), 1, 0)
    
    
    ### 3. FULL INFO CASE  ###
    if_ad_shown_full = np.where((firm_probs_like_all > 0.5) & (firm_probs_sens_all < 0.5), 1, 0)

    # Binary classification of preferences and sensitivities
    binary_true_like = np.where(cust_probs_like > 0.5, 1, 0)
    binary_true_sens = np.where(cust_probs_sens > 0.5, 1, 0)

    # Correct decision calculation
    correct_ad_decision_full = np.where((cust_probs_like > 0.5) & (cust_probs_sens < 0.5), 1, 0)

    # Cost calculations for all methods
    def calculate_cost(if_ad_shown_full):
        cus_cost = 0
        firm_cost = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (binary_true_like[i] - (cust_probs_c[i] * binary_true_sens[i]))
            firm_cost += if_ad_shown_full[i] * ((binary_true_like[i] - (delta * cust_probs_c[i] * binary_true_sens[i])) - alpha)
            cus_cost += cost_increment
        return cus_cost / n, firm_cost

    cust_cost_less, firm_cost_less = calculate_cost(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_cost(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_cost(if_ad_shown_full)

    print(f"Low Info Case - Customer Cost: {cust_cost_less:.4f}, Firm Cost: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Cost: {cust_cost_more:.4f}, Firm Cost: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Cost: {cust_cost_full:.4f}, Firm Cost: {firm_cost_full:.4f}")

    return firm_cost_less, firm_cost_more, firm_cost_full

def cautious_comparison_function(a_b_l, a_b_s, n=100, k=10):
    alpha = 0.1  # Cost to firm to show ad ∈ [0,0.5]
    delta = 0.8  # Firm's discounting of customer's utility ∈ [0,1]
    rho = 0.5

    subset_size = n // k  # Number of customers per subset
    
    # Pulling probabilities for all customers
    firm_probs_like_all = np.random.beta(a_b_l, a_b_l, n)
    firm_probs_sens_all = np.random.beta(a_b_s, a_b_s, n)

    # True customer preferences
    cust_probs_like = np.random.beta(a_b_l, a_b_l, n)
    cust_probs_sens = np.random.beta(a_b_s, a_b_s, n)
    cust_probs_c = np.random.beta(1, 1, n)

    
    ### 1. LESS INFO CASE (SUBSET-BASED GENERAL RULE FOR ALL) ###
    if_ad_shown_less = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    # Firm observes only k customers randomly across all n
    observed_indices = np.random.choice(n, k, replace=False)
    firm_probs_like_observed = firm_probs_like_all[observed_indices]
    firm_probs_sens_observed = firm_probs_sens_all[observed_indices]
    
    # Determine whether to show ads for the entire population based on observed customers
    num_observed_ad_shown = np.sum(firm_probs_like_observed > firm_probs_sens_observed)
    fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

    # Firm applies decision rule to the entire population
    if_ad_shown_less[:] = 1 if fraction_observed_ad_shown > 0.5 else 0

    # Correct decision calculation (if the firm had full info & followed the same rule, what would they do)
    correct_ad_decision_less = np.where(cust_probs_like > cust_probs_sens, 1, 0)


     ### 2. MORE INFO CASE (SUBSET-SPECIFIC DECISIONS) ###
    if_ad_shown_more = np.zeros(n)  # Creating a place to store ad decisions for all customers
    
    for subset in range(k):
        # Define the indices for this subset
        start_idx = subset * subset_size
        end_idx = (subset + 1) * subset_size

        # Firm observes only a portion of each subset
        observed_indices = np.random.choice(range(start_idx, end_idx), int(subset_size * rho), replace=False)


        # Extract observed probabilities
        firm_probs_like_observed = firm_probs_like_all[observed_indices]
        firm_probs_sens_observed = firm_probs_sens_all[observed_indices]

        # Determine whether to show ads for the entire subset based on observed customers
        num_observed_ad_shown = np.sum(firm_probs_like_observed > firm_probs_sens_observed)
        fraction_observed_ad_shown = num_observed_ad_shown / len(observed_indices)

        # Apply the same decision to ALL customers in this subset (even those not observed)
        if_ad_shown_more[start_idx:end_idx] = 1 if fraction_observed_ad_shown > 0.5 else 0

    # Creating new list of firm decisions based on the true customer values (FIRM DOES NOT OBSERVE THIS)
    correct_ad_decision_more = np.where(cust_probs_like > cust_probs_sens, 1, 0)
    

    ### 3. METHOD 3: FULL KNOWLEDGE (PERFECT INFORMATION) ###
    if_ad_shown_full = np.where(firm_probs_like_all > firm_probs_sens_all, 1, 0)

    # True customer preferences
    cust_probs_like = np.random.beta(2, 2, n)
    cust_probs_sens = np.random.beta(2, 2, n)
    cust_probs_c = np.random.beta(1, 1, n)

    # Binary classification of preferences and sensitivities
    binary_true_like = np.where(cust_probs_like > 0.5, 1, 0)
    binary_true_sens = np.where(cust_probs_sens > 0.5, 1, 0)

    # Correct decision calculation
    correct_ad_decision_full = np.where(cust_probs_like > cust_probs_sens, 1, 0)



   # Cost calculations for all methods
    def calculate_cost(if_ad_shown_full):
        cus_cost = 0
        firm_cost = 0
        for i in range(n):
            cost_increment = if_ad_shown_full[i] * (binary_true_like[i] - (cust_probs_c[i] * binary_true_sens[i]))
            firm_cost += if_ad_shown_full[i] * ((binary_true_like[i] - (delta * cust_probs_c[i] * binary_true_sens[i])) - alpha)
            cus_cost += cost_increment
        return cus_cost / n, firm_cost

    cust_cost_less, firm_cost_less = calculate_cost(if_ad_shown_less)
    cust_cost_more, firm_cost_more = calculate_cost(if_ad_shown_more)
    cust_cost_full, firm_cost_full = calculate_cost(if_ad_shown_full)

    print(f"Low Info Case - Customer Cost: {cust_cost_less:.4f}, Firm Cost: {firm_cost_less:.4f}")
    print(f"More Info Case - Customer Cost: {cust_cost_more:.4f}, Firm Cost: {firm_cost_more:.4f}")
    print(f"Full Info Case - Customer Cost: {cust_cost_full:.4f}, Firm Cost: {firm_cost_full:.4f}")

    return firm_cost_less, firm_cost_more, firm_cost_full

