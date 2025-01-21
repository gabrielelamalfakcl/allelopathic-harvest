import torch

class FairnessMetrics:
    def __init__(self):
        pass
        
    def demographic_parity(self, state_values_not_protected, state_values_protected):
        """
        Compute demographic parity metric on the average state values for each episode.
        Input: state values (expected rewards) for not protected and protected groups.
        Output: demographic parity metric (single value for each episode)
        """
        # Ensure inputs are tensors
        if isinstance(state_values_not_protected, float):
            state_values_not_protected = torch.tensor([state_values_not_protected])
        if isinstance(state_values_protected, float):
            state_values_protected = torch.tensor([state_values_protected])

        # Calculate the mean of the state values for both groups
        mean_state_value_not_protected = torch.mean(state_values_not_protected)
        mean_state_value_protected = torch.mean(state_values_protected)

        # Compute demographic parity
        demographic_parity = mean_state_value_not_protected - mean_state_value_protected

        # Normalize by the total state values to avoid disproportionate effects
        total_values = torch.max(mean_state_value_not_protected + mean_state_value_protected, torch.tensor(1.0))
        norm_demographic_parity = demographic_parity / total_values
        
        return demographic_parity, norm_demographic_parity
    
    def conditional_statistical_parity(self, state_values_not_protected_G1, state_values_protected_G1, 
                                       state_values_not_protected_G2, state_values_protected_G2):
        """
        Compute conditional statistical parity metric on the average state values for each episode.
        Input: state values (expected rewards) for each subgroup (two groups G1 and G2).
        Output: conditional stat parity metric (single value for each subgroup for each episode)
        """
        # Ensure inputs are tensors
        if isinstance(state_values_not_protected_G1, float):
            state_values_not_protected_G1 = torch.tensor([state_values_not_protected_G1])
        if isinstance(state_values_protected_G1, float):
            state_values_protected_G1 = torch.tensor([state_values_protected_G1])
        if isinstance(state_values_not_protected_G2, float):
            state_values_not_protected_G2 = torch.tensor([state_values_not_protected_G2])
        if isinstance(state_values_protected_G2, float):
            state_values_protected_G2 = torch.tensor([state_values_protected_G2])

        # Calculate the mean state values for each subgroup
        mean_state_value_not_protected_G1 = torch.mean(state_values_not_protected_G1)
        mean_state_value_protected_G1 = torch.mean(state_values_protected_G1)

        mean_state_value_not_protected_G2 = torch.mean(state_values_not_protected_G2)
        mean_state_value_protected_G2 = torch.mean(state_values_protected_G2)

        # Compute conditional statistical parity for each group
        conditional_stat_parity_G1 = mean_state_value_not_protected_G1 - mean_state_value_protected_G1
        conditional_stat_parity_G2 = mean_state_value_not_protected_G2 - mean_state_value_protected_G2

        # Normalize by the total state values to avoid disproportionate effects
        total_values_G1 = torch.max(mean_state_value_not_protected_G1 + mean_state_value_protected_G1, torch.tensor(1.0))
        total_values_G2 = torch.max(mean_state_value_not_protected_G2 + mean_state_value_protected_G2, torch.tensor(1.0))

        norm_csp_G1 = conditional_stat_parity_G1 / total_values_G1
        norm_csp_G2 = conditional_stat_parity_G2 / total_values_G2

        return conditional_stat_parity_G1, conditional_stat_parity_G2, norm_csp_G1, norm_csp_G2
