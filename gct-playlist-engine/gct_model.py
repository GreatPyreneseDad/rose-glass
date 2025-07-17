import pandas as pd

class GCTModel:
    """Compute coherence metrics for playlist transitions."""

    def __init__(self, psi_weight=1.0, rho_weight=1.0, q_opt_weight=1.0, f_weight=1.0, alpha_weight=1.0):
        self.psi_weight = psi_weight
        self.rho_weight = rho_weight
        self.q_opt_weight = q_opt_weight
        self.f_weight = f_weight
        self.alpha_weight = alpha_weight

    def compute_coherence(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply GCT formulas to compute coherence metrics between tracks."""
        if features_df.empty:
            return features_df

        numeric_cols = features_df.select_dtypes(include="number").columns
        weights = [self.psi_weight, self.rho_weight, self.q_opt_weight, self.f_weight, self.alpha_weight]
        # If there are more columns than weights, repeat the last weight
        while len(weights) < len(numeric_cols):
            weights.append(weights[-1])

        weighted = features_df[numeric_cols] * weights[: len(numeric_cols)]
        features_df = features_df.copy()
        features_df["coherence"] = weighted.sum(axis=1)
        return features_df
