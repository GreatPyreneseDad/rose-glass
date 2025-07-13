#!/usr/bin/env python3
"""
GCT Validation Suite
Implements comprehensive validation framework for reliability and validity testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import logging


@dataclass
class ValidationResult:
    """Container for validation results"""
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float]
    n_samples: int
    interpretation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class GCTValidator:
    """Comprehensive validation suite for GCT measurements"""
    
    def __init__(self, database_manager=None):
        self.db = database_manager
        self.logger = logging.getLogger(__name__)
        
        # Reliability thresholds
        self.reliability_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'questionable': 0.6,
            'poor': 0.5
        }
        
        # Validity correlation thresholds
        self.validity_thresholds = {
            'strong': 0.7,
            'moderate': 0.5,
            'weak': 0.3,
            'negligible': 0.1
        }
    
    def test_retest_reliability(self, 
                               subjects: List[str], 
                               interval_days: int = 14,
                               variable: str = 'coherence') -> ValidationResult:
        """
        Assess measurement stability over time using ICC
        
        Args:
            subjects: List of subject identifiers
            interval_days: Days between test and retest
            variable: Which variable to test (coherence, psi, rho, etc.)
        
        Returns:
            ValidationResult with ICC and interpretation
        """
        if not self.db:
            raise ValueError("Database manager required for test-retest analysis")
        
        # Get test scores
        test_scores = self._get_scores(subjects, variable, "initial")
        
        # Get retest scores
        retest_time = f"initial+{interval_days}d"
        retest_scores = self._get_scores(subjects, variable, retest_time)
        
        # Ensure paired data
        paired_data = self._pair_scores(test_scores, retest_scores)
        
        if len(paired_data) < 3:
            return ValidationResult(
                metric_name="test_retest_reliability",
                value=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                n_samples=len(paired_data),
                interpretation="Insufficient data for reliability analysis"
            )
        
        # Calculate ICC(2,1) - two-way random effects, single measurement
        icc_value, ci_lower, ci_upper, p_value = self._calculate_icc(
            paired_data['test'], 
            paired_data['retest']
        )
        
        interpretation = self._interpret_reliability(icc_value)
        
        return ValidationResult(
            metric_name=f"test_retest_reliability_{variable}",
            value=icc_value,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_samples=len(paired_data),
            interpretation=interpretation
        )
    
    def internal_consistency(self, 
                           item_responses: pd.DataFrame,
                           scale_name: str = "GCT") -> ValidationResult:
        """
        Calculate Cronbach's alpha for internal consistency
        
        Args:
            item_responses: DataFrame with items as columns, subjects as rows
            scale_name: Name of the scale being tested
        
        Returns:
            ValidationResult with Cronbach's alpha
        """
        n_items = item_responses.shape[1]
        n_subjects = item_responses.shape[0]
        
        if n_items < 2 or n_subjects < 2:
            return ValidationResult(
                metric_name=f"internal_consistency_{scale_name}",
                value=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=None,
                n_samples=n_subjects,
                interpretation="Insufficient items or subjects"
            )
        
        # Calculate Cronbach's alpha
        item_variances = item_responses.var(axis=0, ddof=1)
        total_variance = item_responses.sum(axis=1).var(ddof=1)
        
        alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_alpha_ci(item_responses)
        
        interpretation = self._interpret_reliability(alpha)
        
        return ValidationResult(
            metric_name=f"internal_consistency_{scale_name}",
            value=alpha,
            confidence_interval=(ci_lower, ci_upper),
            p_value=None,
            n_samples=n_subjects,
            interpretation=f"Cronbach's α = {alpha:.3f} - {interpretation}"
        )
    
    def convergent_validity(self, 
                          gct_scores: Dict[str, float],
                          external_measure: Dict[str, float],
                          measure_name: str,
                          hypothesized_correlation: float = 0.5) -> ValidationResult:
        """
        Test convergent validity against external measures
        
        Args:
            gct_scores: Dictionary of subject_id: gct_score
            external_measure: Dictionary of subject_id: external_score
            measure_name: Name of external measure
            hypothesized_correlation: Expected correlation strength
        
        Returns:
            ValidationResult with correlation and significance test
        """
        # Get paired scores
        paired_scores = []
        for subject_id in gct_scores:
            if subject_id in external_measure:
                paired_scores.append((
                    gct_scores[subject_id],
                    external_measure[subject_id]
                ))
        
        if len(paired_scores) < 3:
            return ValidationResult(
                metric_name=f"convergent_validity_{measure_name}",
                value=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                n_samples=len(paired_scores),
                interpretation="Insufficient paired data"
            )
        
        gct_vals, external_vals = zip(*paired_scores)
        
        # Calculate correlation
        r, p_value = stats.pearsonr(gct_vals, external_vals)
        
        # Confidence interval for correlation
        ci_lower, ci_upper = self._correlation_confidence_interval(r, len(paired_scores))
        
        # Test against hypothesized correlation
        z_score = self._fisher_z_test(r, hypothesized_correlation, len(paired_scores))
        hypothesis_p = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        interpretation = self._interpret_validity_correlation(r, hypothesis_p)
        
        return ValidationResult(
            metric_name=f"convergent_validity_{measure_name}",
            value=r,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_samples=len(paired_scores),
            interpretation=interpretation
        )
    
    def discriminant_validity(self,
                            gct_scores: Dict[str, float],
                            unrelated_measure: Dict[str, float],
                            measure_name: str,
                            max_correlation: float = 0.3) -> ValidationResult:
        """
        Test discriminant validity (low correlation with unrelated measures)
        
        Args:
            gct_scores: Dictionary of subject_id: gct_score
            unrelated_measure: Dictionary of subject_id: unrelated_score
            measure_name: Name of unrelated measure
            max_correlation: Maximum acceptable correlation
        
        Returns:
            ValidationResult with correlation and interpretation
        """
        # Get paired scores
        paired_scores = []
        for subject_id in gct_scores:
            if subject_id in unrelated_measure:
                paired_scores.append((
                    gct_scores[subject_id],
                    unrelated_measure[subject_id]
                ))
        
        if len(paired_scores) < 3:
            return ValidationResult(
                metric_name=f"discriminant_validity_{measure_name}",
                value=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                n_samples=len(paired_scores),
                interpretation="Insufficient paired data"
            )
        
        gct_vals, unrelated_vals = zip(*paired_scores)
        
        # Calculate correlation
        r, p_value = stats.pearsonr(gct_vals, unrelated_vals)
        
        # Confidence interval
        ci_lower, ci_upper = self._correlation_confidence_interval(r, len(paired_scores))
        
        # Test if correlation is significantly less than threshold
        is_discriminant = abs(r) < max_correlation
        
        interpretation = (
            f"Good discriminant validity - correlation ({r:.3f}) below threshold"
            if is_discriminant else
            f"Poor discriminant validity - correlation ({r:.3f}) exceeds threshold"
        )
        
        return ValidationResult(
            metric_name=f"discriminant_validity_{measure_name}",
            value=r,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_samples=len(paired_scores),
            interpretation=interpretation
        )
    
    def predictive_validity(self,
                          baseline_scores: Dict[str, float],
                          outcomes: Dict[str, float],
                          outcome_name: str,
                          time_lag_days: int) -> ValidationResult:
        """
        Test ability to predict future outcomes
        
        Args:
            baseline_scores: Dictionary of subject_id: baseline_gct_score
            outcomes: Dictionary of subject_id: outcome_measure
            outcome_name: Name of outcome being predicted
            time_lag_days: Days between baseline and outcome
        
        Returns:
            ValidationResult with predictive correlation
        """
        # Get paired scores
        paired_data = []
        for subject_id in baseline_scores:
            if subject_id in outcomes:
                paired_data.append((
                    baseline_scores[subject_id],
                    outcomes[subject_id]
                ))
        
        if len(paired_data) < 10:  # Need more data for prediction
            return ValidationResult(
                metric_name=f"predictive_validity_{outcome_name}",
                value=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                n_samples=len(paired_data),
                interpretation="Insufficient data for predictive analysis"
            )
        
        baseline_vals, outcome_vals = zip(*paired_data)
        
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            baseline_vals, outcome_vals
        )
        
        # R-squared for variance explained
        r_squared = r_value ** 2
        
        # Confidence interval for R-squared
        ci_lower, ci_upper = self._r_squared_confidence_interval(
            r_squared, len(paired_data)
        )
        
        interpretation = (
            f"GCT scores explain {r_squared*100:.1f}% of variance in {outcome_name} "
            f"after {time_lag_days} days (p={p_value:.3f})"
        )
        
        return ValidationResult(
            metric_name=f"predictive_validity_{outcome_name}",
            value=r_squared,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            n_samples=len(paired_data),
            interpretation=interpretation
        )
    
    def factor_structure_validity(self,
                                item_responses: pd.DataFrame,
                                expected_factors: int = 4) -> Dict[str, Any]:
        """
        Validate factor structure using exploratory factor analysis
        
        Args:
            item_responses: DataFrame with items as columns
            expected_factors: Expected number of factors (4 for psi, rho, q, f)
        
        Returns:
            Dictionary with factor loadings and fit statistics
        """
        from sklearn.decomposition import FactorAnalysis
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(item_responses)
        
        # Perform factor analysis
        fa = FactorAnalysis(n_components=expected_factors, random_state=42)
        fa.fit(scaled_data)
        
        # Get loadings
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor_{i+1}' for i in range(expected_factors)],
            index=item_responses.columns
        )
        
        # Calculate fit statistics
        explained_variance = fa.score(scaled_data)
        
        return {
            'loadings': loadings,
            'explained_variance': explained_variance,
            'n_factors': expected_factors,
            'interpretation': self._interpret_factor_structure(loadings)
        }
    
    def _calculate_icc(self, test_scores: List[float], 
                      retest_scores: List[float]) -> Tuple[float, float, float, float]:
        """Calculate ICC(2,1) with confidence intervals"""
        # Convert to numpy arrays
        scores1 = np.array(test_scores)
        scores2 = np.array(retest_scores)
        
        # Calculate means
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        grand_mean = (mean1 + mean2) / 2
        
        # Calculate sum of squares
        n = len(scores1)
        SST = np.sum((scores1 - grand_mean)**2) + np.sum((scores2 - grand_mean)**2)
        SSW = np.sum((scores1 - scores2)**2) / 2
        SSB = SST - SSW * n
        
        # Calculate mean squares
        MSB = SSB / (n - 1)
        MSW = SSW / n
        
        # Calculate ICC
        icc = (MSB - MSW) / (MSB + MSW)
        
        # F-test for significance
        F = MSB / MSW
        df1 = n - 1
        df2 = n
        p_value = 1 - stats.f.cdf(F, df1, df2)
        
        # Confidence intervals (simplified)
        ci_lower = max(0, icc - 1.96 * np.sqrt(2*(1-icc)**2/n))
        ci_upper = min(1, icc + 1.96 * np.sqrt(2*(1-icc)**2/n))
        
        return icc, ci_lower, ci_upper, p_value
    
    def _interpret_reliability(self, reliability_value: float) -> str:
        """Interpret reliability coefficient"""
        for label, threshold in self.reliability_thresholds.items():
            if reliability_value >= threshold:
                return f"{label.capitalize()} reliability"
        return "Unacceptable reliability"
    
    def _interpret_validity_correlation(self, r: float, p: float) -> str:
        """Interpret validity correlation"""
        if p > 0.05:
            return f"Non-significant correlation (r={r:.3f}, p={p:.3f})"
        
        strength = "negligible"
        for label, threshold in self.validity_thresholds.items():
            if abs(r) >= threshold:
                strength = label
                break
                
        direction = "positive" if r > 0 else "negative"
        return f"{strength.capitalize()} {direction} correlation (r={r:.3f}, p={p:.3f})"
    
    def _correlation_confidence_interval(self, r: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        # Fisher z transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # 95% CI in z space
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        
        # Transform back to r
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return r_lower, r_upper
    
    def _bootstrap_alpha_ci(self, data: pd.DataFrame, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence interval for Cronbach's alpha"""
        n_subjects = len(data)
        alphas = []
        
        for _ in range(n_bootstrap):
            # Resample subjects with replacement
            indices = np.random.choice(n_subjects, n_subjects, replace=True)
            resampled = data.iloc[indices]
            
            # Calculate alpha for resample
            n_items = resampled.shape[1]
            item_variances = resampled.var(axis=0, ddof=1)
            total_variance = resampled.sum(axis=1).var(ddof=1)
            
            if total_variance > 0:
                alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
                alphas.append(alpha)
        
        # Get percentile confidence interval
        ci_lower = np.percentile(alphas, 2.5)
        ci_upper = np.percentile(alphas, 97.5)
        
        return ci_lower, ci_upper
    
    def _get_scores(self, subjects: List[str], variable: str, time_point: str) -> Dict[str, float]:
        """Get scores from database"""
        # This would interface with your actual database
        # Placeholder implementation
        scores = {}
        for subject in subjects:
            # In real implementation, query database
            scores[subject] = np.random.uniform(0, 1)
        return scores
    
    def _pair_scores(self, test_scores: Dict[str, float], 
                    retest_scores: Dict[str, float]) -> pd.DataFrame:
        """Create paired dataset from test and retest scores"""
        paired = []
        for subject_id in test_scores:
            if subject_id in retest_scores:
                paired.append({
                    'subject': subject_id,
                    'test': test_scores[subject_id],
                    'retest': retest_scores[subject_id]
                })
        return pd.DataFrame(paired)
    
    def _fisher_z_test(self, r1: float, r2: float, n: int) -> float:
        """Fisher z-test for comparing correlations"""
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        se = np.sqrt(2 / (n - 3))
        return (z1 - z2) / se
    
    def _r_squared_confidence_interval(self, r_squared: float, n: int) -> Tuple[float, float]:
        """Approximate confidence interval for R-squared"""
        # Simplified approach
        se = np.sqrt((1 - r_squared) / (n - 2))
        ci_lower = max(0, r_squared - 1.96 * se)
        ci_upper = min(1, r_squared + 1.96 * se)
        return ci_lower, ci_upper
    
    def _interpret_factor_structure(self, loadings: pd.DataFrame) -> str:
        """Interpret factor analysis results"""
        # Check if items load on expected factors
        high_loadings = (loadings.abs() > 0.4).sum()
        cross_loadings = ((loadings.abs() > 0.3).sum(axis=1) > 1).sum()
        
        if high_loadings.sum() >= len(loadings) * 0.8:
            if cross_loadings < len(loadings) * 0.2:
                return "Clear factor structure with minimal cross-loadings"
            else:
                return "Factor structure present but with significant cross-loadings"
        else:
            return "Weak factor structure - consider model revision"


if __name__ == "__main__":
    # Example usage
    validator = GCTValidator()
    
    # Test internal consistency
    sample_data = pd.DataFrame(
        np.random.rand(100, 10),
        columns=[f'item_{i}' for i in range(10)]
    )
    
    result = validator.internal_consistency(sample_data)
    print(f"Internal Consistency: {result.value:.3f} - {result.interpretation}")