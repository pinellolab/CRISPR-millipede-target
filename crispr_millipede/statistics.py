"""
Statistical analysis utilities for CRISPR-Millipede.

This module provides functions for advanced statistical inference on Bayesian 
posterior samples, including Bayes Factor calculations and hypothesis testing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def compute_bayes_factor_rope(
    posterior_samples: Union[np.ndarray, pd.Series],
    rope_low: float = -0.1,
    rope_high: float = 0.1
) -> Tuple[float, float, float]:
    """
    Compute Bayes Factor for testing whether an effect is outside the ROPE.
    
    The Bayes Factor BF₁₀ quantifies the evidence for H₁ (effect exists and is 
    practically meaningful) versus H₀ (effect is negligible). This implementation
    uses a Region of Practical Equivalence (ROPE) to define what constitutes a 
    "negligible" effect.
    
    BF₁₀ = P(effect outside ROPE | data) / P(effect inside ROPE | data)
    
    Interpretation (Jeffreys, 1961):
    - BF > 100: Extreme evidence for H₁
    - BF > 30: Very strong evidence for H₁
    - BF > 10: Strong evidence for H₁
    - BF > 3: Moderate evidence for H₁
    - BF > 1: Weak evidence for H₁
    - BF = 1: No evidence either way
    - BF < 1: Evidence for H₀ (negligible effect)
    - BF < 0.33: Weak evidence for H₀
    - BF < 0.1: Moderate evidence for H₀
    - BF < 0.01: Strong evidence for H₀
    
    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples of the parameter (e.g., β_{A,B} for interaction term)
    rope_low : float, default=-0.1
        Lower boundary of the Region of Practical Equivalence
    rope_high : float, default=0.1
        Upper boundary of the Region of Practical Equivalence
        
    Returns
    -------
    bf_10 : float
        Bayes Factor for H₁ (effect exists) vs H₀ (effect negligible)
        Returns np.inf if all posterior samples are outside ROPE
    p_rope : float
        Posterior probability that effect is inside ROPE (supports H₀)
    p_outside_rope : float
        Posterior probability that effect is outside ROPE (supports H₁)
        
    Examples
    --------
    >>> # Test whether an interaction effect is practically meaningful
    >>> beta_samples = sel.samples.beta[:, interaction_idx]
    >>> bf_10, p_rope, p_outside = compute_bayes_factor_rope(beta_samples)
    >>> print(f"BF₁₀ = {bf_10:.2f}")
    >>> if bf_10 > 10:
    ...     print("Strong evidence for meaningful interaction")
    >>> elif bf_10 < 0.1:
    ...     print("Strong evidence for negligible interaction")
    
    >>> # Custom ROPE for smaller effect size threshold
    >>> bf_10, _, _ = compute_bayes_factor_rope(beta_samples, 
    ...                                          rope_low=-0.05, 
    ...                                          rope_high=0.05)
    
    Notes
    -----
    Unlike traditional p-values, Bayes Factors can provide evidence FOR the 
    null hypothesis (BF < 1). This is particularly useful in genetic interaction
    studies where you want to conclude that an interaction is negligible.
    
    The ROPE approach is more realistic than point-null hypothesis testing 
    (H₀: β = 0 exactly) because it acknowledges that effects smaller than a 
    certain threshold are not biologically meaningful.
    
    References
    ----------
    Jeffreys, H. (1961). Theory of Probability (3rd ed.). Oxford University Press.
    
    Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian 
    estimation. Advances in Methods and Practices in Psychological Science, 1(2), 
    270-280.
    """
    samples = np.array(posterior_samples)
    
    # Probability inside ROPE (H₀: effect is negligible)
    p_rope = np.mean((samples >= rope_low) & (samples <= rope_high))
    
    # Probability outside ROPE (H₁: effect is meaningful)
    p_outside_rope = 1 - p_rope
    
    # Bayes Factor
    if p_rope > 0:
        bf_10 = p_outside_rope / p_rope
    else:
        bf_10 = np.inf  # Infinite evidence for H₁
    
    return bf_10, p_rope, p_outside_rope


def interpret_bayes_factor(bf_10: float) -> str:
    """
    Interpret Bayes Factor according to Jeffreys (1961) evidence scale.
    
    Parameters
    ----------
    bf_10 : float
        Bayes Factor for H₁ vs H₀
        
    Returns
    -------
    interpretation : str
        Verbal interpretation of the Bayes Factor
        
    Examples
    --------
    >>> bf_10, _, _ = compute_bayes_factor_rope(beta_samples)
    >>> print(interpret_bayes_factor(bf_10))
    'Very strong evidence for H₁'
    """
    if bf_10 > 100:
        return "Extreme evidence for H₁ (interaction exists)"
    elif bf_10 > 30:
        return "Very strong evidence for H₁"
    elif bf_10 > 10:
        return "Strong evidence for H₁"
    elif bf_10 > 3:
        return "Moderate evidence for H₁"
    elif bf_10 > 1:
        return "Weak evidence for H₁"
    elif bf_10 == 1:
        return "No evidence either way"
    elif bf_10 > 0.33:
        return "Weak evidence for H₀"
    elif bf_10 > 0.1:
        return "Moderate evidence for H₀"
    elif bf_10 > 0.01:
        return "Strong evidence for H₀"
    else:
        return "Very strong evidence for H₀ (negligible effect)"


def compute_conditional_effects(
    selector,
    interaction_feature: str,
    rope_low: float = -0.1,
    rope_high: float = 0.1
) -> pd.DataFrame:
    """
    Compute conditional effects for a genetic interaction term.
    
    For an interaction term A_x_B, computes:
    - β_A|B=1: Effect of A when B is present (conditional effect)
    - β_B|A=1: Effect of B when A is present (conditional effect)
    - β_A: Main effect of A alone
    - β_B: Main effect of B alone
    - β_{A,B}: Interaction term
    
    The conditional effect β_A|B=1 tells you: "What is the effect of editing 
    position A when position B is also edited?" This is computed as:
    β_A|B=1 = β_A + β_{A,B}
    
    The difference from the main effect (β_A|B=1 - β_A = β_{A,B}) tells you 
    whether the effect of A depends on the presence of B.
    
    Parameters
    ----------
    selector : NormalLikelihoodVariableSelector
        Millipede selector object with posterior samples retained 
        (must have been fit with streaming=False)
    interaction_feature : str
        Name of the interaction feature (e.g., "160A>G_x_163A>G")
    rope_low, rope_high : float
        ROPE boundaries for Bayes Factor calculation
        
    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns:
        - conditional_on: Which variant is held fixed
        - target_variant: Which variant's effect is being measured
        - beta_mean: Mean conditional effect
        - beta_std: Standard deviation
        - beta_95ci_low, beta_95ci_high: 95% credible interval
        - p_direction: Probability effect is in dominant direction
        - bayes_factor: BF₁₀ for meaningful effect
        - interpretation: Verbal interpretation of Bayes Factor
        
    Examples
    --------
    >>> # Compute conditional effects for interaction between 160A>G and 163A>G
    >>> results = compute_conditional_effects(
    ...     sel_with_posterior, 
    ...     "160A>G_x_163A>G"
    ... )
    >>> print(results)
    
    Notes
    -----
    This function requires that the selector was fit with streaming=False to 
    retain posterior samples. The symmetry property guarantees that:
    β_A|B=1 - β_A = β_B|A=1 - β_B = β_{A,B}
    
    See Also
    --------
    compute_bayes_factor_rope : Compute Bayes Factor for ROPE hypothesis test
    """
    # Parse interaction feature name
    parts = interaction_feature.split("_x_")
    if len(parts) != 2:
        raise ValueError(f"Invalid interaction feature format: {interaction_feature}")
    
    var_a, var_b = parts
    
    # Get feature indices
    feature_names = list(selector.pip.index)
    var_a_idx = feature_names.index(var_a)
    var_b_idx = feature_names.index(var_b)
    int_idx = feature_names.index(interaction_feature)
    
    # Extract posterior samples
    beta_samples = selector.samples.beta
    beta_A = beta_samples[:, var_a_idx]
    beta_B = beta_samples[:, var_b_idx]
    beta_AB = beta_samples[:, int_idx]
    
    # Compute conditional effects
    beta_A_given_B = beta_A + beta_AB  # Effect of A when B=1
    beta_B_given_A = beta_B + beta_AB  # Effect of B when A=1
    
    results = []
    
    for cond_samples, cond_on, target in [
        (beta_A_given_B, var_b, var_a),
        (beta_B_given_A, var_a, var_b)
    ]:
        # Summary statistics
        mean = np.mean(cond_samples)
        std = np.std(cond_samples)
        ci_low, ci_high = np.percentile(cond_samples, [2.5, 97.5])
        
        # Probability of direction
        p_positive = np.mean(cond_samples > 0)
        p_direction = max(p_positive, 1 - p_positive)
        
        # Bayes Factor
        bf_10, p_rope, p_outside = compute_bayes_factor_rope(
            cond_samples, rope_low, rope_high
        )
        
        results.append({
            'conditional_on': cond_on,
            'target_variant': target,
            'beta_mean': mean,
            'beta_std': std,
            'beta_95ci_low': ci_low,
            'beta_95ci_high': ci_high,
            'p_direction': p_direction,
            'p_rope': p_rope,
            'p_outside_rope': p_outside,
            'bayes_factor': bf_10,
            'interpretation': interpret_bayes_factor(bf_10)
        })
    
    return pd.DataFrame(results)
