# Interaction Terms Feature

## Overview

The interaction terms feature allows CRISPR-Millipede to model epistatic/combinatorial effects between variants by creating product features for variant pairs that frequently co-occur (co-edit) in the data.

## Usage

### Enabling Interaction Terms

To enable interaction terms, set two parameters in your `MillipedeDesignMatrixProcessingSpecification`:

```python
from crispr_millipede.modelling.models_inputs import MillipedeDesignMatrixProcessingSpecification

design_matrix_spec = MillipedeDesignMatrixProcessingSpecification(
    # ... other parameters ...
    include_interaction_terms=True,  # Enable interaction term generation
    interaction_term_coediting_frequency_threshold=0.1,  # Minimum 10% co-editing frequency
    # ... other parameters ...
)
```

### Parameters

- **`include_interaction_terms`** (bool, default=False): 
  - Set to `True` to enable automatic generation of interaction terms
  - When `False`, only individual variant effects are modeled

- **`interaction_term_coediting_frequency_threshold`** (float, default=0.1):
  - Minimum co-editing frequency (0.0 to 1.0) required to create an interaction term
  - Valid range: 0.0 to 1.0
  - Examples:
    - `0.1` = Create interaction terms for variant pairs that co-occur in ≥10% of alleles
    - `0.05` = More permissive (≥5% co-editing)
    - `0.2` = More stringent (≥20% co-editing)

### How It Works

1. **Co-editing Frequency Calculation**: 
   For each pair of variants, the function calculates:
   ```
   co-editing frequency = (# alleles with both variants) / (total # alleles)
   ```

2. **Interaction Term Creation**:
   - If co-editing frequency ≥ threshold, an interaction term is created
   - Interaction column name format: `variant1_x_variant2` (sorted lexicographically)
   - Interaction values = product of the two variant indicator values

3. **Example**:
   ```
   Variant A (10A>G): [1, 1, 0, 1, 1, 0]
   Variant B (20C>T): [1, 1, 1, 0, 1, 0]
   
   Co-edits in 3 out of 6 alleles → frequency = 0.5 (50%)
   
   If threshold ≤ 0.5, creates:
   10A>G_x_20C>T: [1, 1, 0, 0, 1, 0]  (element-wise product)
   ```

4. **Model Integration**:
   - Interaction terms are automatically included as features in the Millipede model
   - They are treated like regular variant features but capture combinatorial effects
   - The model will estimate independent effect sizes for interactions

### Recommendations

- **Start with default threshold (0.1)**: Balances sensitivity and computational cost
- **Lower threshold (0.05)** for exploratory analysis with many variants
- **Higher threshold (0.2-0.3)** when focusing on strong epistatic effects
- **Monitor output**: The function prints the number of interaction terms added

### Technical Notes

- Interaction terms are added after normalization but before model fitting
- All interaction terms satisfy the lexicographic naming convention for consistency
- Only pairwise interactions are currently supported (no 3-way or higher-order)
- Interaction terms work with all merge strategies and model types

## Example Output

```
Added 15 interaction terms (co-editing threshold: 0.1)
```

This indicates that 15 variant pairs met the co-editing threshold and were added as interaction features to the model.
