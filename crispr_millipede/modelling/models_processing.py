"""
CRISPR-Millipede models processing module - backward compatibility shim.

This module re-exports all public classes and functions from the refactored
submodules (utils, input_data, models, encodings) to maintain backward compatibility
with existing code that imports from crispr_millipede.modelling.models_processing.

For new code, it is recommended to import directly from the submodules:
- from crispr_millipede.modelling.utils import decay_function, normalize_counts
- from crispr_millipede.modelling.input_data import MillipedeInputDataLoader
- from crispr_millipede.modelling.models import MillipedeModelExperimentalGroup
- from crispr_millipede.modelling.encodings import RawEncodingDataframesExperimentalGroup
"""

# Re-export utility functions
from .utils import (
    decay_function,
    decay_function_2d,
    normalize_counts,
    add_interaction_terms,
)

# Re-export input data classes
from .input_data import (
    MillipedeInputDataLoader,
    MillipedeInputDataExperimentalGroup,
)

# Re-export model classes
from .models import (
    MillipedeModelExperimentalGroup,
)

# Re-export encoding classes
from .encodings import (
    RawEncodingDataframesExperimentalGroup,
    EncodingEditingFrequenciesExperimentalGroup,
)

# Define public API
__all__ = [
    # Utilities
    "decay_function",
    "decay_function_2d",
    "normalize_counts",
    "add_interaction_terms",
    # Input Data
    "MillipedeInputDataLoader",
    "MillipedeInputDataExperimentalGroup",
    # Models
    "MillipedeModelExperimentalGroup",
    # Encodings
    "RawEncodingDataframesExperimentalGroup",
    "EncodingEditingFrequenciesExperimentalGroup",
]
