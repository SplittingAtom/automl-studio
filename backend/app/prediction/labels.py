"""The user-facing name of the model's output quantity, in one place.

Sensitivity curves, insights, and any future view that titles the prediction
must agree on this wording — import it, don't restate it.
"""

from app.training.preprocessing import FeatureSpec


def prediction_output_label(spec: FeatureSpec, class_index: int | None = None) -> str:
    """"Predicted <target>" for values/labels, 'Chance of "<class>"' for a probability."""
    classes = spec.target.classes
    if classes is None or class_index is None:
        return f"Predicted {spec.target.name}"
    return f'Chance of "{classes[class_index]}"'
