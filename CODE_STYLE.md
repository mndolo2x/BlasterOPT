# BlastOpt Botswana Code Style Guide

## Python Coding Standards
- **Python Version:** Python 3.10+ compatible code.
- **Formatting:** Clean PEP 8 compliance with 4-space indentations.
- **Type Annotations:** Use Python standard type hints (`typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`, `pandas.DataFrame`, `numpy.ndarray`) across public module functions.
- **Docstrings:** Use Google/NumPy standard docstrings explaining parameters, return types, and physical unit conventions (e.g. `m`, `mm`, `kg/m3`, `mm/s`, `dB`).

## Data Science & ML Conventions
- **Reproducibility:** Set fixed random seeds (`random_state=42` or `torch.manual_seed(42)`) for data generators, model initializations, and genetic algorithm optimization routines.
- **Graceful Fallbacks:** Handle missing optional dependencies (`torch`, `shap`, `fpdf2`, `kaleido`) cleanly with fallback physics equations or placeholder returns rather than crashing execution.
- **Model Metadata:** Maintain clear citation and metadata dicts in `MODEL_REGISTRY` for research models (author, mine dataset, model structure, evaluation metrics).

## Streamlit UI Components
- Use `st.cache_data` or `st.cache_resource` for expensive data loading, synthetic generator calls, and model loading.
- Standardize multi-page app architecture in `app.py` using sidebar navigation selectboxes.
- Use wide mode layout (`st.set_page_config(layout="wide")`) for data tables and Plotly visualizations.

## Testing Guidelines
- Use `pytest` for all unit test suites in `tests/`.
- Ensure tests cover both happy paths and edge cases (clip ranges, invalid inputs, fallback physics mode).
