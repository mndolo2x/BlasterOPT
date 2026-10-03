"""
Reusable "Select Trained Model" UI component.

Pages call render_page_model_selector(page_key) to render a dropdown
of trained models that can power the page. The selector auto-detects
compatible models from session state — no import step required.
"""
from typing import Tuple, Any, Optional
import streamlit as st
try:
    from src.components.page_registry import PAGE_REGISTRY, get_compatible_models
except ImportError:
    from src.page_registry import PAGE_REGISTRY, get_compatible_models

from src.models import MODEL_REGISTRY


def render_page_model_selector(page_key: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Render a model selector for a specific page.

    Auto-detects which trained models can power the page based on the
    page's required_outputs in PAGE_REGISTRY.

    Args:
        page_key: key in PAGE_REGISTRY

    Returns:
        (model_instance, model_key) or (None, None) if no compatible model.
    """
    if page_key not in PAGE_REGISTRY:
        st.error(f"Unknown page key: {page_key}")
        return None, None

    page = PAGE_REGISTRY[page_key]
    page_name = page["display_name"]
    required = page["required_outputs"]

    trained = st.session_state.get("trained_models", {})

    # Case 1: No models trained at all
    if not trained:
        st.warning(
            f"⚠️ **{page_name}** requires a trained model.  \n"
            f"Go to **ML Model Manager**, generate data, and train a model. "
            f"Then return here."
        )
        return None, None

    # Case 2: Find compatible models
    compatible = get_compatible_models(page_key)

    if not compatible:
        st.error(
            f"⚠️ None of the trained models produce the outputs required by "
            f"**{page_name}**.  \n\n"
            f"**Required outputs:** `{', '.join(required)}`  \n"
            f"**Trained models:** `{', '.join(trained.keys())}`"
        )
        if page.get("requires_specific_model"):
            st.info(
                f"This page requires the **{page['requires_specific_model']}** model. "
                f"Train it in the ML Model Manager."
            )
        return None, None

    # Case 3: Show the dropdown
    selected_key = st.selectbox(
        "Select Trained Model",
        options=compatible,
        format_func=lambda k: MODEL_REGISTRY.get(k, {}).get("display_name", k),
        key=f"{page_key}_model_selector",
    )

    # Show metadata
    meta = st.session_state.get("trained_model_metadata", {}).get(selected_key, {})
    if meta:
        st.caption(
            f"Trained on {meta.get('rows', '?')} rows  •  "
            f"Fingerprint: `{meta.get('fingerprint', '?')}`  •  "
            f"At: {str(meta.get('trained_at', '?'))[:19]}"
        )

    model = trained[selected_key]
    return model, selected_key
