"""
Hierarchical card navigation for BlastOpt Botswana.

Views:
  home      — 7 category cards
  category  — page cards inside a category
  page      — the actual Streamlit module content
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.navigation.categories import CATEGORIES

try:
    import streamlit as st
except ImportError:  # pragma: no cover - UI path requires Streamlit
    class _StreamlitStub:
        session_state: Dict[str, Any] = {}

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("streamlit is required to render navigation")

    st = _StreamlitStub()  # type: ignore[assignment]

# Filename stem -> existing app.py module key
FILE_TO_MODULE: Dict[str, str] = {
    "dashboard": "dashboard",
    "visualize": "visualize",
    "data_ingestion": "ingestion",
    "ml_model_manager": "ml_manager",
    "model_comparison": "comparison",
    "model_cards": "model_cards",
    "debswana_integrations": "integrations",
    "conversational_agent": "agent",
    "blast_pattern": "pattern",
    "blast_pattern_3d": "pattern_3d",
    "timing_design": "timing_design",
    "digital_twin": "digital_twin",
    "predictor": "predictor",
    "pinn": "pinn",
    "uncertainty": "ensemble_uq",
    "ga_optimizer": "optimizer",
    "pareto_optimizer": "pareto",
    "economic_dashboard": "economic",
    "similar_blasts": "recommender",
    "regulatory_compliance": "regulatory",
    "guardrail_log": "guardrail_log",
    "agent_audit_log": "audit_log",
    "drill_connectivity": "connectivity",
    "mwd_monitoring": "mwd",
    "detonator_integration": "detonator",
    "sync_status": "sync",
    "ollama_health": "system_health",
}

NAV_CSS = """
<style>
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
    text-align: left;
    white-space: normal;
    height: auto;
    padding: 0.55rem 0.85rem;
}
.bopt-card {
    background: #f7f9fc;
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 1rem 1.1rem 0.4rem 1.1rem;
    min-height: 168px;
    margin-bottom: 0.35rem;
}
.bopt-card h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.05rem;
    color: #1b2430;
}
.bopt-card p {
    margin: 0;
    color: #4a5568;
    font-size: 0.9rem;
    line-height: 1.35;
}
.bopt-card .bopt-meta {
    color: #718096;
    font-size: 0.78rem;
    margin-top: 0.55rem;
}
.bopt-crumb {
    color: #4a5568;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}
</style>
"""


def page_stem(file_path: str) -> str:
    """Return the filename stem without directory or extension."""
    return Path(file_path).stem


def module_for_page(file_path: str) -> str:
    """Map a page file path to its corresponding app.py module key."""
    stem = page_stem(file_path)
    if stem not in FILE_TO_MODULE:
        raise KeyError(f"No module mapping for navigation page '{stem}'")
    return FILE_TO_MODULE[stem]


def find_page(page_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find a page definition and its category ID given the page ID stem."""
    for cat_id, category in CATEGORIES.items():
        for page in category["pages"]:
            if page_stem(page["file"]) == page_id:
                return cat_id, page
    return None


def all_page_ids() -> List[str]:
    """Return a flat list of all page ID stems defined across categories."""
    return [page_stem(p["file"]) for cat in CATEGORIES.values() for p in cat["pages"]]


def init_nav_state() -> None:
    """Initialize session state defaults for navigation."""
    st.session_state.setdefault("nav_level", "home")
    st.session_state.setdefault("nav_category", None)
    st.session_state.setdefault("nav_page", None)


def go_home() -> None:
    """Navigate to the homepage (7 category cards view)."""
    st.session_state["nav_level"] = "home"
    st.session_state["nav_category"] = None
    st.session_state["nav_page"] = None


def go_category(category_id: str) -> None:
    """Navigate into a category view showing its page cards."""
    if category_id not in CATEGORIES:
        return
    st.session_state["nav_level"] = "category"
    st.session_state["nav_category"] = category_id
    st.session_state["nav_page"] = None


def go_page(category_id: str, page_id: str) -> None:
    """Navigate into an individual page."""
    found = find_page(page_id)
    if not found:
        return
    resolved_cat, _ = found
    st.session_state["nav_level"] = "page"
    st.session_state["nav_category"] = category_id if category_id in CATEGORIES else resolved_cat
    st.session_state["nav_page"] = page_id


def active_module() -> Optional[str]:
    """Return the active module key for rendering in app.py, or None if on home/category."""
    if st.session_state.get("nav_level") != "page":
        return None
    page_id = st.session_state.get("nav_page")
    if not page_id:
        return None
    found = find_page(page_id)
    if not found:
        return None
    _, page = found
    return module_for_page(page["file"])


def inject_nav_styles() -> None:
    """Inject custom CSS for cards and navigation buttons."""
    st.markdown(NAV_CSS, unsafe_allow_html=True)


def _render_breadcrumb() -> None:
    """Render the breadcrumb trail and quick back/home navigation buttons."""
    level = st.session_state.get("nav_level", "home")
    cat_id = st.session_state.get("nav_category")
    page_id = st.session_state.get("nav_page")

    crumbs = ["Home"]
    if cat_id and cat_id in CATEGORIES and level in ("category", "page"):
        crumbs.append(CATEGORIES[cat_id]["title"])
    if page_id and level == "page":
        found = find_page(page_id)
        if found:
            crumbs.append(found[1]["title"])

    st.markdown(
        f'<div class="bopt-crumb">{"  /  ".join(crumbs)}</div>',
        unsafe_allow_html=True,
    )

    if level != "home":
        cols = st.columns([1, 1, 6])
        with cols[0]:
            if st.button("🏠 Home", key="nav_btn_home", use_container_width=True):
                go_home()
                st.rerun()
        with cols[1]:
            if level == "page" and cat_id:
                if st.button("← Category", key="nav_btn_back_cat", use_container_width=True):
                    go_category(cat_id)
                    st.rerun()


def _card_html(icon: str, title: str, description: str, meta: str = "") -> str:
    """Generate HTML snippet for a card."""
    meta_html = f'<div class="bopt-meta">{meta}</div>' if meta else ""
    return (
        f'<div class="bopt-card"><h3>{icon} {title}</h3>'
        f"<p>{description}</p>{meta_html}</div>"
    )


def render_home() -> None:
    """Render the 7 category cards on the homepage."""
    st.subheader("Choose a workspace")
    st.caption("Open a category to see its tools, or jump from the sidebar if you already know the page.")

    items = list(CATEGORIES.items())
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for offset, (cat_id, cat) in enumerate(items[row_start : row_start + 3]):
            with cols[offset]:
                n_pages = len(cat["pages"])
                st.markdown(
                    _card_html(
                        cat["icon"],
                        cat["title"],
                        cat["description"],
                        meta=f"{n_pages} page{'s' if n_pages != 1 else ''}",
                    ),
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"home_cat_{cat_id}", use_container_width=True):
                    go_category(cat_id)
                    st.rerun()


def render_category(category_id: str) -> None:
    """Render page cards for all tools inside a selected category."""
    cat = CATEGORIES[category_id]
    st.subheader(f"{cat['icon']} {cat['title']}")
    st.caption(cat["description"])

    pages = cat["pages"]
    for row_start in range(0, len(pages), 2):
        cols = st.columns(2)
        for offset, page in enumerate(pages[row_start : row_start + 2]):
            with cols[offset]:
                pid = page_stem(page["file"])
                st.markdown(
                    _card_html(page["icon"], page["title"], page["description"]),
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"cat_{category_id}_{pid}", use_container_width=True):
                    go_page(category_id, pid)
                    st.rerun()


def render_hierarchical_sidebar() -> None:
    """Render category expanders and sub-pages in the Streamlit sidebar."""
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("🏠 Home", key="sidebar_home", use_container_width=True):
        go_home()
        st.rerun()

    current_cat = st.session_state.get("nav_category")
    current_page = st.session_state.get("nav_page")

    for cat_id, cat in CATEGORIES.items():
        expanded = current_cat == cat_id
        with st.sidebar.expander(f"{cat['icon']}  {cat['title']}", expanded=expanded):
            if st.button("View category", key=f"side_cat_{cat_id}", use_container_width=True):
                go_category(cat_id)
                st.rerun()
            for page in cat["pages"]:
                pid = page_stem(page["file"])
                label = f"{page['icon']} {page['title']}"
                if current_page == pid:
                    st.markdown(f"**→ {label}**")
                elif st.button(label, key=f"side_page_{pid}", use_container_width=True):
                    go_page(cat_id, pid)
                    st.rerun()


def render_nav_chrome(include_sidebar: bool = True) -> Optional[str]:
    """Render styles, optional sidebar, breadcrumbs, and home/category views.

    Returns the app.py module key when a page should render, otherwise None.
    """
    init_nav_state()
    inject_nav_styles()
    if include_sidebar:
        render_hierarchical_sidebar()
    _render_breadcrumb()

    level = st.session_state.get("nav_level", "home")
    if level == "home":
        render_home()
        return None
    if level == "category":
        cat_id = st.session_state.get("nav_category")
        if cat_id not in CATEGORIES:
            go_home()
            st.rerun()
            return None
        render_category(cat_id)
        return None
    return active_module()


__all__ = [
    "CATEGORIES",
    "FILE_TO_MODULE",
    "NAV_CSS",
    "active_module",
    "all_page_ids",
    "find_page",
    "go_category",
    "go_home",
    "go_page",
    "init_nav_state",
    "inject_nav_styles",
    "module_for_page",
    "page_stem",
    "render_category",
    "render_hierarchical_sidebar",
    "render_home",
    "render_nav_chrome",
]
