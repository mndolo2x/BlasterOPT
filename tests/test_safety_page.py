"""
Unit tests for Safety & Environmental Streamlit page renderer.
"""

from unittest.mock import patch, MagicMock
from src.safety.safety_environmental import render_safety_environmental_page


@patch("streamlit.plotly_chart")
@patch("streamlit.download_button")
@patch("streamlit.button", return_value=True)
@patch("streamlit.json")
@patch("streamlit.metric")
@patch("streamlit.error")
@patch("streamlit.success")
@patch("streamlit.info")
@patch("streamlit.table")
@patch("streamlit.write")
@patch("streamlit.markdown")
@patch("streamlit.selectbox", side_effect=["ANFO", "ANFO"])
@patch("streamlit.number_input", side_effect=[1000.0, 3.0, 500.0, 1000.0, 10.0, 5000.0, 150.0, 3.0, 500.0, 800.0, 450.0, 1200.0])
@patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))])
@patch("streamlit.tabs")
@patch("streamlit.caption")
@patch("streamlit.title")
def test_render_safety_environmental_page(
    mock_title,
    mock_caption,
    mock_tabs,
    mock_columns,
    mock_number_input,
    mock_selectbox,
    mock_markdown,
    mock_write,
    mock_table,
    mock_info,
    mock_success,
    mock_error,
    mock_metric,
    mock_json,
    mock_button,
    mock_download,
    mock_plotly,
):
    tab1, tab2, tab3, tab4, tab5 = MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_tabs.return_value = [tab1, tab2, tab3, tab4, tab5]

    render_safety_environmental_page()

    assert mock_title.called
    assert mock_caption.called
    assert mock_tabs.called
