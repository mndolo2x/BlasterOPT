"""Local, read-only browser view for BlasterOPT's historical records."""

import argparse
import logging
import sqlite3
from contextlib import closing
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.storage_paths import DIAMOND_DB_PATH, PERFORMANCE_DB_PATH

logger = logging.getLogger(__name__)

MEASUREMENTS = """
    SELECT a.blast_id AS "Blast ID", b.site_id AS "Site",
           a.measurement_timestamp AS "Recorded", a.target_metric AS "Metric",
           a.measured_value AS "Measured value", a.unit AS "Unit"
    FROM actual_measurements AS a
    LEFT JOIN blast_records AS b ON a.blast_id = b.blast_id
    ORDER BY a.measurement_timestamp DESC, a.measurement_id DESC
    LIMIT 100
"""

OBSERVATIONS = """
    SELECT blast_id AS "Blast ID", sample_lot_id AS "Sample lot",
           observation_date AS "Recorded", stage AS "Stage",
           stone_size_class AS "Size class", stone_count AS "Stones",
           total_carats AS "Carats", damage_type AS "Recorded damage"
    FROM diamond_observations
    ORDER BY observation_date DESC, observation_id DESC
    LIMIT 100
"""

DEMO_MEASUREMENTS = (
    ("Blast ID", "Site", "Recorded", "Metric", "Measured value", "Unit"),
    [
        ("DEMO-002", "Example mine", "2026-01-15", "Fragment size (D80)", 310, "mm"),
        ("DEMO-001", "Example mine", "2026-01-12", "Fragment size (D80)", 285, "mm"),
    ],
)

DEMO_OBSERVATIONS = (
    ("Blast ID", "Sample lot", "Recorded", "Stage", "Size class", "Stones", "Carats", "Recorded damage"),
    [
        ("DEMO-002", "DEMO-LOT-02", "2026-01-16", "Recovery", "Large", 4, 5.60, "Chipping"),
        ("DEMO-001", "DEMO-LOT-01", "2026-01-13", "Recovery", "Medium", 7, 4.25, "None observed"),
    ],
)


def render_records(database: Path, query: str) -> str:
    if not database.is_file():
        return f"<p>Database missing. Copy an existing <code>{escape(database.name)}</code> to <code>data/processed/</code>.</p>"

    try:
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            cursor = connection.execute(query)
            headers = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
    except sqlite3.Error:
        logger.exception("Could not read historical database %s", database)
        return "<p>The database could not be read. Check the VS Code terminal for details.</p>"

    if not rows:
        return "<p>No records have been saved in this database.</p>"

    return render_table(headers, rows)


def render_table(headers, rows) -> str:
    heading = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value)) if value is not None else '—'}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def create_app(
    performance_db: Path = PERFORMANCE_DB_PATH,
    diamond_db: Path = DIAMOND_DB_PATH,
    *,
    demo: bool = False,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.get("/", response_class=HTMLResponse)
    def historical_records() -> HTMLResponse:
        if demo:
            description = '<p class="demo">Demo mode · Fictional records</p>'
            measurements = render_table(*DEMO_MEASUREMENTS)
            observations = render_table(*DEMO_OBSERVATIONS)
        else:
            description = "<p>Historical records · Latest 100 entries per register</p>"
            measurements = render_records(performance_db, MEASUREMENTS)
            observations = render_records(diamond_db, OBSERVATIONS)
        return HTMLResponse(
            f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BlasterOPT | Historical records</title>
<style>
    :root {{ color-scheme: dark; font: 16px/1.5 Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #101722; color: #e6edf5; }}
    main {{ max-width: 76rem; margin: auto; padding: 2.5rem 1rem; }}
    h1, h2 {{ line-height: 1.2; }}
    h1 {{ color: #ffc966; }}
    .demo {{ color: #ffc966; font-weight: 600; }}
    section {{ margin-top: 2rem; padding: 1rem; border: 1px solid #394759; border-radius: .5rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; text-align: left; }}
    th, td {{ border-bottom: 1px solid #394759; padding: .6rem; }}
    th {{ background: #223149; }}
    code {{ overflow-wrap: anywhere; }}
</style>
<main>
    <h1>BlasterOPT</h1>
    {description}
    <section><h2>Post-blast measurements</h2>{measurements}</section>
    <section><h2>Diamond observations</h2>{observations}</section>
</main>
</html>""",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the BlasterOPT historical records dashboard.")
    parser.add_argument("--demo", action="store_true", help="Display fictional records without opening databases.")
    args = parser.parse_args()
    uvicorn.run(create_app(demo=True) if args.demo else app, host="127.0.0.1", port=8000)
