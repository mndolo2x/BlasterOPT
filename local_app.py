"""Local browser workspace for BlasterOPT's historical records."""

import argparse
import logging
import sqlite3
from contextlib import closing
from html import escape
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock
from time import monotonic

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.diamond_preservation.ingestion import DAMAGE_TYPES, OBSERVATION_STAGES
from src.local_imports import COLUMNS, MAX_BYTES, preview_import, save_import
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

REGISTER_TITLES = {"measurements": "Post-blast measurements", "diamonds": "Diamond observations"}


def render_records(database: Path, query: str) -> str:
    if not database.is_file():
        return f"<p>Database missing. Import records below or copy an existing <code>{escape(database.name)}</code> to <code>data/processed/</code>.</p>"

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


def import_form(kind: str) -> str:
    title = REGISTER_TITLES[kind]
    if kind == "measurements":
        instructions = "Use <code>open_pit</code> or <code>underground</code> for <code>operation_type</code>."
    else:
        stages = ", ".join(OBSERVATION_STAGES)
        damage = ", ".join(DAMAGE_TYPES)
        instructions = f"Stages include {stages}. Damage types include {damage}."
    return f"""
        <form action="/import/{kind}/preview" method="post" enctype="multipart/form-data">
            <label for="{kind}-file">Upload {title.lower()}</label>
            <input id="{kind}-file" type="file" name="file" accept=".csv,.xlsx" required>
            <button type="submit">Preview import</button>
        </form>
        <p><a href="/import/{kind}/template">Download CSV template</a></p>
        <details><summary>File format</summary>
            <p>Keep the template column names. Supply a unique source ID for each row.
            Use ISO dates such as <code>2026-11-12</code>. Excel imports use the first worksheet.
            The limit is 1,000 rows and 2 MiB per file.</p>
            <p>{instructions}</p>
        </details>
    """


def require_local_post(request: Request) -> None:
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(403, "Submit this form from the local dashboard.")


def create_app(
    performance_db: Path = PERFORMANCE_DB_PATH,
    diamond_db: Path = DIAMOND_DB_PATH,
    *,
    demo: bool = False,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    databases = {"measurements": performance_db, "diamonds": diamond_db}
    pending = {}
    pending_lock = Lock()

    @app.get("/", response_class=HTMLResponse)
    def historical_records() -> HTMLResponse:
        if demo:
            description = '<p class="demo">Demo mode · Fictional records</p>'
            measurements = render_table(*DEMO_MEASUREMENTS)
            observations = render_table(*DEMO_OBSERVATIONS)
        else:
            description = "<p>Historical records · Latest 100 entries per register</p>"
            measurements = render_records(performance_db, MEASUREMENTS) + import_form("measurements")
            observations = render_records(diamond_db, OBSERVATIONS) + import_form("diamonds")
        return render_page(f"""
            {description}
            <section><h2>Post-blast measurements</h2>{measurements}</section>
            <section><h2>Diamond observations</h2>{observations}</section>
        """)

    if not demo:
        @app.get("/import/{kind}/template")
        def template(kind: str) -> Response:
            if kind not in COLUMNS:
                raise HTTPException(404)
            return Response(
                ",".join(COLUMNS[kind]) + "\r\n",
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{kind}-template.csv"'},
            )

        @app.post("/import/{kind}/preview")
        async def preview(kind: str, request: Request) -> HTMLResponse:
            if kind not in COLUMNS:
                raise HTTPException(404)
            require_local_post(request)
            length = request.headers.get("content-length", "")
            if not length.isdecimal() or int(length) > MAX_BYTES + 65536:
                return import_result("Choose a CSV or .xlsx file up to 2 MiB.", status=413)
            async with request.form(max_files=1, max_fields=1, max_part_size=MAX_BYTES) as form:
                upload = form.get("file")
                if not isinstance(upload, UploadFile) or not upload.filename:
                    return import_result("Choose a CSV or .xlsx file.", status=422)
                filename = upload.filename.replace("\\", "/").rsplit("/", 1)[-1]
                data = await upload.read(MAX_BYTES + 1)
            records, errors = await run_in_threadpool(preview_import, kind, filename, data)
            if errors:
                shown = "".join(f"<li>{escape(error)}</li>" for error in errors[:25])
                return render_page(
                    f"<h2>Fix {len(errors)} import issue(s)</h2><p>No rows were saved. Showing up to 25 issues.</p>"
                    f'<ul>{shown}</ul><a href="/">Return to upload</a>', 422
                )

            now = monotonic()
            token = token_urlsafe(32)
            with pending_lock:
                for key in list(pending):
                    if pending[key][0] <= now:
                        del pending[key]
                if len(pending) >= 16:
                    del pending[next(iter(pending))]
                pending[token] = (now + 900, kind, records, filename[:255])

            headers = COLUMNS[kind]
            table = render_table(headers, [[row[key] for key in headers] for row in records[:10]])
            return render_page(f"""
                <h2>Preview {REGISTER_TITLES[kind].lower()}</h2>
                <p>{len(records)} valid row(s) in {escape(filename)}. Showing up to 10.
                Existing IDs will be skipped when you save.</p>
                {table}
                <form action="/import/confirm" method="post">
                    <input type="hidden" name="token" value="{token}">
                    <button type="submit">Save import</button> <a href="/">Cancel</a>
                </form>
            """)

        @app.post("/import/confirm")
        async def confirm(request: Request) -> HTMLResponse:
            require_local_post(request)
            async with request.form(max_files=0, max_fields=1, max_part_size=1024) as form:
                token = form.get("token", "")
            with pending_lock:
                draft = pending.pop(token, None) if isinstance(token, str) else None
            if not draft or draft[0] <= monotonic():
                return import_result("This preview has expired or was already used. Upload the file again.", status=409)
            _, kind, records, filename = draft
            try:
                added, skipped = await run_in_threadpool(save_import, kind, databases[kind], records, filename)
            except ValueError as exc:
                return import_result(f"No rows were saved. {exc}", status=409)
            except (OSError, sqlite3.Error):
                logger.exception("Failed to save %s import", kind)
                return import_result("No rows were saved. Check the VS Code terminal for details.", status=500)
            return import_result(f"Saved {added} row(s). Skipped {skipped} existing ID(s).")

    return app


def import_result(message: str, status: int = 200) -> HTMLResponse:
    return render_page(f'<h2>Import result</h2><p>{escape(message)}</p><a href="/">Back to dashboard</a>', status)


def render_page(content: str, status: int = 200) -> HTMLResponse:
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
    a {{ color: #ffc966; }}
    form {{ display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; margin-top: 1rem; }}
    button {{ padding: .45rem .8rem; font: inherit; cursor: pointer; }}
</style>
<main>
    <h1>BlasterOPT</h1>
    {content}
</main>
</html>""",
        status_code=status,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        },
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the BlasterOPT historical records dashboard.")
    parser.add_argument("--demo", action="store_true", help="Display fictional records without opening databases.")
    args = parser.parse_args()
    uvicorn.run(create_app(demo=True) if args.demo else app, host="127.0.0.1", port=8000)
