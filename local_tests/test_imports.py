import csv
import re
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook

from local_app import create_app
from src.local_imports import DIAMOND_COLUMNS, MEASUREMENT_COLUMNS
from src.performance_feedback.db import get_actual_measurements_for_blast


ORIGIN = {"Origin": "http://127.0.0.1"}
MEASUREMENT = (
    "M-01", "B-01", "Example mine", "open_pit", "Bench A",
    "2026-11-10", "2026-11-12", "d50_mm", 245.5, "mm",
)
OBSERVATION = (
    "O-01", "B-01", "LOT-12", "2026-11-12", "recovery",
    "Large", 3, 4.75, "none", "Team A",
)


def make_csv(columns, *rows):
    buffer = StringIO()
    csv.writer(buffer).writerows([columns, *rows])
    return buffer.getvalue().encode("utf-8-sig")


class LocalImportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.performance = Path(temp.name) / "performance.db"
        self.diamonds = Path(temp.name) / "diamonds.db"
        self.client = TestClient(create_app(self.performance, self.diamonds), base_url=ORIGIN["Origin"])
        self.addCleanup(self.client.close)

    def preview(self, kind, data, filename="records.csv"):
        return self.client.post(
            f"/import/{kind}/preview",
            headers=ORIGIN,
            files={"file": (filename, data)},
        )

    def confirm(self, preview):
        self.assertEqual(preview.status_code, 200, preview.text)
        token = re.search(r'name="token" value="([^"]+)"', preview.text).group(1)
        return self.client.post("/import/confirm", headers=ORIGIN, data={"token": token})

    def test_csv_preview_save_and_duplicate_are_separate(self):
        data = make_csv(MEASUREMENT_COLUMNS, MEASUREMENT)
        preview = self.preview("measurements", data)
        self.assertIn("245.5", preview.text)
        self.assertFalse(self.performance.exists())

        saved = self.confirm(preview)
        self.assertIn("Saved 1 row(s). Skipped 0 existing ID(s).", saved.text)
        self.assertIn("Example mine", self.client.get("/").text)
        self.assertEqual(self.confirm(preview).status_code, 409)

        modified = (*MEASUREMENT[:8], 300, "mm")
        retry = self.confirm(self.preview("measurements", make_csv(MEASUREMENT_COLUMNS, modified)))
        self.assertIn("Saved 0 row(s). Skipped 1 existing ID(s).", retry.text)
        actuals = get_actual_measurements_for_blast("B-01", db_path=str(self.performance))
        self.assertEqual([record.measured_value for record in actuals], [245.5])

    def test_xlsx_preview_and_save(self):
        workbook = Workbook()
        workbook.active.append(DIAMOND_COLUMNS)
        workbook.active.append(OBSERVATION)
        workbook.create_sheet("Ignored").append(["Only the first sheet is imported"])
        with BytesIO() as file:
            workbook.save(file)
            preview = self.preview("diamonds", file.getvalue(), "records.XLSX")
        self.assertFalse(self.diamonds.exists())
        self.assertIn("Saved 1 row(s).", self.confirm(preview).text)
        self.assertIn("LOT-12", self.client.get("/").text)

    def test_invalid_rows_block_the_entire_preview(self):
        bad_number = (*MEASUREMENT[:8], "NaN", "mm")
        invalid = self.preview("measurements", make_csv(MEASUREMENT_COLUMNS, MEASUREMENT, bad_number))
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("Row 3:", invalid.text)
        self.assertNotIn("Save import", invalid.text)
        self.assertFalse(self.performance.exists())

    def test_transaction_rolls_back_on_existing_blast_conflict(self):
        self.confirm(self.preview("measurements", make_csv(MEASUREMENT_COLUMNS, MEASUREMENT)))
        new = ("M-02", "B-02", *MEASUREMENT[2:])
        conflict = ("M-03", "B-01", "Different mine", *MEASUREMENT[3:])
        failed = self.confirm(self.preview("measurements", make_csv(MEASUREMENT_COLUMNS, new, conflict)))
        self.assertEqual(failed.status_code, 409)
        self.assertIn("No rows were saved.", failed.text)
        self.assertNotIn("B-02", self.client.get("/").text)

    def test_demo_and_external_origins_cannot_import(self):
        with TestClient(create_app(demo=True), base_url=ORIGIN["Origin"]) as demo:
            self.assertEqual(demo.post("/import/diamonds/preview", headers=ORIGIN).status_code, 404)
            self.assertNotIn("Preview import", demo.get("/").text)
        self.assertEqual(self.client.post("/import/confirm", headers={"Origin": "https://example.org"}).status_code, 403)
        template = self.client.get("/import/diamonds/template")
        self.assertEqual(template.text.strip(), ",".join(DIAMOND_COLUMNS))

    def test_rejects_corrupt_workbook_and_formula_cells(self):
        corrupt = self.preview("diamonds", b"not a workbook", "invalid.xlsx")
        self.assertEqual(corrupt.status_code, 422)
        workbook = Workbook()
        workbook.active.append(DIAMOND_COLUMNS)
        workbook.active.append((*OBSERVATION[:-1], "=1+1"))
        with BytesIO() as file:
            workbook.save(file)
            formula = self.preview("diamonds", file.getvalue(), "formula.xlsx")
        self.assertEqual(formula.status_code, 422)
        self.assertIn("Replace formulas", formula.text)
        self.assertFalse(self.diamonds.exists())
