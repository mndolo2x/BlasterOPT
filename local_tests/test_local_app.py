import tempfile
import unittest
from contextlib import chdir
from pathlib import Path

from fastapi.testclient import TestClient

from local_app import create_app
from src.diamond_preservation import DiamondObservation, save_diamond_observation
from src.diamond_preservation import db as diamond_db
from src.performance_feedback import ActualMeasurement, BlastRecord
from src.performance_feedback import db as performance_db


class LocalHistoricalRecordsTests(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        self.performance = self.root / "performance #1.db"
        self.diamonds = self.root / "diamond observations.db"
        self.client = TestClient(
            create_app(self.performance, self.diamonds),
            base_url="http://127.0.0.1",
        )
        self.addCleanup(self.client.close)

    def test_missing_databases_do_not_create_files(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("Database missing"), 2)
        self.assertNotIn("Fictional records", response.text)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_demo_displays_fictional_records_without_using_databases(self):
        self.performance.write_bytes(b"preserve existing file")
        with TestClient(
            create_app(self.performance, self.diamonds, demo=True),
            base_url="http://127.0.0.1",
        ) as demo_client:
            response = demo_client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Demo mode · Fictional records", response.text)
        self.assertIn("DEMO-001", response.text)
        self.assertIn("DEMO-LOT-02", response.text)
        self.assertNotIn("Database missing", response.text)
        self.assertNotIn("The database could not be read", response.text)
        self.assertEqual(self.performance.read_bytes(), b"preserve existing file")
        self.assertFalse(self.diamonds.exists())

    def test_reads_existing_records_and_escapes_html(self):
        performance_db.init_db(self.performance)
        performance_db.save_blast_record(
            BlastRecord(blast_id="BENCH-1", site_id="<b>Test site</b>", location_identifier="Bench 1"),
            self.performance,
        )
        performance_db.save_actual_measurement(
            ActualMeasurement(
                measurement_id="measurement-1", blast_id="BENCH-1", target_metric="d50_mm",
                measured_value=245.5, unit="mm",
            ),
            self.performance,
        )
        diamond_db.init_db(self.diamonds)
        save_diamond_observation(
            DiamondObservation(blast_id="BENCH-1", sample_lot_id="LOT-1", total_carats=7.25),
            self.diamonds,
        )

        original_files = {path: path.read_bytes() for path in (self.performance, self.diamonds)}
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        for value in ("BENCH-1", "245.5", "LOT-1", "7.25", "&lt;b&gt;Test site&lt;/b&gt;"):
            self.assertIn(value, response.text)
        self.assertNotIn("<b>Test site</b>", response.text)
        self.assertEqual({path: path.read_bytes() for path in original_files}, original_files)
        self.assertEqual(self.client.post("/").status_code, 405)

    def test_empty_and_unreadable_databases_are_reported_separately(self):
        self.performance.write_text("invalid database", encoding="utf-8")
        diamond_db.init_db(self.diamonds)
        with self.assertLogs("local_app", level="ERROR"):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("The database could not be read", response.text)
        self.assertIn("No records have been saved", response.text)

    def test_non_local_host_header_is_rejected(self):
        self.assertEqual(self.client.get("/", headers={"Host": "external.example"}).status_code, 400)

    def test_existing_database_helpers_accept_bare_filenames(self):
        with chdir(self.root):
            performance_db.init_db("performance.db")
            diamond_db.init_db("diamonds.db")
        self.assertTrue((self.root / "performance.db").is_file())
        self.assertTrue((self.root / "diamonds.db").is_file())
        self.assertEqual(Path(performance_db.DB_PATH).parent, Path(diamond_db.DB_PATH).parent)
        self.assertTrue(Path(performance_db.DB_PATH).is_absolute())
