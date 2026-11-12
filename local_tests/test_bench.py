import unittest
from contextlib import chdir
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from local_app import create_app


class BenchPageTests(unittest.TestCase):
    def test_bench_page_and_offline_assets_work_from_another_directory(self):
        with TemporaryDirectory() as directory, chdir(directory):
            with TestClient(create_app(demo=True), base_url="http://127.0.0.1") as client:
                page = client.get("/digital-twin")
                self.assertEqual(page.status_code, 200)
                self.assertIn("SIMULATED DEMO", page.text)
                self.assertIn('href="/digital-twin"', client.get("/").text)
                for path in ("bench.css", "bench.js", "vendor/three.min.js", "vendor/OrbitControls.js"):
                    response = client.get("/bench-assets/" + path)
                    self.assertEqual(response.status_code, 200, path)
                    self.assertGreater(len(response.content), 100)
                self.assertNotIn("https://", page.text)
