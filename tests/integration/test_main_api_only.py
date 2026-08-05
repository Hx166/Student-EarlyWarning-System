"""tests/integration/test_main_api_only.py \u2014\u2014 \u4e3b\u5165\u53e3 api-only \u6a21\u5f0f\u96c6\u6210\u6d4b\u8bd5\u3002"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from src.database import DatabaseService
from src.api.app import create_app


class MainApiOnlyTest(unittest.TestCase):
    """\u5224\u65ad api-only \u6a21\u5f0f\u662f\u5426\u6ce8\u5165\u6570\u636e\u5e93\u4f9b\u67e5\u8be2\u63a5\u53e3\u4f7f\u7528\u3002"""


    def test_api_only_injects_database(self):
        """main api-only \u5206\u652f\u5e94\u5c06\u6570\u636e\u5e93\u6ce8\u5165 _run_flask\u3002"""
        import src.main as m

        captured = {}

        def fake_run_flask(host, port, debug, database=None):
            captured["host"] = host
            captured["port"] = port
            captured["db_is_none"] = database is None
            captured["db_conn"] = database is not None and database.conn is not None
            raise SystemExit

        orig = m._run_flask
        m._run_flask = fake_run_flask
        old_argv = sys.argv
        sys.argv = ["main.py", "--api-only"]
        try:
            with self.assertRaises(SystemExit):
                m.main()
        finally:
            m._run_flask = orig
            sys.argv = old_argv

        self.assertFalse(captured["db_is_none"])
        self.assertTrue(captured["db_conn"])

    def test_create_app_injected_db_serves_api(self):
        """\u6ce8\u5165\u6570\u636e\u5e93\u540e\u67e5\u8be2\u63a5\u53e3\u8fd4\u56de\u771f\u5b9e\u6570\u636e\u3002"""
        db = DatabaseService(db_path=":memory:").connect()
        try:
            sid = db.upsert_student("S9001", "\u6d4b\u8bd5")
            seid = db.create_session("\u8bfe\u7a0b", "\u8001\u5e08", "A101")
            db.insert_state_record(sid, seid, 0.3, 0.2, 5.0, 10.0, 2.0, 0.1, 82.0, 18.0)
            db.insert_warning(sid, seid, "high", "fatigue", "high_fatigue_score", 30.0)

            app = create_app(database=db)
            app.config["TESTING"] = True
            client = app.test_client()

            state = client.get("/api/state?limit=10").get_json()
            self.assertEqual(state["code"], 0)
            self.assertEqual(len(state["data"]), 1)

            warnings = client.get("/api/warnings").get_json()
            self.assertEqual(warnings["code"], 0)
            self.assertEqual(len(warnings["data"]), 1)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
