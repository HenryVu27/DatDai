"""Tests for db.py conversation state functions."""
import json
import pytest
from app import db


@pytest.fixture(autouse=True)
def fresh_session(tmp_path, monkeypatch):
    """Use a temp SQLite DB for each test."""
    import app.db as db_module
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_use_pg", False)
    monkeypatch.setattr("app.config.DB_PATH", db_path)
    # Reset module-level sqlite state
    yield
    try:
        import os
        os.unlink(db_path)
    except FileNotFoundError:
        pass


class TestConvState:
    def test_get_returns_none_when_no_state(self):
        db.create_session("sess1")
        result = db.get_conv_state("sess1")
        assert result is None

    def test_upsert_then_get(self):
        db.create_session("sess1")
        state = {"current_topic": "bồi thường", "last_list": ["1. A", "2. B", "3. C"]}
        db.upsert_conv_state("sess1", state, turn=1)
        result = db.get_conv_state("sess1")
        assert result is not None
        assert result["current_topic"] == "bồi thường"
        assert result["last_list"] == ["1. A", "2. B", "3. C"]

    def test_upsert_overwrites_previous_state(self):
        db.create_session("sess1")
        db.upsert_conv_state("sess1", {"topic": "first"}, turn=1)
        db.upsert_conv_state("sess1", {"topic": "second", "extra": "data"}, turn=2)
        result = db.get_conv_state("sess1")
        assert result["topic"] == "second"
        assert result["extra"] == "data"

    def test_state_is_free_form_json(self):
        db.create_session("sess1")
        # Any JSON shape is valid — no fixed schema
        state = {
            "chu_de": "thủ tục cấp sổ đỏ",
            "buoc_dang_ban": 3,
            "co_quan": "UBND cấp huyện",
            "ho_so": ["CMND", "đơn xin cấp"],
        }
        db.upsert_conv_state("sess1", state, turn=1)
        result = db.get_conv_state("sess1")
        assert result["buoc_dang_ban"] == 3
        assert result["ho_so"] == ["CMND", "đơn xin cấp"]

    def test_delete_session_removes_state(self):
        db.create_session("sess1")
        db.upsert_conv_state("sess1", {"x": 1}, turn=1)
        db.delete_session("sess1")
        # Session deleted — recreate and check state is gone
        db.create_session("sess1")
        assert db.get_conv_state("sess1") is None
