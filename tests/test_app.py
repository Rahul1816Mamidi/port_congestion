from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run():
    return AppTest.from_file(APP, default_timeout=180).run()


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_and_scenarios_move_the_prediction():
    at = _run()
    assert not at.exception
    at.selectbox(key="preset").select("calm").run()
    assert not at.exception
    calm = int(_metrics(at)["Congestion probability"].rstrip("%"))
    at.selectbox(key="preset").select("storm-peak").run()
    assert not at.exception
    storm = int(_metrics(at)["Congestion probability"].rstrip("%"))
    assert calm < 30 < 70 < storm


def test_app_missing_data_and_loading_a_heldout_call():
    at = _run()
    at.checkbox(key="unk_queue").check().run()
    assert not at.exception
    assert any("Missing inputs" in i.value for i in at.info)
    at.button(key="load_call").click().run()
    assert not at.exception
    assert any("Actual outcome" in c.value for c in at.caption)
