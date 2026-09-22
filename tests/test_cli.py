from pathlib import Path

from src.acl.cli import build_console_summary
from src.acl.simulation import run


def test_console_summary_contains_final_state_and_all_log_paths(tmp_path):
    world = run(turns=1, seed=7)

    event_log = Path("logs/latest.jsonl")
    cognition_log = Path("logs/latest_cognition.jsonl")
    console_log = Path("logs/latest_console.log")

    summary = build_console_summary(
        world,
        event_log=event_log,
        cognition_log=cognition_log,
        console_log=console_log,
    )

    assert "Finished 1 turns." in summary
    assert "Ada" in summary
    assert "Log: logs/latest.jsonl" in summary
    assert "Cognition log: logs/latest_cognition.jsonl" in summary
    assert "Console log: logs/latest_console.log" in summary
