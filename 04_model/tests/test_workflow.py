from __future__ import annotations

# Consolidated from test_workflow_cli.py
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workflow import cli


def test_cli_command_targets_exist():
    assert all((cli.ROOT / script).is_file() for script, _ in cli.COMMANDS.values())


def test_cli_passes_arguments_and_project_root_to_workflow(monkeypatch):
    call = {}

    def fake_run(arguments, *, cwd, check):
        call["arguments"] = arguments
        call["cwd"] = cwd
        call["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(["map", "--dry-run", "--output", "example.tif"])

    assert result == 7
    assert call["arguments"] == [
        cli.sys.executable,
        str(cli.ROOT / cli.COMMANDS["map"][0]),
        "--dry-run",
        "--output",
        "example.tif",
    ]
    assert call["cwd"] == cli.ROOT
    assert call["check"] is False


def test_cli_dispatches_analysis_subcommands(monkeypatch):
    calls = []

    def fake_run(arguments, *, cwd, check):
        calls.append(arguments)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    for command, subcommand in (
        ("historical-summary", "summary"),
        ("gcm-weights", "weights"),
        ("gcm-screening-plot", "plot"),
    ):
        assert cli.main([command]) == 0
        assert calls[-1][1:] == [str(cli.ROOT / cli.COMMANDS[command][0]), subcommand]


def test_cli_help_does_not_launch_a_workflow(monkeypatch, capsys):
    run = Mock(side_effect=AssertionError("help should not launch a workflow"))
    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli.main(["--help"]) == 0
    assert "agera5-download" in capsys.readouterr().out
    assert cli.main(["map", "--help"]) == 0
    assert "four-panel late-century" in capsys.readouterr().out
    run.assert_not_called()


def test_cli_rejects_unknown_workflow():
    with pytest.raises(SystemExit, match="2"):
        cli.main(["not-a-workflow"])
