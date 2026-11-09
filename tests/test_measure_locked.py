import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_measure_locked_runs_the_command_and_passes_its_exit_code(tmp_path: Path) -> None:
    # A private HOME gives the test its own lock file; the machine-wide one may be held
    # by a real measurement.
    import os

    env = {**os.environ, "HOME": str(tmp_path)}
    script = ROOT / "scripts" / "measure_locked.py"
    ok = subprocess.run([sys.executable, str(script), "--", sys.executable, "-c", "print('ran')"],
                        capture_output=True, text=True, cwd=ROOT, env=env)
    assert ok.returncode == 0 and "ran" in ok.stdout
    bad = subprocess.run([sys.executable, str(script), "--", sys.executable, "-c", "raise SystemExit(7)"],
                         capture_output=True, text=True, cwd=ROOT, env=env)
    assert bad.returncode == 7


def test_reproduce_check_reads_published_commands_from_a_ref() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_reproduce_commands as check

    assert check.published_commands("HEAD")


def test_committed_data_carries_no_home_directory_paths() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "scrub_home_paths.py"), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + "run: uv run python scripts/scrub_home_paths.py"


def test_scrub_replaces_a_measuring_worktree_with_the_checkout() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import scrub_home_paths as scrub

    assert scrub.scrub("/Users/x/orca/workspaces/repo/cov-F-agno/src/a.py") == "<checkout>/src/a.py"
    # A stderr tail can start partway through the home directory.
    assert scrub.scrub("ame/orca/workspaces/repo/wt/node_modules/b.js") == "<checkout>/node_modules/b.js"
    assert scrub.scrub("/home/x/.cache/uv/c.py") == "~/.cache/uv/c.py"
    # An agent's worktree inside a checkout, with or without the home prefix.
    assert scrub.scrub("/Users/x/mcp-mirror/.claude/worktrees/agent-a1/src/d.py") == "<checkout>/src/d.py"
    assert scrub.scrub("~/mcp-mirror/.claude/worktrees/agent-a1/e.json") == "<checkout>/e.json"
    # A path that ends at the worktree or at the directory holding it, and the IDE's bin.
    assert scrub.scrub('"workingDirectory": "~/orca/workspaces/repo/wt"') == '"workingDirectory": "<checkout>"'
    assert scrub.scrub("find ~/orca/workspaces/repo -name x") == "find <workspace> -name x"
    assert scrub.scrub("PATH=/usr/bin:/Applications/Orca.app/Contents/Resources/bin:~/b") == "PATH=/usr/bin:<ide>/bin:~/b"


def test_reproduce_check_splits_chained_commands() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_reproduce_commands as check

    assert check.split_command("a --x && b  &&  c") == ["a --x", "b", "c"]
    assert not any(" && " in part for part in check.published_commands())


def test_reproduce_check_removes_checkouts_left_by_dead_runs(tmp_path, monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import tempfile

    import check_reproduce_commands as check

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    dead, live = tmp_path / "dead", tmp_path / "live"
    for path, pid in ((dead, 2**22 + 12345), (live, __import__("os").getpid())):
        path.mkdir()
        (path / check.MARKER).write_text(str(pid))
    removed = check.remove_orphaned_checkouts()
    assert removed == [dead] and not dead.exists() and live.exists()


def test_reproduce_check_removes_read_only_trees(tmp_path) -> None:
    # Go writes its module cache read-only; the cleanup has to delete it anyway.
    sys.path.insert(0, str(ROOT / "scripts"))
    import os

    import check_reproduce_commands as check

    module = tmp_path / "checkout" / "cache" / "go-runners" / "modcache" / "example.com@v1"
    module.mkdir(parents=True)
    (module / "go.mod").write_text("module example.com\n")
    for path in (module / "go.mod", module, module.parent):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    check._remove(tmp_path / "checkout")
    assert not (tmp_path / "checkout").exists()


def test_a_harness_run_under_measure_locked_does_not_deadlock(tmp_path) -> None:
    import os

    env = {**os.environ, "HOME": str(tmp_path)}
    script = ROOT / "scripts" / "measure_locked.py"
    inner = (
        "import sys; sys.path.insert(0, 'scripts'); import measure_harness as m\n"
        "with m.measurement_lock(): print('inner ran')"
    )
    result = subprocess.run([sys.executable, str(script), "--", sys.executable, "-c", inner],
                            capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
    assert result.returncode == 0 and "inner ran" in result.stdout
