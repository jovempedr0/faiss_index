import os
import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def test_dotenv_is_loaded_from_the_applications_working_directory(tmp_path):
    # Regression test: config.py called a bare load_dotenv(), whose find_dotenv() starts
    # searching from the directory of the calling module — config.py itself, wherever
    # the package lives — not from the working directory the docs promise. An
    # application's own .env was never found. Has to run a real script file in a
    # subprocess: under `python -c` (or a REPL) find_dotenv falls back to the working
    # directory anyway, which would hide the bug.
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / ".env").write_text("FAISS_INDEX_PATH_INDICES=/indices/from/app/dotenv\n", encoding="utf-8")
    (app_dir / "app.py").write_text(
        "from faiss_index import config\nprint(config.DEFAULT_PATH_INDICES)\n", encoding="utf-8"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("FAISS_INDEX_PATH_INDICES", "FAISS_INDEX_DOTENV_PATH")}
    env["PYTHONPATH"] = str(_SRC_DIR)

    result = subprocess.run([sys.executable, "app.py"], cwd=app_dir, capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/indices/from/app/dotenv"
