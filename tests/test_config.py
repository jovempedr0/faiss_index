import os
import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _run_app(app_dir, env_overrides=None):
    """Runs a real script file in `app_dir` that prints the default this library resolved."""
    (app_dir / "app.py").write_text(
        "from faiss_index import config\nprint(config.DEFAULT_PATH_INDICES)\n", encoding="utf-8"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("FAISS_INDEX_PATH_INDICES", "FAISS_INDEX_DOTENV_PATH")}
    env["PYTHONPATH"] = str(_SRC_DIR)
    env.update(env_overrides or {})

    result = subprocess.run([sys.executable, "app.py"], cwd=app_dir, capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_does_not_load_a_dotenv_found_in_the_working_directory(tmp_path):
    # Regression test: importing this package searched for a .env from the working
    # directory upwards and loaded it into os.environ — for the whole process, from a
    # file the application never pointed at. An import must not have that side effect.
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / ".env").write_text("FAISS_INDEX_PATH_INDICES=/indices/from/app/dotenv\n", encoding="utf-8")

    assert _run_app(app_dir) == "../faiss_index"  # the built-in default, not the .env's


def test_dotenv_path_env_var_still_loads_the_file_it_names(tmp_path):
    # The supported way in: the application says which .env, so nothing is guessed.
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "custom.env").write_text("FAISS_INDEX_PATH_INDICES=/indices/from/app/dotenv\n", encoding="utf-8")

    output = _run_app(app_dir, {"FAISS_INDEX_DOTENV_PATH": str(app_dir / "custom.env")})

    assert output == "/indices/from/app/dotenv"
