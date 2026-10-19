import os

from lab.config import load_env


def test_env_file_overrides_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HSL_TEST_KEY", "old")
    monkeypatch.setenv("HSL_KEEP", "shell")
    env = tmp_path / ".env"
    env.write_text('# comment\nHSL_TEST_KEY="new"\nexport HSL_OTHER=x\nHSL_KEEP=\n')
    loaded = load_env(env)
    assert os.environ["HSL_TEST_KEY"] == "new"       # .env wins over the shell
    assert os.environ["HSL_OTHER"] == "x"            # `export` prefix handled
    assert os.environ["HSL_KEEP"] == "shell"         # empty values don't clobber
    assert loaded == ["HSL_TEST_KEY", "HSL_OTHER"]
    monkeypatch.delenv("HSL_OTHER")
