from pathlib import Path

from geoluck.config import get_paths, project_root


def test_project_root_from_src_file() -> None:
    root = project_root(Path(__file__))
    assert (root / "pyproject.toml").exists()


def test_get_paths_matches_repo_layout() -> None:
    paths = get_paths(Path(__file__))
    assert paths.data_raw.name == "data_raw"
    assert paths.data_web == paths.data_final / "web"
    assert paths.do.name == "do"
    assert paths.web.name == "web"
    assert paths.web_public == paths.web / "public"
