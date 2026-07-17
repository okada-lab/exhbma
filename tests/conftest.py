import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--tutorial",
        action="store_true",
        default=False,
        help="run tutorial test cases",
    )
    parser.addoption("--force-update", action="store_true", help="reset cache for test")


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "tutorial: mark test as tutorial test cases")


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    if config.getoption("--tutorial"):
        # --tutorials given in cli: do not skip tutorial dataset tests
        return

    skip_tutorial = pytest.mark.skip(reason="need --tutorial option to run")
    for item in items:
        if "tutorial" in item.keywords:
            item.add_marker(skip_tutorial)


@pytest.fixture
def force_update(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--force-update"))
