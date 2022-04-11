import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tutorial",
        action="store_true",
        default=False,
        help="run tutorial test cases",
    )
    parser.addoption("--force-update", action="store_true", help="reset cache for test")


def pytest_configure(config):
    config.addinivalue_line("markers", "tutorial: mark test as tutorial test cases")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--tutorial"):
        # --tutorials given in cli: do not skip tutorial dataset tests
        return

    skip_tutorial = pytest.mark.skip(reason="need --tutorial option to run")
    for item in items:
        if "tutorial" in item.keywords:
            item.add_marker(skip_tutorial)


@pytest.fixture
def force_update(request):
    return request.config.getoption("--force-update")
