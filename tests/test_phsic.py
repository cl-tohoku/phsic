from phsic import __version__
from phsic import app


def test_version():
    assert __version__ == '0.1.0'


def test_app():
    app
