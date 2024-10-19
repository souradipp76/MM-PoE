# test_main.py

import pytest
from unittest.mock import patch
import runpy


def test_main_called():
    with patch("mm_poe.cli.main") as mock_main:
        # Simulate running __main__.py as the main module
        runpy.run_module("mm_poe.__main__", run_name="__main__")
        mock_main.assert_called_once()


def test_main_not_called_when_imported():
    with patch("mm_poe.cli.main") as mock_main:
        # Import __main__.py as a module; __name__ will not be '__main__'
        import mm_poe.__main__

        mock_main.assert_not_called()
