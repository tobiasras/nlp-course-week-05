"""Load the hyphenated UI module so uvicorn can import main:app."""

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("information-retrieval-ui.py")
_spec = importlib.util.spec_from_file_location("information_retrieval_ui", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
