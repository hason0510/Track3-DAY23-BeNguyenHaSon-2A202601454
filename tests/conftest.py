"""Pytest bootstrap: make .env variables visible to the test process.

``tests/test_graph_smoke.py`` evaluates its skip markers (``os.getenv(...)``) at module
import time, which happens before ``langgraph_agent_lab`` is imported, so the package's
own ``load_dotenv()`` would be too late. conftest.py is imported first, so loading here
is what makes the live smoke tests actually run when a key is configured.
"""

from dotenv import load_dotenv

load_dotenv(override=False)
