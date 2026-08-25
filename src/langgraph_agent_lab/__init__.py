"""Day 08 LangGraph agent lab starter."""

from dotenv import load_dotenv

# Load .env exactly once per process, before any module reads os.getenv().
# Existing environment variables win (override=False), so CI secrets and shell
# injection keep priority over the local file.
load_dotenv(override=False)

__all__ = ["__version__"]
__version__ = "0.1.0"
