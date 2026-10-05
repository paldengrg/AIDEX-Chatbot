"""Test setup: runs before any app module is imported.

Tests use the free "mock" LLM provider, a throwaway ChromaDB folder and a
throwaway SQLite database, so they need no API key and never touch real data.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="aidx-test-")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["CHROMA_DIR"] = os.path.join(_tmp, "chroma")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db").replace("\\", "/")
os.environ["SECRET_KEY"] = "test-secret-key"
