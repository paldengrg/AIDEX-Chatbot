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
# High limits so the test suite itself is never rate limited (the limiter has
# its own tests).
os.environ["RATE_LIMIT_CHAT_PER_MINUTE"] = "10000"
os.environ["RATE_LIMIT_AUTH_PER_15MIN"] = "10000"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "test-admin-pass"
