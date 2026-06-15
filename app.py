"""HF Spaces entry point — downloads models then launches the dashboard."""
import sys
import os

# Ensure project root is importable (src/*, configs/, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import download_models
download_models.download_all()

from src.utils.database import init_db
init_db()

# Run the Streamlit frontend in the same process.
# Pass correct __file__ so sys.path inside frontend/app.py resolves properly.
exec(
    open("frontend/app.py", encoding="utf-8").read(),
    {"__file__": os.path.abspath("frontend/app.py"), "__name__": "__main__"},
)
