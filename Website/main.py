"""Launch the Flask website; run this file from the ``Website`` directory."""

import os

# Keep Ultralytics' writable settings inside this project when the user's
# roaming profile is unavailable (for example, in a restricted Windows setup).
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), ".ultralytics"))

from Website import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
