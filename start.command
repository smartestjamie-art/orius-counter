#!/bin/sh
# Double-click to open Orius Counter in your browser. It must be served from an address
# (not opened as a file) so the counter can run in the background.
# Others on the same network can use it too: http://<this Mac's IP>:8766 (no camera there).
cd "$(dirname "$0")"
PORT=${PORT:-8766}
( sleep 1; open "http://localhost:$PORT/" ) &
echo "Orius Counter is running at http://localhost:$PORT/  (close this window to stop it)"
exec python3 -m http.server "$PORT"
