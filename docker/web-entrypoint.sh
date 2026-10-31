#!/bin/sh
# Write the API base the browser should call. The page reads window.API_BASE
# and falls back to localhost:8000, so this only has to run when it differs.
#
# This is a browser app: the URL must be reachable from the PERSON'S machine,
# not from inside the Docker network. http://api:8000 resolves only between
# containers, which is the exact mistake the Streamlit version made before it
# was changed to fetch images server-side.
set -e
: "${API_BASE:=http://localhost:8000}"
cat > /usr/share/nginx/html/config.js <<JS
window.API_BASE = "${API_BASE}";
JS
sed -i 's#<script src="app.js"></script>#<script src="config.js"></script><script src="app.js"></script>#' \
    /usr/share/nginx/html/index.html 2>/dev/null || true
