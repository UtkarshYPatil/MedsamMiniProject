# The web client: static files on nginx. No Python, no model, no backend
# import -- it is a browser app that speaks HTTP to the API, which is what
# makes the frontend/backend split real rather than nominal.
#
# About 50 MB against ~800 MB for the Streamlit image, because there is no
# runtime here at all.
FROM nginx:1.27-alpine

COPY frontend/web/ /usr/share/nginx/html/

# API_BASE is baked at start-up so the same image works against localhost,
# a tunnel or a deployed host without a rebuild.
COPY docker/web-entrypoint.sh /docker-entrypoint.d/40-api-base.sh
RUN chmod +x /docker-entrypoint.d/40-api-base.sh

EXPOSE 80
