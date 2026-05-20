FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "routellm[serve]"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6060

ENTRYPOINT ["/entrypoint.sh"]
