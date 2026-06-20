FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app.py ./
COPY artifacts ./artifacts
COPY assets ./assets
COPY release ./release
RUN pip install --no-cache-dir .

ARG WCI_ARTIFACT_BUNDLE_URL=""
RUN if [ -f artifacts/model.joblib ]; then \
      worldcup-artifacts --verify; \
    elif [ -n "$WCI_ARTIFACT_BUNDLE_URL" ]; then \
      worldcup-artifacts --url "$WCI_ARTIFACT_BUNDLE_URL"; \
    else \
      echo "Missing model artifact. Set WCI_ARTIFACT_BUNDLE_URL to a release asset."; \
      exit 1; \
    fi

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
