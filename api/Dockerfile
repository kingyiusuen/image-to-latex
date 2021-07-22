# Base image
FROM python:3.8-slim

# Install dependencies
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && make install \
    && apt-get purge -y --auto-remove build-essential

# Copy only the relevant directories
COPY image_to_latex image_to_latex
COPY api api
COPY artifacts artifacts

# Start app
ENTRYPOINT ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]