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
COPY app app
COPY artifacts artifacts

# Export ports
EXPOSE 8000

# Start app
ENTRYPOINT ["python", "api/app.py"]