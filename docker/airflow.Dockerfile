FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY docker/requirements-airflow.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/project
