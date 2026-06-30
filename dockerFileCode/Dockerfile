FROM docker.io/python:3.11-slim

LABEL description="ctASLM2 Neuroglancer VizApp runtime"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    neuroglancer \
    tornado
