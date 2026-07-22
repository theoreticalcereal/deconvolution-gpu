FROM docker.io/continuumio/miniconda3:24.11.1-0

COPY conda-linux-64.lock requirements.txt constraints.txt /tmp/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libice6 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxmu6 \
        libxrender1 \
        libxt6 \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN conda create --yes --name app --file /tmp/conda-linux-64.lock

RUN /opt/conda/envs/app/bin/python -m pip install \
        --no-cache-dir \
        --constraint /tmp/constraints.txt \
        --requirement /tmp/requirements.txt

RUN /opt/conda/envs/app/bin/python -m pip check \
    && /opt/conda/envs/app/bin/python -c "import numpy, numba, scipy, skimage, tifffile, zarr, pandas; assert numpy.__version__ == '1.26.4'; assert numba.__version__ == '0.59.1'; assert scipy.__version__ == '1.15.2'; assert tifffile.__version__ == '2025.5.10'; assert zarr.__version__ == '2.18.3'; print('scientific stack versions and binary imports verified')"

RUN conda clean --all --yes

ENV CONDA_PREFIX=/opt/conda/envs/app
ENV CONDA_DEFAULT_ENV=app
ENV PATH=/opt/conda/envs/app/bin:/opt/conda/bin:$PATH
ENV PYTHONNOUSERSITE=1

WORKDIR /workspace
ENTRYPOINT ["python"]
