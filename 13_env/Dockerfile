FROM condaforge/miniforge3:24.11.3-0

WORKDIR /workspace

COPY 13_env/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && mamba clean -afy

COPY . /workspace

SHELL ["mamba", "run", "-n", "simple-cacao-wa", "/bin/bash", "-lc"]

RUN pip install -e .

CMD ["mamba", "run", "-n", "simple-cacao-wa", "python", "-m", "pytest"]
