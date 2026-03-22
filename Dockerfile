FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace/criticism_bot

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/criticism_bot/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /workspace/criticism_bot

CMD ["/bin/bash"]
