FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV ALLOWED_HOSTS=majesticshon123-deepverify.hf.space,localhost,127.0.0.1
ENV DEBUG=False

EXPOSE 7860

CMD python download_weights.py && \
    python manage.py migrate --run-syncdb && \
    python manage.py collectstatic --noinput && \
    gunicorn core.wsgi --bind 0.0.0.0:7860 --timeout 300 --workers 1
