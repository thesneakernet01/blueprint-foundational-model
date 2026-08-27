# SPDX-License-Identifier: Apache-2.0
# Thin backend image for the UI demo: web layer only, no GPU/torch/cuDF.
# The engine boots in DEMO-FALLBACK mode (synthetic, clearly-labelled scores)
# when the heavy stack / checkpoint / artifacts are absent — exactly this image.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-demo.txt ./
# numpy is used by the engine's fallback scoring (pulled by joblib too; explicit
# here so it's guaranteed present).
RUN pip install --no-cache-dir -r requirements-demo.txt numpy

COPY app.py ./
COPY tfm_demo ./tfm_demo

# Not in CML, so app binds $HOST:$PORT (see tfm_demo/config.server_host_port).
ENV HOST=0.0.0.0 PORT=7100
EXPOSE 7100

CMD ["python", "app.py"]
