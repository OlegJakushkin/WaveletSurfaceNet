#!/bin/bash
# Resilient waveshape trainer (region-composed recipe): AUTO-RESUME from the latest checkpoint on every
# (re)start, retry on crash, park when done.  Run under `docker run --restart unless-stopped` so a host
# reboot -> Docker Desktop autostart -> container restart -> this script resumes at the stored epoch.
cd /work
conda run -n fvdb pip install -q matplotlib bitsandbytes 2>&1 | tail -1
while [ ! -f assets/waveshape.done ]; do
    R=""
    [ -f assets/waveshape_latest.pt ] && R="--resume assets/waveshape_latest.pt"
    echo "=== launching train.py $R ==="
    conda run --no-capture-output -n fvdb python -u train.py --epochs 16 --batch 4 --out waveshape $R \
        && touch assets/waveshape.done
    sleep 10
done
echo "TRAINING DONE (marker present) -- parking"
sleep infinity
