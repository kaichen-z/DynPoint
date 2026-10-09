"""Per-scene wall-clock of each stage from work/log.txt timestamps. Run: python tests/timing.py data/nsff_*/work"""
import json
import os
import re
import sys
from datetime import datetime, timedelta


def ts(line):
    return datetime.strptime(line[:8], "%H:%M:%S")


for work in sys.argv[1:]:
    p = os.path.join(work, "log.txt")
    if not os.path.exists(p):
        continue
    lines = open(p).read().splitlines()
    stages = {}
    start = None
    for ln in lines:
        if "[cli] " in ln:
            cmd = ln.split("[cli] ")[1].split()[0]
            start = (cmd, ts(ln))
        for key, pat in (("preprocess", "[pre] saved"), ("train_flow", "dynamic-pixel"), ("train_render", "[nr] it"),
                         ("eval", "views:"), ("video", "[video] wrote")):
            if pat in ln and start and start[0] in (key, "all"):
                if key == "train_render" and not re.search(r"\[nr\] it (\d+)/\1 ", ln):
                    continue
                d = ts(ln) - start[1]
                if d < timedelta(0):
                    d += timedelta(days=1)
                stages[key] = round(d.total_seconds())
    m = os.path.join(work, "eval", "metrics.json")
    met = json.load(open(m))["mean"] if os.path.exists(m) else {}
    nr = [ln for ln in lines if re.search(r"\[nr\] it (\d+)/\1 ", ln)]
    iters = re.search(r"it (\d+)/", nr[-1]).group(1) if nr else "?"
    print(f"{work:40s} pre {stages.get('preprocess', '?'):>4}s  flow {stages.get('train_flow', '?'):>4}s  render({iters} it) {stages.get('train_render', '?'):>5}s  "
          f"eval {stages.get('eval', '?'):>4}s  video {stages.get('video', '?'):>3}s  | " +
          (f"PSNR {met['psnr']:.2f} SSIM {met['ssim']:.3f} LPIPS {met['lpips']:.3f}" if met else "no metrics"))
