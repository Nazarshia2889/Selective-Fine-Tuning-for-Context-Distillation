"""Print the run grid without Modal or torch: python scripts/sweep_dry_run.py [configs/sweep.yaml]"""
import sys
from collections import Counter

from sfcd.config import expand_lr_tune, expand_main, load_base, load_sweep

sw = load_sweep(sys.argv[1] if len(sys.argv) > 1 else "configs/sweep.yaml")
base = load_base(sw.base_config)
lt, mn = expand_lr_tune(sw, base), expand_main(sw, base)
print(f"lr_tune: {len(lt)}  main: {len(mn)}  total: {len(lt) + len(mn)}")
print("main by granularity:", dict(Counter(r.granularity for r in mn)))
print("main by selector:", dict(Counter(r.selector for r in mn)))
