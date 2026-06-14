"""Analyze dialect datasets and build Iraqi phrase resources."""
import json
import os

proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dialect_dir = os.path.join(proj, "backend", "dialect")

# ── 1. IADD dataset ─────────────────────────────────────────────────────────
with open(os.path.join(dialect_dir, "IADD", "IADD.json"), encoding="utf-8") as f:
    iadd = json.load(f)

regions = set(d.get("Region") for d in iadd)
print("IADD total:", len(iadd))
print("Regions:", regions)

iraqi_rows = [d for d in iadd if d.get("Region", "").upper() in ("IRQ", "IQ", "IRAQ", "IRAQI")]
print("Iraqi rows:", len(iraqi_rows))

# Show sample
for row in iraqi_rows[:5]:
    print(" •", row["Sentence"][:100])
