import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
st = json.loads((root / "outputs" / "task5_stats.json").read_text(encoding="utf-8"))
rm, n = st["categories"].get("retrieval_miss", 0), st["n_errors"]
p = root / "report.md"
t = p.read_text(encoding="utf-8")
t = t.replace("{{RM}}", str(rm)).replace("{{NERR}}", str(n)).replace("{{FRAC}}", f"{rm / max(1, n):.0%}")
p.write_text(t, encoding="utf-8")
print(f"Q4: {rm}/{n} = {rm / max(1, n):.2f}")
