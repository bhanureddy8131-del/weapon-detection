from pathlib import Path
from collections import Counter

root = Path("The_final-4")
label_dirs = [p for p in root.rglob("labels") if p.is_dir()]

if not label_dirs:
    print("ERROR: No labels directories found under", root.resolve())
    raise SystemExit(1)

for label_dir in label_dirs:
    counts = Counter()
    problems = []
    files = list(label_dir.rglob("*.txt"))

    for file in files:
        try:
            lines = file.read_text(encoding="utf-8-sig").splitlines()
        except Exception as exc:
            problems.append((file, 0, f"read error: {exc}"))
            continue

        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue

            parts = line.split()
            counts[len(parts)] += 1

            try:
                values = [float(x) for x in parts]
                valid = (
                    len(values) == 5
                    or (len(values) >= 7 and len(values) % 2 == 1)
                )
                if not valid:
                    problems.append(
                        (file, line_no, f"unexpected {len(parts)} values")
                    )
                elif any(not 0 <= x <= 1 for x in values[1:]):
                    problems.append(
                        (file, line_no, "coordinates outside 0..1")
                    )
            except ValueError:
                problems.append((file, line_no, "non-numeric value"))

    print(f"\nLABEL DIRECTORY: {label_dir}")
    print(f"Files: {len(files)}")
    print("Row lengths (5 = detection; 7+ odd = possible polygon):", dict(counts))
    print(f"Problem rows: {len(problems)}")

    for file, line_no, reason in problems[:15]:
        print(f"  {file}:{line_no} - {reason}")
