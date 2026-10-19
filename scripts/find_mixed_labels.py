from pathlib import Path

root = Path("The_final-4")
mixed_count = 0
polygon_count = 0

for label_dir in sorted(p for p in root.rglob("labels") if p.is_dir()):
    for file in sorted(label_dir.rglob("*.txt")):
        try:
            lines = file.read_text(encoding="utf-8-sig").splitlines()
        except Exception as exc:
            print("READ ERROR:", file, exc)
            continue

        rows = []
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                values = [float(v) for v in parts]
            except ValueError:
                print("INVALID NUMBER:", file, "line", line_no)
                continue

            if len(values) == 5:
                kind = "box"
            elif len(values) >= 7 and len(values) % 2 == 1:
                kind = "polygon"
                polygon_count += 1
            else:
                print("INVALID ROW LENGTH:", file, "line", line_no,
                      "values:", len(values))
                continue

            rows.append((line_no, kind, len(values)))

        kinds = {row[1] for row in rows}
        if len(kinds) > 1:
            mixed_count += 1
            print("\nMIXED FILE:", file)
            for line_no, kind, length in rows:
                print(f"  line {line_no}: {kind}, {length} values")

print("\nMixed files:", mixed_count)
print("Polygon rows found:", polygon_count)
