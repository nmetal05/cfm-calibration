"""
fix_routes.py — Remove vType definitions from routes_peak.rou.xml
Run once: python fix_routes.py
"""

INPUT = "routes_peak.rou.xml"
OUTPUT = "routes_peak_novtype.rou.xml"

# We use string-based approach because the vType lines use 
# normc() in speedFactor which can cause XML parsing issues
with open(INPUT, "r", encoding="utf-8") as f:
    lines = f.readlines()

removed = 0
kept_lines = []
types_found = set()

for line in lines:
    if "<vType " in line:
        # Extract the id for logging
        import re
        match = re.search(r'id="([^"]+)"', line)
        if match:
            types_found.add(match.group(1))
        removed += 1
        # Skip this line (don't add to output)
    else:
        kept_lines.append(line)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.writelines(kept_lines)

print(f"Removed {removed} vType definitions: {types_found}")
print(f"Saved to: {OUTPUT}")
print(f"File size: {len(kept_lines)} lines")

# Count vehicle types actually used by vehicles
types_used = set()
for line in kept_lines:
    import re
    match = re.search(r'type="([^"]+)"', line)
    if match:
        types_used.add(match.group(1))

print(f"Vehicle types referenced by vehicles: {types_used}")