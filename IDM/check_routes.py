"""Check what vTypes are defined inside routes_peak.rou.xml"""

with open("routes_peak.rou.xml", "r") as f:
    lines = f.readlines()

print("=== First 80 lines of routes_peak.rou.xml ===")
for i, line in enumerate(lines[:80]):
    print(f"{i+1:4d}: {line.rstrip()}")

print("\n=== Lines containing 'vType' ===")
for i, line in enumerate(lines):
    if "vType" in line or "vtype" in line.lower():
        print(f"{i+1:4d}: {line.rstrip()}")

print("\n=== Lines containing 'type_' ===")
count = 0
for i, line in enumerate(lines):
    if 'type="type_' in line and count < 10:
        print(f"{i+1:4d}: {line.rstrip()}")
        count += 1
if count == 10:
    print("   ... (showing first 10 only)")