"""
Run once: python filter_routes.py
Creates routes_peak.rou.xml with only 17:00-19:00 departures
"""
import xml.etree.ElementTree as ET

INPUT = "routes_24h.rou.xml"
OUTPUT = "routes_peak.rou.xml"
T_START = 61200   # 17:00
T_END = 68400     # 19:00

tree = ET.parse(INPUT)
root = tree.getroot()

to_remove = []
kept = 0

for elem in root:
    depart = elem.get("depart")
    if depart is not None:
        t = float(depart)
        if t < T_START or t > T_END:
            to_remove.append(elem)
        else:
            kept += 1

for elem in to_remove:
    root.remove(elem)

tree.write(OUTPUT, xml_declaration=True, encoding="UTF-8")
print(f"Total vehicles in file: {kept + len(to_remove)}")
print(f"Kept (17:00-19:00): {kept}")
print(f"Removed: {len(to_remove)}")
print(f"Saved to: {OUTPUT}")