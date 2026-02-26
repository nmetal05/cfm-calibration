"""Test the fix — run ONE simulation and verify it works"""
import subprocess
import time
from pathlib import Path
from write_vtype import write_vtype_file
from parse_output import parse_edge_data, summary_statistics

BASE_DIR = Path(".")
sim_dir = BASE_DIR / "debug_sim"
sim_dir.mkdir(exist_ok=True)

# Default type_3 parameters
theta = [0.9, 0.1, 0.35, 1.5]
vtype_path = sim_dir / "vtype.xml"
write_vtype_file(theta, str(vtype_path))

# EdgeData output
edgedata_output = sim_dir / "edgedata.xml"
edgedata_add = sim_dir / "edgedata.add.xml"
with open(edgedata_add, "w") as f:
    f.write(f"""<additional>
    <edgeData id="sbi" freq="300" file="{edgedata_output.resolve()}" excludeEmpty="true"/>
</additional>""")

# Delete old output
if edgedata_output.exists():
    edgedata_output.unlink()

print("Running SUMO...")
t_start = time.time()

result = subprocess.run(
    [
        "sumo",
        "-c", str((BASE_DIR / "sbi_peak.sumocfg").resolve()),
        "--additional-files", f"{vtype_path.resolve()},{edgedata_add.resolve()}",
        "--seed", "42",
    ],
    capture_output=True,
    text=True,
    timeout=300,
    cwd=str(BASE_DIR.resolve())
)

elapsed = time.time() - t_start

print(f"\nReturn code: {result.returncode}")
print(f"Time: {elapsed:.1f}s")

if result.stderr:
    # Show only first 500 chars of stderr
    print(f"\nStderr (first 500 chars):")
    print(result.stderr[:500])

if edgedata_output.exists():
    size = edgedata_output.stat().st_size
    print(f"\n✅ EdgeData file exists! Size: {size:,} bytes")
    
    if size > 1500:
        data = parse_edge_data(str(edgedata_output))
        print(f"   Time bins: {len(data)}")
        if len(data) > 0:
            first_bin = list(data.keys())[0]
            print(f"   First time bin: {first_bin}s ({first_bin/3600:.1f}h)")
            print(f"   Edges in first bin: {len(data[first_bin])}")
            
            # Compute summary stats
            x = summary_statistics(data)
            if x is not None:
                print(f"\n   ✅ Summary statistics computed!")
                print(f"   Length: {len(x)}")
                print(f"   First 10 values: {x[:10]}")
                print(f"   Mean of all: {x.mean():.3f}")
                print(f"\n   🎉 EVERYTHING WORKS! Ready for benchmark!")
            else:
                print("   ❌ Summary stats returned None")
    else:
        print("   ❌ File too small — probably empty")
        print(open(edgedata_output).read()[:500])
else:
    print("\n❌ No edgedata file produced")
    print("Stderr:", result.stderr[:1000])