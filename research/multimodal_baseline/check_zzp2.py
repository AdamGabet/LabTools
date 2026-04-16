import zipfile
import os

z = zipfile.ZipFile(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/zzp/1001201093/00_00_visit/2021_10_03.zzp"
)

print("=" * 80)
print("ALL FILES IN ONE SLEEP STUDY")
print("=" * 80)

for name in z.namelist():
    info = z.getinfo(name)
    print(f"\n{name} ({info.file_size:,} bytes):")

    # Read first part of each file
    data = z.read(name)
    print(f"  Raw bytes (first 200): {data[:200]}")

    # Try to decode as text
    try:
        text = data.decode("utf-8", errors="replace")
        print(f"  First 500 chars as text:\n{text[:500]}")
    except:
        pass

    # Check if it's binary timeseries
    if name.endswith(".dat"):
        # Look for patterns
        print(f"  File size: {len(data)} bytes")
        # Check for repeated patterns (possible timeseries)
        if len(data) > 1000:
            print(f"  First 100 bytes (hex): {data[:100].hex()}")
