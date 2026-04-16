import zipfile

z = zipfile.ZipFile(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/zzp/1001201093/00_00_visit/2021_10_03.zzp"
)
print("Files in zzp:")
for n in z.namelist()[:30]:
    print(f"  {n}")
print(f"\nTotal: {len(z.namelist())} files")
