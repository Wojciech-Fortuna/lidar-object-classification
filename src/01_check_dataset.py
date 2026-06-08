from pathlib import Path
from plyfile import PlyData

DATA_PATH = Path(
    r"C:\Users\Wojciech Fortuna\IWIUM\Project\DALESObjects\train"
)

ply_files = sorted(DATA_PATH.glob("*.ply"))
ply_files = [p for p in ply_files if not p.name.startswith("._")]

print(f"Number of PLY files: {len(ply_files)}")

sample_file = ply_files[0]
print(f"Sample file: {sample_file.name}")

ply = PlyData.read(sample_file)

print("\nPLY elements:")
for element in ply.elements:
    print(f"Element name: {element.name}")
    print(f"Fields: {element.data.dtype.names}")
    print(f"Number of records: {len(element.data)}")