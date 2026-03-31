"""
Remove files matching a glob pattern from a Modal volume.

Usage:
    modal run remove.py
"""
from modal import App, Image, Volume

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10")
app = App(image=image)


@app.function(volumes={DATASET_DIR: volume}, timeout=60000)
def remove_files_by_pattern(directory, pattern):
    import os
    import glob

    full_pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(full_pattern)
    print(f"Found {len(matching_files)} files matching '{pattern}' in {directory}")

    for file_path in matching_files:
        try:
            os.remove(file_path)
            print(f"  Removed: {file_path}")
        except Exception as e:
            print(f"  Error removing {file_path}: {e}")

    volume.commit()
    return f"Removed {len(matching_files)} files"


@app.local_entrypoint()
def main():
    # Adjust these for your needs
    directory = "/embeddings/wikipedia-en-chunked-500-nomic-embed-text-v1.5-64_32-top5"
    pattern = "temp*"
    print(f"Removing '{pattern}' from '{directory}'")
    result = remove_files_by_pattern.remote(directory, pattern)
    print(result)
