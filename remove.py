from modal import App, Image, Volume, Secret

DATASET_DIR="/embeddings"
VOLUME = "embeddings"
SAE = "64_32"

# SAMPLE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3/train"
# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10"
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10/combined"

SAMPLE_DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500/train"
DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}-top5"
SAVE_DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}-top5/combined"





# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "pandas", "datasets==2.16.1", "apache_beam==2.53.0"
)
app = App(image=image) 

@app.function(
    volumes={DATASET_DIR: volume}, 
    timeout=60000,
)
def remove_files_by_pattern(directory, pattern):
    """
    Remove all files in the specified directory that match the given pattern.
    
    Args:
        directory: Directory to search for files
        pattern: File pattern to match (e.g., "temp*" for files starting with "temp")
    """
    import os
    import glob
    
    # Get the full path pattern
    full_pattern = os.path.join(directory, pattern)
    
    # Find all files matching the pattern
    matching_files = glob.glob(full_pattern)
    
    # Count files to be removed
    file_count = len(matching_files)
    print(f"Found {file_count} files matching pattern '{pattern}' in {directory}")
    
    # Remove each file
    for file_path in matching_files:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Commit changes to the volume
    volume.commit()
    
    return f"Removed {file_count} files matching pattern '{pattern}'"

@app.local_entrypoint()
def main():
        
    directory = "/embeddings/wikipedia-en-chunked-500-nomic-embed-text-v1.5-64_32-top10"
    pattern = "temp*"
    print(f"Removing files matching '{pattern}' from '{directory}'")
    result = remove_files_by_pattern.remote(directory, pattern)
    print(result)
   

