from modal import App, Image, Volume


# We first set out configuration variables for our script.
DATASET_DIR = "/data"
# VOLUME = "embedding-fineweb-edu"
# DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-10BT-chunked-120"
# DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-10BT-chunked-500"
# files = [f"data-{i:05d}-of-00099.parquet" for i in range(99)]



VOLUME = "datasets"
# DATASET_SAVE_CHUNKED = f"RedPajama-Data-V2-sample-10B-chunked-120"
# DATASET_SAVE_CHUNKED = f"RedPajama-Data-V2-sample-10B-chunked-500"
# files = [f"data-{i:05d}-of-00150.parquet" for i in range(150)]
# DATASET_SAVE_CHUNKED = f"pile-uncopyrighted-chunked-500"
# DATASET_SAVE_CHUNKED = f"pile-uncopyrighted-chunked-120"
# files = [f"data-{i:05d}-of-01987.parquet" for i in range(200)]
DATASET_SAVE_CHUNKED = f"wikipedia-en-chunked-120"
# DATASET_SAVE_CHUNKED = f"wikipedia-en-chunked-500"
files = [f"data-{i:05d}-of-00041.parquet" for i in range(41)]




# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"



@app.function(volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import pandas as pd
  
    # Load the dataset as a Hugging Face dataset
    # print(f"Loading dataset from {DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    df = pd.read_parquet(f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}")
    print("dataset", len(df))

    return {
        "file": file,
        "num_rows": len(df),
        "tokens": df["chunk_token_count"].sum(),
        "less2": df[df["chunk_token_count"] < 2].shape[0],
        "less10": df[df["chunk_token_count"] < 10].shape[0],
        "less50": df[df["chunk_token_count"] < 50].shape[0],
    }

@app.local_entrypoint()
def main():
    from tqdm import tqdm
    responses = []
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)
        responses.append(resp)

    total_rows = 0
    total_tokens = 0
    total_less2 = 0
    total_less10 = 0
    total_less50 = 0
    for resp in tqdm(responses):
        total_rows += resp['num_rows']
        total_tokens += resp['tokens']
        total_less2 += resp['less2']
        total_less10 += resp['less10']
        total_less50 += resp['less50']
    print(f"Total rows processed: {total_rows}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total less2: {total_less2}")
    print(f"Total less10: {total_less10}")
    print(f"Total less50: {total_less50}")


