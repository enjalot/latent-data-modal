from modal import App, Image, Volume

NUM_CPU=4
MAX_TOKENS = 500
# MAX_TOKENS = 120
OVERLAP = 0.1 # 10% overlap when chunking
BATCH_SIZE = 200 # number of rows to process per thread at once

# We first set out configuration variables for our script.
DATASET_DIR = "/data"

# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
# VOLUME = "embedding-fineweb-edu"
# DATASET_SAVE ="fineweb-edu-sample-10BT"
# DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-10BT-chunked-{MAX_TOKENS}"
# TEXT_KEY = "text"
# KEEP_KEYS = ["id", "url", "score", "dump"] 
# files = [f"data-{i:05d}-of-00099.arrow" for i in range(99)]

# VOLUME = "embedding-fineweb-edu"
# DATASET_SAVE ="fineweb-edu-sample-100BT"
# DATASET_SAVE_CHUNKED = f"fineweb-edu-sample-100BT-chunked-{MAX_TOKENS}"
# KEEP_KEYS = ["id", "url", "score", "dump"] 


# https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
# VOLUME = "datasets"
# DATASET_SAVE ="RedPajama-Data-V2-sample-10B"
# DATASET_SAVE_CHUNKED = f"RedPajama-Data-V2-sample-10B-chunked-{MAX_TOKENS}"
# TEXT_KEY = "raw_content"
# KEEP_KEYS = ["doc_id", "meta"]
# files = [f"data-{i:05d}-of-00150.arrow" for i in range(150)]

# https://huggingface.co/datasets/monology/pile-uncopyrighted
# VOLUME = "datasets"
# DATASET_SAVE ="pile-uncopyrighted"
# DATASET_SAVE_CHUNKED = f"pile-uncopyrighted-chunked-{MAX_TOKENS}"
# TEXT_KEY = "text"
# KEEP_KEYS = ["meta"]
# files = [f"data-{i:05d}-of-01987.arrow" for i in range(200)]

#https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.en
# VOLUME = "datasets"
# DATASET_SAVE ="wikipedia-en"
# DATASET_SAVE_CHUNKED = f"wikipedia-en-chunked-{MAX_TOKENS}"
# TEXT_KEY = "text"
# KEEP_KEYS = ["id", "url", "title"]
# files = [f"data-{i:05d}-of-00041.arrow" for i in range(41)]

VOLUME = "datasets"
DATASET_SAVE ="medrag-pubmed"
DATASET_SAVE_CHUNKED = f"medrag-pubmed-{MAX_TOKENS}"
TEXT_KEY = "content"
KEEP_KEYS = ["id", "title", "PMID"]
files = [f"data-{i:05d}-of-00138.arrow" for i in range(138)]




# MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

def chunk_row(row, tokenizer):
    # print("ROW", row)
    text = row[TEXT_KEY]
    chunks = []

    # TODO: don't save an empty chunk

    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    if token_count > MAX_TOKENS:
        overlap = int(MAX_TOKENS * OVERLAP)
        start_index = 0
        ci = 0
        while start_index < len(tokens):
            end_index = min(start_index + MAX_TOKENS, len(tokens))
            chunk = tokens[start_index:end_index]
            if len(chunk) < overlap:
                break
            chunks.append({
                "chunk_index": ci,
                "chunk_text": tokenizer.decode(chunk),
                "chunk_tokens": chunk,
                "chunk_token_count": len(chunk),
                **{key: row[key] for key in KEEP_KEYS}
            })
            start_index += MAX_TOKENS - overlap
            ci += 1
    else:
        chunks.append({
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_tokens": tokens,
            "chunk_token_count": token_count,
            **{key: row[key] for key in KEEP_KEYS}
        })

    return chunks


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import pandas as pd
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer
    from datasets import load_from_disk, load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=MAX_TOKENS)

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    dataset = load_dataset("arrow", data_files=f"{DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    df = pd.DataFrame(dataset['train'])
    print("dataset", len(df))
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds for {file}") 

    chunks_list = []
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        pbar = tqdm(total=len(df), desc=f"Processing Rows for {file}")
        
        # this gets called inside each thread
        def process_batch(batch):
            batch_chunks = []
            for row in batch:
                row_chunks = chunk_row(row, tokenizer)
                pbar.update(1)
                batch_chunks.extend(row_chunks)
            return batch_chunks

        print(f"making batches for {file}")
        batches = [df.iloc[i:i + BATCH_SIZE].to_dict(orient="records") for i in range(0, len(df), BATCH_SIZE)]
        print(f"made batches for {file}")
        print(f"setting up futures for {file}")
        futures = [executor.submit(process_batch, batch) for batch in batches]
        print(f"in the future for {file}")
        for future in as_completed(futures):
            chunks_list.extend(future.result())
        pbar.close()

    chunked_df = pd.DataFrame(chunks_list)
    file_name = file.split(".")[0]
    import os
    output_dir = f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"saving to {output_dir}/{file_name}.parquet")
    chunked_df.to_parquet(f"{output_dir}/{file_name}.parquet")
    print(f"done with {file}, {len(chunks_list)} chunks")
    volume.commit()
    return f"All done with {file}", len(chunks_list)


@app.local_entrypoint()
def main():
    # download_dataset.remote()
    # from huggingface_hub import HfFileSystem
    # hffs = HfFileSystem()
    # files = hffs.ls("datasets/HuggingFaceFW/fineweb-edu/sample/10BT", detail=False)

    # files = [f"data-{i:05d}-of-00989.arrow" for i in range(989)]
    # files = [f"data-{i:05d}-of-00011.arrow" for i in range(11)]
    
    # process_dataset.remote(file, max_tokens=MAX_TOKENS, num_cpu=NUM_CPU)
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


