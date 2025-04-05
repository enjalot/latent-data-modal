"""
Extract the features for the embeddings of a dataset using a pre-trained SAE model

modal run features.py
"""

import os
import time
from tqdm import tqdm
from latentsae.sae import Sae
from modal import App, Image, Volume, Secret, gpu, enter, method

DATASET_DIR="/embeddings"
VOLUME = "embeddings"

# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4" 
# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2" 
# DIRECTORY = f"{DATASET_DIR}/RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2" 
# DIRECTORY = f"{DATASET_DIR}/pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2" 
# DIRECTORY = f"{DATASET_DIR}/medrag-pubmed-500-nomic-embed-text-v1.5"
# FILES = [f"{DIRECTORY}/train/data-{i:05d}-of-00138.npy" for i in range(138)]
DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5"
FILES = [f"{DIRECTORY}/train/data-{i:05d}-of-00041.npy" for i in range(41)]
SAE = "64_32"
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-2"
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3"
# SAE = "64_128"
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}"
# SAE = "64_64"

SAVE_DIRECTORY = f"{DIRECTORY}-{SAE}"


# MODEL_ID = "enjalot/sae-all-MiniLM-L6-v2"
# D_IN = 384
MODEL_ID = "enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT"
MODEL_DIR = "/model"
D_IN = 768
MODEL_REVISION="main"

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)

def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "einops==0.7.0",
        "latentsae==0.1.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        secrets=[Secret.from_name("huggingface-secret")],
    )
)
app = App(image=st_image)  # Note: prior to April 2024, "app" was called "stub"

with st_image.imports():
    import numpy as np
    import torch

@app.cls(
    volumes={DATASET_DIR: volume}, 
    timeout=60 * 100,
    scaledown_window=60 * 10,
    allow_concurrent_inputs=1,
    image=st_image,
)
class SAEModel:
    @enter()
    def start_engine(self):
        # import torch
        self.device = torch.device("cpu")
        print("🥶 cold starting inference")
        start = time.monotonic_ns()
        self.model = Sae.load_from_hub(MODEL_ID, SAE, device=self.device)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @method()
    def make_features(self, file):
        # Redownload the dataset
        import time
        from datasets import load_dataset
        import torch
        import pandas as pd
        import numpy as np
        import time

        start = time.monotonic_ns()
        print("loading", file)
        # dataset = load_dataset("arrow", data_files=f"{DIRECTORY}/train/{file}")
        # # df = pd.read_parquet(f"{DIRECTORY}/train/{file}")
        # print("loaded")
        # df = pd.DataFrame(dataset['train'])
        # print("converted to dataframe")
        # embeddings = df['embedding'].to_numpy()
        # print("converted to numpy")
        # embeddings = np.array([np.array(e).astype(np.float32) for e in embeddings])
        duration_s = (time.monotonic_ns() - start) / 1e9
        # read the npy memmapped file
        size= os.path.getsize(file) // (D_IN * 4)
        embeddings = np.memmap(file, 
                      dtype='float32', 
                      mode='r', 
                      shape=(size, D_IN))
        print("loaded", file, "in", duration_s)
 
        start = time.monotonic_ns()
        print("Encoding embeddings with SAE")

        # batch_size = 4096
        batch_size = 128
        num_batches = (len(embeddings) + batch_size - 1) // batch_size
        all_acts = np.zeros((len(embeddings), 64))
        all_indices = np.zeros((len(embeddings), 64))
        for i in tqdm(range(num_batches), desc="Encoding batches"):
            batch_embeddings = embeddings[i * batch_size:(i + 1) * batch_size]
            batch_embeddings_tensor = torch.from_numpy(batch_embeddings).float().to(self.device)
            batch_features = self.model.encode(batch_embeddings_tensor)
            all_acts[i * batch_size:(i + 1) * batch_size] = batch_features.top_acts.detach().cpu().numpy()
            all_indices[i * batch_size:(i + 1) * batch_size] = batch_features.top_indices.detach().cpu().numpy()

        duration_s = (time.monotonic_ns() - start) / 1e9
        print("encoding completed", duration_s)

        df = pd.DataFrame()
        df['top_acts'] = list(all_acts)
        df['top_indices'] = list(all_indices)
        # # df.drop(columns=['embedding'], inplace=True)
        # if 'chunk_tokens' in df.columns:
        #     df.drop(columns=['chunk_tokens'], inplace=True)
        print("features generated for", file)

        file_name = os.path.basename(file).split(".")[0]
        output_dir = f"{SAVE_DIRECTORY}/train"
        os.makedirs(output_dir, exist_ok=True)
        print(f"saving to {output_dir}/{file_name}.parquet")
        df.to_parquet(f"{output_dir}/{file_name}.parquet")

        volume.commit()
        return f"done with {file}"

@app.local_entrypoint()
def main():

    # files = files[0:10]
    
    model = SAEModel()

    for resp in model.make_features.map(FILES, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)



