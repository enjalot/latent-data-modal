"""
Try to figure out the optimal batch size for embedding on a given GPU
"""
import os
import json
import time
import asyncio
import subprocess

from modal import App, Image, Secret, Volume, build, enter, exit, gpu, method

# We first set out configuration variables for our script.
## Embedding Containers Configuration
# GPU_CONCURRENCY = 100
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]

MODEL_DIR = "/model"
MODEL_REVISION="e5cf08aadaa33385f5990def41f7a23405aec398"

GPU_CONCURRENCY = 1
# GPU_CONFIG = gpu.A100(size="80GB")
# GPU_CONFIG = gpu.A100(size="40GB")
# GPU_CONFIG = gpu.A10G()
GPU_CONFIG = gpu.H100()
# BATCH_SIZE = 512
BATCH_SIZE = 64
# BATCH_SIZE = 128
MAX_TOKENS = 8192
# MAX_TOKENS = 2048


## Dataset-Specific Configuration
DATASET_READ_VOLUME = Volume.from_name(
    "embedding-fineweb-edu", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = Volume.from_name(
    "checkpoint", create_if_missing=True
)
DATASET_DIR = "/data"
# DATASET_SAVE ="fineweb-edu-sample-10BT"
DATASET_SAVE ="fineweb-edu-sample-10BT-100k"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = True

## Upload-Specific Configuration
# DATASET_HF_UPLOAD_REPO_NAME = "enjalot/fineweb-edu-sample-10BT"
DATASET_HF_UPLOAD_REPO_NAME = f"enjalot/{DATASET_SAVE}"
UPLOAD_TO_HF = False


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
        "einops==0.7.0"
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
with st_image.imports():
    import numpy as np
    import torch
    from torch.cuda.amp import autocast
    from transformers import AutoTokenizer, AutoModel

app = App(
    "fineweb-embeddings-st"
)  

@app.cls(
    gpu=GPU_CONFIG,
    # cpu=16,
    concurrency_limit=GPU_CONCURRENCY,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=1,
    image=st_image,
)
class TransformerModel:
    @enter()
    def start_engine(self):
        # import torch
        # from transformers import AutoTokenizer, AutoModel

        self.device = torch.device("cuda")

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)#, rotary_scaling_factor=2 )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=MAX_TOKENS)
        self.model.to(self.device)
        self.model.eval()

        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @method()
    def embed(self, inputs):
        tok = self.tokenizer
        
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(torch.cuda.memory_summary(device=self.device, abbreviated=True))

        # print(f"CUDA memory allocated before encoding: {torch.cuda.memory_allocated() / 1e6} MB")

        start = time.monotonic_ns()
        encoded_input = tok(inputs, padding=True, truncation=True, return_tensors='pt')
        print("encoded in", (time.monotonic_ns() - start) / 1e9)

        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}
        # print("moved to device", (time.monotonic_ns() - start) / 1e9)
        # print("encoded input size", encoded_input['input_ids'].nelement() * encoded_input['input_ids'].element_size() / 1e6, "MB")

        # print(f"CUDA memory allocated after encoding: {torch.cuda.memory_allocated() / 1e6} MB")
        start = time.monotonic_ns()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        with torch.no_grad():#, autocast():
            # print(f"CUDA memory allocated before embedding: {torch.cuda.memory_allocated() / 1e6} MB")
            model_output = self.model(**encoded_input)
            # print(f"CUDA memory allocated after model output: {torch.cuda.memory_allocated() / 1e6} MB")
            # print(f"model output size: {model_output.nelement() * model_output.element_size() / 1e6} MB")
            embeddings = model_output[0][:, 0]
            # print(f"Embedding size: {embeddings.nelement() * embeddings.element_size() / 1e6} MB")
            # print(f"CUDA memory allocated after embedding: {torch.cuda.memory_allocated() / 1e6} MB")
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            normalized_embeddings_cpu = normalized_embeddings.cpu().numpy()
            # print(f"CUDA memory allocated after got embeddings: {torch.cuda.memory_allocated() / 1e6} MB")
            # # Clean up torch memory
            # del encoded_input
            # del model_output
            # del embeddings
            # del normalized_embeddings
            # torch.cuda.empty_cache()
            duration_ms = (time.monotonic_ns() - start) / 1e6
            print(f"embedding took {duration_ms:.0f}ms")
            print(torch.cuda.memory_summary(device=self.device, abbreviated=True))

            return inputs, normalized_embeddings_cpu



@app.local_entrypoint()
def full_job():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=MAX_TOKENS)
    batch_size = BATCH_SIZE

    test = "I "
    test = test * 1022
    tokens = tok.encode(test)
    print("tokens", len(tokens))

    inputs = [test] * (384)

    model = TransformerModel()
    [inputs, embeddings] = model.embed.remote(inputs=inputs)
    print("done")

