"""
Shared configuration for the latent-data-modal pipeline.

All dataset, model, and infrastructure settings live here so every script
pulls from one source of truth.  Switch datasets by changing ACTIVE_DATASET.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Everything needed to locate and process one dataset."""
    name: str                          # HuggingFace repo id
    save_name: str                     # local directory name on the volume
    text_key: str = "text"             # column that holds the raw text
    keep_keys: list[str] = field(default_factory=list)
    hf_subset: Optional[str] = None    # e.g. "20231101.en" for wikipedia
    hf_data_files: Optional[str] = None
    num_files: int = 1                 # number of arrow/parquet shards
    volume: str = "datasets"           # Modal volume name for raw data


DATASETS = {
    "fineweb-edu-10BT": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        save_name="fineweb-edu-sample-10BT",
        text_key="text",
        keep_keys=["id", "url", "score", "dump"],
        hf_data_files="sample/10BT/*.parquet",
        num_files=99,
        volume="embedding-fineweb-edu",
    ),
    "fineweb-edu-100BT": DatasetConfig(
        name="HuggingFaceFW/fineweb-edu",
        save_name="fineweb-edu-sample-100BT",
        text_key="text",
        keep_keys=["id", "url", "score", "dump"],
        hf_data_files="sample/100BT/*.parquet",
        num_files=989,
        volume="embedding-fineweb-edu",
    ),
    "redpajama-10B": DatasetConfig(
        name="togethercomputer/RedPajama-Data-V2",
        save_name="RedPajama-Data-V2-sample-10B",
        text_key="raw_content",
        keep_keys=["doc_id", "meta"],
        hf_subset="sample-10B",
        num_files=150,
    ),
    "pile-uncopyrighted": DatasetConfig(
        name="monology/pile-uncopyrighted",
        save_name="pile-uncopyrighted",
        text_key="text",
        keep_keys=["meta"],
        num_files=200,  # first 200 of 1987
    ),
    "wikipedia-en": DatasetConfig(
        name="wikimedia/wikipedia",
        save_name="wikipedia-en",
        text_key="text",
        keep_keys=["id", "url", "title"],
        hf_subset="20231101.en",
        num_files=41,
    ),
    "medrag-pubmed": DatasetConfig(
        name="MedRAG/pubmed",
        save_name="medrag-pubmed",
        text_key="content",
        keep_keys=["id", "title", "PMID"],
        num_files=138,
    ),
}


# ---------------------------------------------------------------------------
# Embedding model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    model_id: str
    dim: int                    # embedding dimension
    max_tokens: int = 512       # model max sequence length
    prefix: str = ""            # text prefix for task-specific prompting
    prefix_token_count: int = 0


MODELS = {
    "nomic-v1.5": ModelConfig(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        max_tokens=512,
        prefix="clustering: ",
        prefix_token_count=4,
    ),
    "bge-base": ModelConfig(
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
        max_tokens=512,
    ),
    "minilm": ModelConfig(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        max_tokens=256,
    ),
}


# ---------------------------------------------------------------------------
# SAE (Sparse Autoencoder) definitions
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    """Configuration for a pre-trained SAE."""
    model_id: str       # HuggingFace repo for the SAE weights
    k: int = 64         # top-k features to extract
    expansion: int = 32 # expansion factor (latent_dim = d_in * expansion)

    @property
    def slug(self) -> str:
        return f"{self.k}_{self.expansion}"

    @property
    def num_features(self) -> int:
        return self.k * self.expansion  # not exactly—actual is d_in * expansion


SAE_MODELS = {
    "nomic-64x32": SAEConfig(
        model_id="enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT",
        k=64,
        expansion=32,
    ),
    "nomic-64x128": SAEConfig(
        model_id="enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT",
        k=64,
        expansion=128,
    ),
}


# ---------------------------------------------------------------------------
# Active selections — change these to switch what the pipeline operates on
# ---------------------------------------------------------------------------

ACTIVE_DATASET = "wikipedia-en"
ACTIVE_MODEL = "nomic-v1.5"
ACTIVE_SAE = "nomic-64x32"


# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS = 500    # max tokens per chunk (also try 120)
CHUNK_OVERLAP = 0.1       # 10 % sliding-window overlap


# ---------------------------------------------------------------------------
# TEI / GPU infrastructure
# ---------------------------------------------------------------------------

GPU_TYPE = "A10G"
GPU_CONCURRENCY = 10      # max parallel GPU containers
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-1.2"
# For A100/H100 use the appropriate images:
# TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:1.2"       # A100
# TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:hopper-1.2" # H100


# ---------------------------------------------------------------------------
# Derived helpers (used across scripts)
# ---------------------------------------------------------------------------

def get_dataset() -> DatasetConfig:
    return DATASETS[ACTIVE_DATASET]

def get_model() -> ModelConfig:
    return MODELS[ACTIVE_MODEL]

def get_sae() -> SAEConfig:
    return SAE_MODELS[ACTIVE_SAE]

def model_slug() -> str:
    return get_model().model_id.split("/")[-1]

def chunked_dataset_name() -> str:
    ds = get_dataset()
    return f"{ds.save_name}-chunked-{CHUNK_MAX_TOKENS}"

def embedding_dataset_name() -> str:
    return f"{chunked_dataset_name()}-{model_slug()}"

def features_dataset_name() -> str:
    return f"{embedding_dataset_name()}-{get_sae().slug}"

def shard_files(ext: str = "arrow") -> list[str]:
    ds = get_dataset()
    return [f"data-{i:05d}-of-{ds.num_files:05d}.{ext}" for i in range(ds.num_files)]
