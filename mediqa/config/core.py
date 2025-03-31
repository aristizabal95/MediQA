from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load
from typing import Optional

import mediqa

# Project Directories
PACKAGE_ROOT = Path(mediqa.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"

class DataConfig(BaseModel):
    eval_src_file: str
    split_frac: float
    val_file: str
    test_file: str
    knowledge_dset: str
    knowledge_path: str

class ReaderConfig(BaseModel):
    model_name: str
    quantize: bool
    temperature: float
    do_sample: bool
    repetition_penalty: float
    max_new_tokens: int

class RAGConfig(BaseModel):
    embedding_dimension: int
    index_name: str
    encoder_name: str
    chunk_size: int
    batch_size: int
    encoding_strategy: str
    db_location: str
    n_results: int
    knowledge_path: str

class Config(BaseModel):
    """Master config object."""
    data_config: DataConfig
    reader_config: ReaderConfig
    rag_config: RAGConfig

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration"""
    if cfg_path is None:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as f:
            parsed_config = load(f.read())
            return parsed_config
    raise OSError(f"Did not find config file at: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        data_config=DataConfig(**parsed_config.data),
        reader_config=ReaderConfig(**parsed_config.data),
        rag_config=RAGConfig(**parsed_config.data),
    )

    return _config

config = create_and_validate_config()