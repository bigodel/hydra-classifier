from pathlib import Path
from typing import TypedDict, Union, TypeAlias, Tuple

# Constants
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
METADATA_FILEPATH = DATA_DIR / 'metadata.csv'

BATCH_SIZE = 8
EPOCHS = 1
BERT_BASE = 'bert-base-uncased'
MAX_SEQUENCE_LENGHT = 512
MODEL_DIR = Path('./model')

# Types
FilePath: TypeAlias = Union[str, Path]


class PageMetadata(TypedDict):
    page_number: int
    file_relpath: FilePath
    width: int
    height: int
    label: str


ImageSize: TypeAlias = Tuple[int, int]
ImageInputShape: TypeAlias = Tuple[int, int, int]
