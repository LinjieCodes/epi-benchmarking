import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"

# cell line list
CELL_LINES = ["GM12878", "HUVEC", "HeLa-S3", "IMR90", "K562", "NHEK"]

# dataset types
DATA_TYPES = ["train", "val", "test"]


def load_sequence_data(seq_file: Path) -> Dict[str, str]:

    df = pd.read_csv(seq_file)
    return dict(zip(df['region'], df['sequence']))


def load_pairs_data(pairs_file: Path) -> pd.DataFrame:
 
    return pd.read_csv(pairs_file)


def prepare_cell_data(cell_line: str, data_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # construct file paths
    cell_dir = DATA_DIR / data_type / cell_line
    pairs_file = cell_dir / "pairs_hg38.csv"
    e_seq_file = cell_dir / "e_seq.csv"
    p_seq_file = cell_dir / "p_seq.csv"
    
    # load data
    pairs_df = load_pairs_data(pairs_file)
    e_sequences = load_sequence_data(e_seq_file)
    p_sequences = load_sequence_data(p_seq_file)
    
    # extract sequences and labels
    enhancers = [e_sequences[enhancer] for enhancer in pairs_df['enhancer_name']]
    promoters = [p_sequences[promoter] for promoter in pairs_df['promoter_name']]
    labels = pairs_df['label'].values
    
    return np.array(enhancers), np.array(promoters), labels


def create_datasets_for_cell(cell_line: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    datasets = {}
    for data_type in DATA_TYPES:
        datasets[data_type] = prepare_cell_data(cell_line, data_type)
    
    return datasets


def get_available_cells() -> List[str]:
    available_cells = []
    test_dir = DATA_DIR / "test"
    
    if test_dir.exists():
        for item in test_dir.iterdir():
            if item.is_dir() and (item / "pairs_hg38.csv").exists():
                available_cells.append(item.name)
    
    return sorted(available_cells)


def load_all_test_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    test_data = {}
    for cell_line in CELL_LINES:
        if (DATA_DIR / "test" / cell_line).exists():
            test_data[cell_line] = prepare_cell_data(cell_line, "test")
    
    return test_data


def load_train_data(cell_line: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return prepare_cell_data(cell_line, "train")