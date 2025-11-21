import torch
import numpy as np
import itertools
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import pickle
from pathlib import Path

# Import configuration
from config import (
    CACHE_DIR,
    MAX_ENHANCER_LENGTH, 
    MAX_PROMOTER_LENGTH,
    ENHANCER_FEATURE_DIM,
    PROMOTER_FEATURE_DIM,
    PROJECT_ROOT,
    CACHE_DIR
)

# Global tokenizer cache
_TOKENIZER_CACHE = None
_TOKENIZER_CACHE_PATH = os.path.join(CACHE_DIR, "tokenizer_cache.pkl")

def get_tokenizer(force_recreate: bool = False) -> Dict[str, int]:
    global _TOKENIZER_CACHE
    
    # Return cache directly if exists and not forcing recreation
    if _TOKENIZER_CACHE is not None and not force_recreate:
        return _TOKENIZER_CACHE
    
    # Try to load cache from file
    cache_dir = os.path.dirname(_TOKENIZER_CACHE_PATH)
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(_TOKENIZER_CACHE_PATH) and not force_recreate:
        try:
            with open(_TOKENIZER_CACHE_PATH, 'rb') as f:
                _TOKENIZER_CACHE = pickle.load(f)
            print(f"Loaded tokenizer from cache: {_TOKENIZER_CACHE_PATH}")
            return _TOKENIZER_CACHE
        except Exception as e:
            print(f"Failed to load tokenizer cache: {e}, recreating")
    
    # Create new tokenizer
    print("Creating tokenizer dictionary...")
    bases = ['A', 'C', 'G', 'T']
    k = 6  # 6-mer
    
    # Use itertools.product to generate all possible 6-mer combinations
    products = itertools.product(bases, repeat=k)
    tokens = [''.join(p) for p in products]
    
    # Create token to index mapping dictionary
    token_dict = {token: idx + 1 for idx, token in enumerate(tokens)}  # Start indexing from 1
    token_dict['null'] = 0  # null token index is 0
    
    _TOKENIZER_CACHE = token_dict
    
    # Save to cache file
    try:
        with open(_TOKENIZER_CACHE_PATH, 'wb') as f:
            pickle.dump(_TOKENIZER_CACHE, f)
        print(f"Saved tokenizer to cache: {_TOKENIZER_CACHE_PATH}")
    except Exception as e:
        print(f"Failed to save tokenizer cache: {e}")
    
    return _TOKENIZER_CACHE


def sequence_to_tokens_fast(sequence: str, k: int = 6) -> List[str]:
    # Pre-allocate list size for better performance
    seq_len = len(sequence)
    if seq_len < k:
        return ['null']
    
    # Use list comprehension to generate k-mers, more efficient than loops
    tokens = [sequence[i:i+k] for i in range(seq_len - k + 1)]
    
    # Check if each token contains 'N', replace with 'null' if so
    return [token if 'N' not in token else 'null' for token in tokens]


def tokens_to_ids_fast(tokens: List[str], tokenizer: Dict[str, int]) -> torch.Tensor:
    # Use list comprehension and dictionary lookup, more efficient than loops
    token_ids = [tokenizer.get(token, 0) for token in tokens]
    return torch.tensor(token_ids, dtype=torch.long)


class OptimizedSequenceDataset(Dataset):
    
    def __init__(
        self, 
        enhancers: List[str], 
        promoters: List[str], 
        labels: List[int],
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Validate data length consistency
        assert len(enhancers) == len(promoters) == len(labels), \
            "Enhancers, promoters, and labels must have the same length"
        
        self.length = len(labels)
        
        # Get tokenizer
        self.tokenizer = get_tokenizer()
        
        # Create cache directory
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Pre-compute feature tensors to avoid repeated creation
        self.enhancer_features = torch.zeros(*ENHANCER_FEATURE_DIM)
        self.promoter_features = torch.zeros(*PROMOTER_FEATURE_DIM)
        
        # Initialize cache index
        self._cache_index = set()
        if self.use_cache and self.cache_dir:
            self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index"""
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'rb') as f:
                    self._cache_index = pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache index: {e}")
                self._cache_index = set()
    
    def _save_cache_index(self):
        """Save cache index"""
        if not self.use_cache or not self.cache_dir:
            return
            
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self._cache_index, f)
        except Exception as e:
            print(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, idx: int) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"data_{idx}.pt")
    
    def _process_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Get raw data
        enhancer_seq = self.enhancers[idx]
        promoter_seq = self.promoters[idx]
        label = self.labels[idx]
        
        # Convert to tokens
        enhancer_tokens = sequence_to_tokens_fast(enhancer_seq)
        promoter_tokens = sequence_to_tokens_fast(promoter_seq)
        
        # Convert to ID tensors
        enhancer_ids = tokens_to_ids_fast(enhancer_tokens, self.tokenizer)
        promoter_ids = tokens_to_ids_fast(promoter_tokens, self.tokenizer)
        
        # Truncate sequences that are too long
        if len(enhancer_ids) > MAX_ENHANCER_LENGTH:
            enhancer_ids = enhancer_ids[:MAX_ENHANCER_LENGTH]
        if len(promoter_ids) > MAX_PROMOTER_LENGTH:
            promoter_ids = promoter_ids[:MAX_PROMOTER_LENGTH]
        
        # Clone feature tensors to avoid all samples sharing the same tensor
        enhancer_features = self.enhancer_features.clone()
        promoter_features = self.promoter_features.clone()
        
        return enhancer_ids, promoter_ids, enhancer_features, promoter_features, label
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # If using cache and data exists in cache, load directly
        if self.use_cache and self.cache_dir and idx in self._cache_index:
            cache_path = self._get_cache_path(idx)
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    print(f"Failed to load cached data: {e}")
        
        # Process data
        data = self._process_item(idx)
        
        # Save to cache
        # if self.use_cache and self.cache_dir:
        #     cache_path = self._get_cache_path(idx)
        #     try:
        #         torch.save(data, cache_path)
        #         self._cache_index.add(idx)
        #         # Save index every 100 samples to reduce IO operations
        #         if idx % 100 == 0:
        #             self._save_cache_index()
        #     except Exception as e:
        #         print(f"Failed to save cached data: {e}")
        
        return data
    
    def __len__(self) -> int:
        return self.length


def create_optimized_dataset(
    enhancers: List[str], 
    promoters: List[str], 
    labels: List[int],
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    num_workers: int = 4
) -> OptimizedSequenceDataset:
    # if there is no cache directory specified, create a default cache directory
    if cache_dir is None and use_cache:
        cache_dir = os.path.join(CACHE_DIR, "dataset_cache")
    

    dataset = OptimizedSequenceDataset(
        enhancers=enhancers,
        promoters=promoters,
        labels=labels,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    return dataset


def clear_tokenizer_cache():

    global _TOKENIZER_CACHE
    _TOKENIZER_CACHE = None
    if os.path.exists(_TOKENIZER_CACHE_PATH):
        os.remove(_TOKENIZER_CACHE_PATH)
       


def warmup_cache(dataset: OptimizedSequenceDataset, num_samples: int = 100):
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Preheating cache"):
        _ = dataset[i]
    print("Cache preheating completed")
