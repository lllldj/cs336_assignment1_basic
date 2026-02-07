from __future__ import annotations

import os
import regex as re
import heapq

import numpy.typing as npt
import torch

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import IO, Any, BinaryIO
from jaxtyping import Bool, Float, Int
from torch import Tensor



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(filepath, bound_st, bound_ed, pattern, special_tokens):
    with open(filepath, "rb") as f:
        f.seek(bound_st)
        chunk = f.read(bound_ed - bound_st).decode("utf-8", errors="ignore")
        special_pat = "|".join(map(re.escape, special_tokens))
        chunk_set = [s for s in re.split(special_pat, chunk) if s]
        corpus_weights = {}
        for small_chunk in chunk_set:
            splited_text = re.findall(pattern, small_chunk)
            for words in splited_text:
                data_u8 = words.encode("utf-8")
                corpus_weights[data_u8] = corpus_weights.get(data_u8, 0) + 1
    return corpus_weights


def build_vocab_and_merges_from_tokens(
    tokens: dict[int, tuple[int, ...]],
    created_ids: list[int] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    tokens:
        - base token: tokens[id] = (byte,)      where byte is 0..255
        - merged token: tokens[id] = (l_id, r_id)

    created_ids:
        optional list of merged token ids in creation order.
        If None, infer by sorting ids with len(tokens[id])==2.
        (This is correct iff merged token ids are assigned monotonically increasing.)
    """
    vocab: dict[int, bytes] = {}
    visiting: set[int] = set()

    def decode(tid: int) -> bytes:
        if tid in vocab:
            return vocab[tid]
        if tid in visiting:
            raise ValueError(f"Cycle detected at token {tid}")
        if tid not in tokens:
            raise KeyError(f"Token {tid} not found in tokens table")

        visiting.add(tid)
        spec = tokens[tid]

        if len(spec) == 1:
            b = spec[0]
            if not (0 <= b <= 255):
                raise ValueError(f"Base token {tid} has invalid byte {b}")
            out = bytes([b])
        elif len(spec) == 2:
            l_id, r_id = spec
            out = decode(l_id) + decode(r_id)
        else:
            raise ValueError(f"Token {tid} has invalid arity {len(spec)}")

        visiting.remove(tid)
        vocab[tid] = out
        return out

    # infer creation order of merges if not provided
    if created_ids is None:
        created_ids = sorted([tid for tid, spec in tokens.items() if len(spec) == 2])

    merges: list[tuple[bytes, bytes]] = []
    for tid in created_ids:
        l_id, r_id = tokens[tid]  # guaranteed len==2
        lb, rb = decode(l_id), decode(r_id)
        merges.append((lb, rb))
        decode(tid)  # ensure vocab for this merged token is filled

    # ensure vocab for all tokens exists (optional, but usually desired)
    for tid in tokens.keys():
        decode(tid)

    return vocab, merges

        



def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    merge_ops = vocab_size - 256 - len(special_tokens)
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
    parellel_params = [(input_path, start, end, PAT, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        results = list(ex.map(pre_tokenize, *zip(*parellel_params)))
    
    word_weights = {}   #{words: (word_now,frequency)}
    dict_of_pair = {}   #{(ch1, ch2): frequency}, true frequency
    pair_to_words = {}  #{(ch1, ch2): set(words)}
    tokens = {i:(i,) for i in range(256)}  #{token_id: [bytestring]}    
    
    for dic in results:
        for k, v in dic.items():
            word_weights[k] = word_weights.get(k, (k,0))
            word_weights[k] = (k, word_weights[k][1] + v)
    for k,v in word_weights.items():
        for i in range(len(k)-1):
            ch1 = k[i]
            ch2 = k[i+1]
            pair = (ch1, ch2)
            dict_of_pair[pair] = dict_of_pair.get(pair, 0) + v[1]
            pair_to_words.setdefault(pair, set()).add(k)
            
    pair_freq_heap = [(-freq, pair) for pair, freq in dict_of_pair.items()]
    heapq.heapify(pair_freq_heap)
    
    valid_merge = 0
    token_id = 256
    while(valid_merge < merge_ops):
        neg_freq, pair = heapq.heappop(pair_freq_heap)
        freq = -neg_freq
        if dict_of_pair.get(pair, 0) != freq:
            continue
        
        idx_now = token_id
        token_id += 1
        tokens[idx_now] = pair
        
        for word_id in pair_to_words[pair]:
            new_word = []
            word = word_weights[word_id][0]
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(idx_now)
                    i += 2           
                    dict_of_pair[pair] -= word_weights[word_id][1]
                else:
                    new_word.append(word[i])
                    i += 1
            
            for i in range(len(new_word)-1):
                if new_word[i] == idx_now:
                    if i + 1 < len(new_word):
                        if new_word[i+1] == idx_now:
                            new_pair_post = (idx_now, idx_now)
                            old_pair_post = (pair[1],pair[0])
                            dict_of_pair[new_pair_post] = dict_of_pair.get(new_pair_post, 0) + word_weights[word_id][1]
                            dict_of_pair[old_pair_post] = dict_of_pair.get(old_pair_post, 0) - word_weights[word_id][1]
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[new_pair_post], new_pair_post))
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[old_pair_post], old_pair_post))   
                            pair_to_words.setdefault(new_pair_post, set()).add(word_id)
                        else:
                            new_pair_post = (idx_now, new_word[i+1])
                            old_pair_post = (pair[1], new_word[i+1])
                            dict_of_pair[new_pair_post] = dict_of_pair.get(new_pair_post, 0) + word_weights[word_id][1]
                            dict_of_pair[old_pair_post] = dict_of_pair.get(old_pair_post, 0) - word_weights[word_id][1]
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[new_pair_post], new_pair_post))
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[old_pair_post], old_pair_post))
                            pair_to_words.setdefault(new_pair_post, set()).add(word_id)
                    if i > 0:
                        if new_word[i-1] == idx_now:
                            pass
                        else:
                            new_pair_pre = (new_word[i-1], idx_now)
                            old_pair_pre = (new_word[i-1], pair[0])
                            dict_of_pair[new_pair_pre] = dict_of_pair.get(new_pair_pre, 0) + word_weights[word_id][1]
                            dict_of_pair[old_pair_pre] = dict_of_pair.get(old_pair_pre, 0) - word_weights[word_id][1]
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[new_pair_pre], new_pair_pre))
                            heapq.heappush(pair_freq_heap, (-dict_of_pair[old_pair_pre], old_pair_pre))
                            pair_to_words.setdefault(new_pair_pre, set()).add(word_id)

            word_weights[word_id] = (new_word, word_weights[word_id][1])
            
        valid_merge += 1
        #print(f"Token {idx_now} , pair {pair}", end="\r")
        
    v, m = build_vocab_and_merges_from_tokens(tokens)
    tk_id = 
    for tk in special_tokens:
        v[tk_id] = tk.encode("utf-8")
        tk_id += 1
    return v,m