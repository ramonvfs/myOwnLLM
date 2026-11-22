import regex as re
import unicodedata
from collections import Counter

SPLIT_PATTERN = r"""[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self) -> None:
        self.pattern = re.compile(SPLIT_PATTERN)
        self.merges = {}
        self.special_tokens = {}
        self.special_pattern = None
        self.inverse_special = {}
        self.vocab = self._build_vocab()
        self.cache = {}
        self.cache_max = 1_000_000

    def train(self, text, vocab_size, verbose=False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = Counter(re.findall(self.pattern, text))
        ids_lists = [
            (list(chunk.encode("utf-8")), freq) for chunk, freq in chunks.items()
        ]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for ids, freq in ids_lists:
                self._get_stats(ids, freq, stats)
            if not stats:
                break
            pair = max(stats, key=stats.__getitem__)
            if stats[pair] < 2:
                break
            idx = 256 + i
            ids_lists = [(self._merge(ids, pair, idx), freq) for ids, freq in ids_lists]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = self._build_vocab()
        if self.special_tokens:
            self.register_special_tokens(list(self.special_tokens))
        self.cache.clear()

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        ids = []
        for chunk in re.findall(self.pattern, text):
            chunk_ids = self.cache.get(chunk)
            if chunk_ids is None:
                chunk_ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(self.cache) >= self.cache_max:
                    self.cache.clear()
                self.cache[chunk] = chunk_ids
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text, allowed_special="none"):
        if allowed_special == "none" or not self.special_tokens:
            return self.encode_ordinary(text)
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        else:
            allowed = set(allowed_special)  # e.g.: {"<|endoftext|>"}

        ids = []
        for part in self.special_pattern.split(text):
            if part in allowed:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def register_special_tokens(self, tokens):
        start = 256 + len(self.merges)
        self.special_tokens = {tok: start + i for i, tok in enumerate(tokens)}
        self.inverse_special = {i: tok for tok, i in self.special_tokens.items()}
        self.vocab = self._build_vocab()
        if self.special_tokens:
            alts = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(t) for t in alts) + ")"
            )
        else:
            self.special_pattern = None

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    @staticmethod
    def _get_stats(ids, freq=1, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + freq
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    @staticmethod
    def _render_token(t):
        try:
            s = t.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"\\x{b:02x}" for b in t)
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}")
        return "".join(chars)

    def save(self, prefix):
        for tok in self.special_tokens:
            assert "\n" not in tok, f"Special token with line break: {tok!r}"
        assert "\n" not in self.pattern.pattern
        for k, (pair, idx) in enumerate(self.merges.items()):
            assert idx == 256 + k, f"Merges out of order: {pair} -> {idx}"

        with open(prefix + ".model", "w", encoding="utf-8") as f:
            f.write("myownllm bpe v3\n")
            f.write(f"{self.pattern.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for tok, idx in self.special_tokens.items():
                f.write(f"{idx} {tok}\n")
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = self._render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self._render_token(self.vocab[idx0])
                    s1 = self._render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, path):
        assert path.endswith(".model")

        with open(path, encoding="utf-8") as f:
            version = f.readline().rstrip("\n")
            if version != "myownllm bpe v3":
                raise ValueError(f"Version not found: {version!r}")

            pattern = f.readline().rstrip("\n")

            specials = []
            n_special = int(f.readline())
            for _ in range(n_special):
                id_str, tok = f.readline().rstrip("\n").split(" ", 1)
                specials.append((tok, int(id_str)))

            merges = {}
            idx = 256
            for line in f:
                if not line.strip():
                    continue
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.pattern = re.compile(pattern)
        self.merges = merges
        self.special_tokens = {}
        self.register_special_tokens([tok for tok, _ in specials])

        for tok, file_id in specials:
            if self.special_tokens[tok] != file_id:
                raise ValueError(
                    f"Inconsistent id for {tok!r}: file={file_id}, "
                    f"expected={self.special_tokens[tok]}"
                )
        self.cache.clear()
