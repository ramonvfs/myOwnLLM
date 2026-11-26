"""Compara tokenizers num mesmo texto de validação em PT-BR.

Métricas (todas sobre o mesmo texto):
  - bytes/token   : compressão (maior = cada token cobre mais texto)
  - tokens/palavra: "fertilidade" (menor = menos cortes por palavra; 1.0 = uma palavra por token)
  - palavras/ctx  : quantas palavras de português cabem numa janela de N tokens
  - tempo         : tempo do encode (a 1a chamada; os seus usam cache)

Uso:
    uv add tiktoken transformers   # transformers só se quiser os tokenizers do HF
    uv run python scripts/compare_tokenizers.py --val data/wiki_sample/val.txt \
        --model data/tokenizer/wiki32k.model --ctx 1024 2048
"""

import argparse
import time
from itertools import islice
from pathlib import Path

from myownllm.tokenizer.bpe import BPETokenizer

HF_TOKENIZERS = [
    "TucanoBR/Tucano-160m",  # decoder nativo em PT (base Llama 2)
    "neuralmind/bert-base-portuguese-cased",  # BERTimbau (WordPiece, encoder)
]
TIKTOKEN_ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


def truncate(tok, vocab_size):
    """Tokenizer com só os primeiros (vocab_size - 256) merges de `tok`."""
    t = BPETokenizer()
    t.pattern = tok.pattern
    t.merges = dict(islice(tok.merges.items(), vocab_size - 256))
    t.vocab = t._build_vocab()
    t.register_special_tokens(list(tok.special_tokens))
    return t


def own_tokenizers(model_path, sizes):
    base = BPETokenizer()
    base.load(model_path)
    for size in sizes:
        if size - 256 > len(base.merges):
            print(
                f"[aviso] {model_path} tem só {len(base.merges)} merges; pulando {size}"
            )
            continue
        t = truncate(base, size)
        yield f"myownllm-{size // 1024}k", len(t.vocab), t.encode_ordinary


def tiktoken_tokenizers():
    try:
        import tiktoken
    except ImportError:
        print("[aviso] tiktoken não instalado; pulando")
        return
    for name in TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(name)
        yield f"tiktoken/{name}", enc.n_vocab, enc.encode_ordinary


def hf_tokenizers():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[aviso] transformers não instalado; pulando os do Hugging Face")
        return
    for name in HF_TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # sem rede, nome errado etc.
            print(f"[aviso] não consegui carregar {name}: {e}")
            continue
        yield name, len(tok), lambda s, tok=tok: tok.encode(s, add_special_tokens=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, required=True)
    p.add_argument(
        "--model",
        required=True,
        help="seu .model (o maior; os menores saem por truncagem)",
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[8192, 16384, 32768])
    p.add_argument("--ctx", type=int, nargs="+", default=[1024, 2048])
    args = p.parse_args()

    text = args.val.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    print(
        f"texto: {n_bytes / 1024:.0f} KB, {n_words} palavras, {len(text)} caracteres\n"
    )

    rows = []
    for source in (
        own_tokenizers(args.model, args.sizes),
        tiktoken_tokenizers(),
        hf_tokenizers(),
    ):
        for name, vocab_size, encode in source:
            t0 = time.perf_counter()
            ids = encode(text)
            dt = time.perf_counter() - t0
            rows.append((name, vocab_size, n_bytes / len(ids), len(ids) / n_words, dt))

    ctx_cols = "".join(f"  pal/{c:<5d}" for c in args.ctx)
    print(
        f"{'tokenizer':42s} {'vocab':>7s} {'bytes/tok':>9s} {'tok/pal':>7s}{ctx_cols} {'tempo':>7s}"
    )
    for name, vocab_size, bpt, fert, dt in sorted(rows, key=lambda r: -r[2]):
        ctx = "".join(f"  {c / fert:9.0f}" for c in args.ctx)
        print(f"{name:42s} {vocab_size:7d} {bpt:9.2f} {fert:7.2f}{ctx} {dt:6.2f}s")


if __name__ == "__main__":
    main()
