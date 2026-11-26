"""Amostra de texto da Wikipedia em português para treinar/avaliar o tokenizer.
 
Lê o dataset `wikimedia/wikipedia` em modo streaming (não baixa o dump inteiro),
normaliza cada artigo em NFC e escreve dois arquivos com artigos disjuntos:
 
    <out-dir>/train.txt
    <out-dir>/val.txt
 
Uso:
    uv run python scripts/data/sample_wikipedia.py --out-dir data/wiki_sample \
        --train-mb 1 --val-mb 0.2
"""

import argparse
import time
import unicodedata
from pathlib import Path

from datasets import load_dataset

SEPARATOR = "\n\n"  # entre artigos; no pré-treino isso vira <|endoftext|>


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--train-mb",
        type=float,
        default=1.0,
        help="tamanho alvo do treino, em MB (UTF-8)",
    )
    p.add_argument(
        "--val-mb",
        type=float,
        default=0.2,
        help="tamanho alvo da validação, em MB (UTF-8)",
    )
    p.add_argument(
        "--config",
        default="20231101.pt",
        help="config do wikimedia/wikipedia (data.língua)",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="1 a cada N artigos vai para a validação",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="descarta artigos mais curtos que isso",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "train": int(args.train_mb * 1024 * 1024),
        "val": int(args.val_mb * 1024 * 1024),
    }
    written = {"train": 0, "val": 0}
    n_articles = {"train": 0, "val": 0}
    n_skipped = 0

    ds = load_dataset("wikimedia/wikipedia", args.config, split="train", streaming=True)
    # O stream vem em ordem de id; embaralhar com um buffer evita pegar só os
    # primeiros artigos criados (que não são uma amostra representativa).
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    files = {
        split: open(args.out_dir / f"{split}.txt", "w", encoding="utf-8")
        for split in targets
    }
    t0 = time.perf_counter()
    try:
        for i, article in enumerate(ds):
            text = unicodedata.normalize("NFC", article["text"]).strip()
            if len(text) < args.min_chars:
                n_skipped += 1
                continue

            split = "val" if i % args.val_every == 0 else "train"
            if written[split] >= targets[split]:
                # esse split já encheu: manda para o outro, se ainda couber
                split = "train" if split == "val" else "val"
                if written[split] >= targets[split]:
                    break

            chunk = text + SEPARATOR
            files[split].write(chunk)
            written[split] += len(chunk.encode("utf-8"))
            n_articles[split] += 1
    finally:
        for f in files.values():
            f.close()

    dt = time.perf_counter() - t0
    for split in targets:
        print(
            f"{split:5s}: {n_articles[split]:6d} artigos, {written[split] / 1024 / 1024:.2f} MB "
            f"-> {args.out_dir / f'{split}.txt'}"
        )
    print(f"descartados (< {args.min_chars} chars): {n_skipped}")
    print(f"tempo: {dt:.1f}s")


if __name__ == "__main__":
    main()
