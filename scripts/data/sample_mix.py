"""Amostra uma mistura de fontes em PT-BR para treinar/avaliar o tokenizer.

Mistura padrão:
    ~90% FineWeb2-HQ por_Latn (web filtrada por qualidade), sem domínios de outros
         países lusófonos (.pt, .ao, .mz, ...)
    ~10% Wikipedia PT

Cada fonte é lida em streaming, limpa (NFC, NBSP -> espaço, sem caracteres de
controle) e escrita em dois arquivos com documentos disjuntos:

    <out-dir>/train.txt
    <out-dir>/val.txt

Uso:
    uv run python scripts/data/sample_mix.py --out-dir data/mix_sample --train-mb 50 --val-mb 2
"""

import argparse
import re
import regex
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from datasets import load_dataset

SEPARATOR = "\n\n"  # entre documentos; no pré-treino vira <|endoftext|>

NON_BR_TLDS = (".pt", ".ao", ".mz", ".cv", ".gw", ".st", ".tl", ".mo")

# caracteres de controle (categoria Unicode "Cc"), exceto \n e \t
_CONTROL = re.compile(r"[^\S\n\t ]|[\x00-\x08\x0b-\x1f\x7f-\x9f]")

import regex

PT_PT = regex.compile(
    r"\b(facto|factos|fenómenos?|económic\w*|académic\w*|António|género|"
    r"actual\w*|activ\w*|acção|acções|direcção|óptim\w*|equipa|equipas|"
    r"telemóvel|autocarro|comboio|ecrã|utilizador\w*|registo|percentagem|"
    r"(?:estou|está|estás|estão|estava|estavam|estamos) a \p{L}+r)\b",
    regex.IGNORECASE,
)
PT_BR = regex.compile(
    r"\b(fato|fatos|fenômenos?|econômic\w*|acadêmic\w*|Antônio|gênero|"
    r"atual\w*|ativ\w*|ação|ações|direção|ótim\w*|equipe|equipes|"
    r"celular|ônibus|trem|tela|usuári\w*|registro|porcentagem|"
    r"(?:estou|está|estão|estava|estavam|estamos) \p{L}+ndo)\b",
    regex.IGNORECASE,
)


def looks_br(doc) -> bool:
    text = doc.get("text", "")
    pt, br = len(PT_PT.findall(text)), len(PT_BR.findall(text))
    return not (pt >= 2 and pt > br)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")  # NBSP -> espaço normal
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(lambda m: " " if m.group().isspace() else "", text)
    return text.strip()


def not_other_lusophone(doc) -> bool:
    host = urlparse(doc.get("url") or "").hostname or ""
    return not host.endswith(NON_BR_TLDS)


@dataclass
class Source:
    name: str
    path: str
    config: str
    share: float  # fração do total (as frações somam 1)
    text_field: str = "text"
    data_files: list[str] | None = None
    columns: list[str] | None = None
    keep: Callable[[dict], bool] = lambda doc: True
    filters: list[str] = field(default_factory=list)


SOURCES = [
    Source(
        name="fineweb2-hq",
        path="epfml/FineWeb2-HQ",
        config="por_Latn",
        share=0.9,
        data_files=[
            f"hf://datasets/epfml/FineWeb2-HQ/por_Latn/000_{i:05d}.parquet"
            for i in (0, 30, 60, 90, 120, 150, 180, 210)
        ],
        columns=["text", "url", "quality_score"],
        keep=lambda d: not_other_lusophone(d) and looks_br(d),
        filters=[f"sem {', '.join(NON_BR_TLDS)}"],
    ),
    Source(
        name="wikipedia-pt",
        path="wikimedia/wikipedia",
        config="20231101.pt",
        keep=lambda d: looks_br(d),
        share=0.1,
    ),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-mb", type=float, default=50.0)
    p.add_argument("--val-mb", type=float, default=2.0)
    p.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="1 a cada N documentos aceitos vai para a validação",
    )
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="descarta documentos com quality_score abaixo disso (só fontes que têm o campo)",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    return p.parse_args()


def open_stream(src: Source, args):
    kwargs = {"split": "train", "streaming": True}
    if src.columns:
        kwargs["columns"] = src.columns
    try:
        if src.data_files:
            ds = load_dataset("parquet", data_files=src.data_files, **kwargs)
        else:
            ds = load_dataset(src.path, src.config, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("columns", None)
        if src.data_files:
            ds = load_dataset("parquet", data_files=src.data_files, **kwargs)
        else:
            ds = load_dataset(src.path, src.config, **kwargs)
        print(f"[aviso] {src.name}: não deu para selecionar colunas; baixando todas")
    return ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)


def sample_source(src: Source, targets: dict, files: dict, args) -> dict:
    written = {"train": 0, "val": 0}
    stats = {"train": 0, "val": 0, "short": 0, "filtered": 0, "low_quality": 0}
    accepted = 0

    for doc in open_stream(src, args):
        if not src.keep(doc):
            stats["filtered"] += 1
            continue
        if (
            args.min_quality is not None
            and "quality_score" in doc
            and doc["quality_score"] < args.min_quality
        ):
            stats["low_quality"] += 1
            continue
        text = clean(doc[src.text_field])
        if len(text) < args.min_chars:
            stats["short"] += 1
            continue

        split = "val" if accepted % args.val_every == 0 else "train"
        accepted += 1
        if written[split] >= targets[split]:
            split = "train" if split == "val" else "val"
            if written[split] >= targets[split]:
                break

        chunk = text + SEPARATOR
        files[split].write(chunk)
        written[split] += len(chunk.encode("utf-8"))
        stats[split] += 1

    stats["train_mb"] = written["train"] / 1024 / 1024
    stats["val_mb"] = written["val"] / 1024 / 1024
    return stats


def main():
    args = parse_args()
    total = sum(s.share for s in SOURCES)
    assert abs(total - 1.0) < 1e-6, f"as frações das fontes somam {total}, não 1"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        split: open(args.out_dir / f"{split}.txt", "w", encoding="utf-8")
        for split in ("train", "val")
    }
    report = []
    t0 = time.perf_counter()
    try:
        for src in SOURCES:
            targets = {
                "train": int(args.train_mb * src.share * 1024 * 1024),
                "val": int(args.val_mb * src.share * 1024 * 1024),
            }
            print(
                f"-> {src.name}: {targets['train'] / 1024 / 1024:.1f} MB treino, "
                f"{targets['val'] / 1024 / 1024:.2f} MB val"
            )
            stats = sample_source(src, targets, files, args)
            report.append((src, stats))
    finally:
        for f in files.values():
            f.close()

    print(
        f"\n{'fonte':14s} {'treino':>9s} {'val':>8s} {'docs tr':>8s} {'docs val':>8s} "
        f"{'curtos':>7s} {'filtr.':>7s} {'qual.':>6s}"
    )
    for src, s in report:
        print(
            f"{src.name:14s} {s['train_mb']:7.1f}MB {s['val_mb']:6.2f}MB {s['train']:8d} {s['val']:8d} "
            f"{s['short']:7d} {s['filtered']:7d} {s['low_quality']:6d}"
        )
    print(f"\ntempo: {time.perf_counter() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
