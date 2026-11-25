import argparse
import time
from pathlib import Path

from myownllm.tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--path", type=Path, required=True, help="train file")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output file prefix"
    )
    parser.add_argument(
        "--vocab-size", type=int, required=True, help="vocab size with no special"
    )
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text = args.path.read_text(encoding="utf-8")
    print(f"texto: {len(text.encode('utf-8')) / 1024 / 1024:.1f} MB", flush=True)

    tok = BPETokenizer()
    t0 = time.perf_counter()
    tok.train(text, args.vocab_size, verbose=args.verbose)
    dt = time.perf_counter() - t0

    tok.register_special_tokens(args.special)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.output))

    print(f"merges: {len(tok.merges)}, final vocab: {len(tok.vocab)}", flush=True)
    print(f"train time: {dt:.0f}s ({dt / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
