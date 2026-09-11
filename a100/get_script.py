#!/usr/bin/env python3
"""Obtain a runnable training script for a record, with optional A100 adaptations.

Most record directories ship no .py: the training script is embedded in the run
log itself. The log opens with a verbatim copy of the source that ran, between
two 100-character '=' lines, followed by the environment block and step lines.
This tool prefers a real .py and otherwise extracts the embedded source. Every
adaptation is an anchored textual edit that fails loudly if its anchor does not
match exactly the expected number of times, and the output must pass py_compile.

Usage:
  python a100/get_script.py RECORD_DIR OUT.py [--devbs N] [--no-compile] [--features]

--devbs N       rewrite the standard 'device_batch_size : int = 64' line to N.
                If the record lacks that exact line, prints DEVBS_PATTERN_MISSING
                with whatever device_batch_size line it does have, and copies the
                source unmodified (the sbatch surfaces the warning).
--no-compile    eager variant: comment out 'model = torch.compile(model)' and any
                '@torch.compile' decorators, to compare against the compiled run.
--features      print markers used for classification (fp8/fa3/flex, iterations,
                batch/seq settings, torch version from the log's env block).
"""
import argparse
import glob
import os
import py_compile
import re
import sys


def extract_embedded(text):
    lines = text.split("\n")
    bars = [i for i, l in enumerate(lines) if l.startswith("====") and len(l.strip()) >= 80]
    if len(bars) >= 2 and bars[0] < 5:
        code = "\n".join(lines[bars[0] + 1:bars[1]])
    elif lines and lines[0].lstrip().startswith("import") and bars:
        code = "\n".join(lines[:bars[0]])
    else:
        return None
    return code if "import" in code[:300] else None


def readme_linked_log(record_dir):
    """The dir can hold several logs (side runs, baselines). The record's own
    log is the one the README table links; prefer it over any size heuristic."""
    base = os.path.basename(os.path.normpath(record_dir))
    readme = os.path.join(os.path.normpath(record_dir), "..", "..", "..", "README.md")
    try:
        text = open(readme, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = re.search(rf"{re.escape(base)}/([\w.-]+\.(?:txt|log))", text)
    if m:
        cand = os.path.join(record_dir, m.group(1))
        if os.path.exists(cand):
            return cand
    return None


def find_source(record_dir, pinned_log=None):
    pys = sorted(glob.glob(os.path.join(record_dir, "*.py")))
    for p in pys:
        if os.path.basename(p).startswith("train"):
            return p, open(p, encoding="utf-8", errors="replace").read(), "py"
    if pys:
        return pys[0], open(pys[0], encoding="utf-8", errors="replace").read(), "py"
    txts = sorted(glob.glob(os.path.join(record_dir, "*.txt")), key=os.path.getsize, reverse=True)
    preferred = [os.path.join(record_dir, pinned_log)] if pinned_log else []
    linked = readme_linked_log(record_dir)
    if linked:
        preferred.append(linked)
    for t in preferred + txts:
        if not os.path.exists(t):
            sys.exit(f"FATAL: requested log {t} does not exist")
        code = extract_embedded(open(t, encoding="utf-8", errors="replace").read())
        if code:
            return t, code, "log"
    return None, None, None


def replace_counted(src, needle, repl, what, expected=1):
    n = src.count(needle)
    if n != expected:
        sys.exit(f"FATAL: anchor for {what} matched {n}x, expected {expected}; record needs manual handling")
    return src.replace(needle, repl)


def features_of(src, raw_log):
    out = []
    out.append("fp8=YES" if re.search(r"float8|e4m3|e5m2|scaled_mm", src) else "fp8=no")
    out.append("fa3=YES" if re.search(r"flash_attn", src) else "fa3=no")
    out.append("flex=YES" if "flex_attention" in src else "flex=no")
    m = re.search(r"num_iterations\s*[:=]\s*(?:int\s*=\s*)?(\d+)", src)
    out.append(f"iters={m.group(1) if m else '?'}")
    for name in ("device_batch_size", "batch_size", "sequence_length", "seq_len", "max_seq_len"):
        for m in re.finditer(rf"^\s*{name}\s*[:=][^\n#]*", src, re.M):
            out.append(m.group(0).strip().replace(" ", ""))
    if raw_log:
        m = re.search(r"Running pytorch ([^\s]+)", raw_log)
        if m:
            out.append(f"ran-on-torch={m.group(1)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("out")
    ap.add_argument("--devbs", type=int)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--log", help="pin a specific log filename inside the record dir")
    args = ap.parse_args()

    src_path, src, kind = find_source(args.record_dir, pinned_log=args.log)
    if src is None:
        sys.exit(f"FATAL: no .py and no log with embedded source in {args.record_dir}")

    if args.features:
        raw = open(src_path, encoding="utf-8", errors="replace").read() if kind == "log" else None
        print("FEATURES:", " ".join(features_of(src, raw)))

    if args.devbs is not None:
        needle = "device_batch_size : int = 64"
        if src.count(needle) == 1:
            src = src.replace(needle, f"device_batch_size : int = {args.devbs}")
        else:
            have = re.findall(r"^\s*device_batch_size[^\n]*", src, re.M)
            print(f"DEVBS_PATTERN_MISSING: standard line absent; found {have or 'nothing'}; leaving source unmodified")

    if args.no_compile:
        src = replace_counted(
            src, "model = torch.compile(model)",
            "# model = torch.compile(model)  # disabled for the eager run",
            "model compile call")
        n_dec = len(re.findall(r"^@torch\.compile\s*$", src, re.M))
        if n_dec:
            src = re.sub(r"^@torch\.compile\s*$",
                         "# @torch.compile  # disabled for the eager run",
                         src, flags=re.M)
        print(f"no-compile: model compile call commented, {n_dec} decorator(s) commented")

    with open(args.out, "w") as f:
        f.write(src)
    py_compile.compile(args.out, doraise=True)
    print(f"OK: wrote {args.out} (source={src_path}, kind={kind})")


if __name__ == "__main__":
    main()
