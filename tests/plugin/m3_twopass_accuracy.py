#!/usr/bin/env python3
"""Two-pass gsm8k accuracy probe for the M3 LMCache byte-offload load path.

Method follows zejun's K3 LMCache-offload report (2026-09-04):

  A single pass against an offload server only ever SAVEs -- it never LOADs, so
  it is a vacuous test of the load path.  To force loads:

    1. Prefix-distinctness.  Prepend a unique "[[UNIQ <salt> qNNNNN]]" marker so
       every question's ENTIRE prefix (few-shot included) is distinct.  The
       distinct-KV footprint then blows past the HBM paged pool, pass-1 entries
       are evicted from HBM and offloaded, and pass-2 must reload them.
       Without this the shared few-shot prefix simply stays hot in HBM and the
       external tier is never asked for anything.
    2. Chunk gate.  LMCache stores only full chunks.  M3 runs chunk_size=128 and
       OFFLOAD_MIN_LOAD_TOKENS=256, so a 5-shot prompt (~1k tokens) clears both.
    3. Pass-2 re-sends the identical prompts.  Under greedy decoding pass-2
       should reproduce pass-1 iff the reloaded bytes are correct.

  Aggregate delta_acc is the trustworthy signal.  Token-identity is NOT: this
  build is not bit-reproducible across a prefix-cache hit even with the offload
  tier off, so identical_frac is contaminated and is reported for information
  only.  The noise band must come from the OFF arm's own two-pass delta, never
  from the per-run sampling stderr.

Per-pass external-cache counter deltas are captured, because the whole point is
to prove the load path actually fired.  A pass-2 with zero external hits is a
vacuous run and its delta means nothing.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request

ANS = re.compile(r"####\s*([\-0-9\.\,]+)")
NUM = re.compile(r"(-?[0-9][0-9,]*\.?[0-9]*)")


def gold(ans):
    m = ANS.search(ans)
    return m.group(1).strip().replace(",", "") if m else None


def pred(text):
    # last number in the continuation, gsm8k flexible-extract convention
    ms = NUM.findall(text.replace(",", ""))
    return ms[-1].rstrip(".") if ms else None


def metrics(port):
    """Cumulative prefix-cache counters, so a pass can prove it loaded."""
    try:
        raw = (
            urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10)
            .read()
            .decode()
        )
    except (OSError, ValueError) as e:
        return {"error": str(e)}
    out = {}
    for key in (
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "external_prefix_cache_queries_total",
        "external_prefix_cache_hits_total",
    ):
        m = re.search(rf"^vllm:{key}\{{[^}}]*}} ([0-9.e+]+)$", raw, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out


def build(shots, q, salt, idx):
    """One self-contained user turn: unique salt, then the few-shot block."""
    head = f"[[UNIQ {salt} q{idx:05d}]]\n"
    body = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots
    )
    return head + body + f"Question: {q}\nAnswer:"


def ask(port, prompt, maxtok):
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "minimax-m3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": maxtok,
                "seed": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read().decode())
            u = r.get("usage", {})
            return {
                "text": r["choices"][0]["message"]["content"],
                "finish": r["choices"][0].get("finish_reason"),
                "cached": (u.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
                "ptok": u.get("prompt_tokens", 0),
            }
        except (OSError, ValueError, KeyError, IndexError) as e:
            if attempt == 2:
                return {
                    "text": "",
                    "finish": "ERROR",
                    "err": str(e),
                    "cached": 0,
                    "ptok": 0,
                }
            time.sleep(2)


def run_pass(port, prompts, maxtok, conc, tag):
    before = metrics(port)
    t0 = time.time()
    out = [None] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(ask, port, p, maxtok): i for i, p in enumerate(prompts)}
        for done, f in enumerate(cf.as_completed(futs), start=1):
            out[futs[f]] = f.result()
            if done % 200 == 0:
                print(
                    f"  [{tag}] {done}/{len(prompts)}  {time.time()-t0:.0f}s",
                    flush=True,
                )
    after = metrics(port)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after if k in before}
    return out, delta, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--maxtok", type=int, default=320)
    ap.add_argument("--out", default="/workspace/tmp/m3lmc/twopass")
    a = ap.parse_args()

    from datasets import Dataset

    root = (
        "/root/hf_cache/datasets/openai___gsm8k/main/0.0.0/"
        "740312add88f781978c0658806c59bc2815b9866"
    )
    test = Dataset.from_file(f"{root}/gsm8k-test.arrow")
    train = Dataset.from_file(f"{root}/gsm8k-train.arrow")
    shots = [train[i] for i in range(a.shots)]
    items = [test[i] for i in range(min(a.n, len(test)))]

    prompts = [build(shots, it["question"], a.salt, i) for i, it in enumerate(items)]
    golds = [gold(it["answer"]) for it in items]
    print(
        f"[{a.salt}] port={a.port} n={len(prompts)} shots={a.shots} conc={a.conc}",
        flush=True,
    )
    print(
        f"[{a.salt}] prompt chars: min={min(map(len,prompts))} max={max(map(len,prompts))}",
        flush=True,
    )

    res = {}
    for p in (1, 2):
        outs, delta, dur = run_pass(a.port, prompts, a.maxtok, a.conc, f"{a.salt}p{p}")
        ok = sum(1 for o, g in zip(outs, golds) if g and pred(o["text"]) == g)
        err = sum(1 for o in outs if o["finish"] == "ERROR")
        res[f"pass{p}"] = {
            "exact_match": ok / len(outs),
            "correct": ok,
            "n": len(outs),
            "errors": err,
            "seconds": round(dur, 1),
            "cached_tokens": sum(o["cached"] for o in outs),
            "prompt_tokens": sum(o["ptok"] for o in outs),
            "counter_delta": delta,
        }
        res[f"_texts{p}"] = [o["text"] for o in outs]
        c = res[f"pass{p}"]
        print(
            f"[{a.salt}] pass{p}: acc={c['exact_match']:.4f} err={err} {dur:.0f}s "
            f"cached={c['cached_tokens']}/{c['prompt_tokens']} "
            f"extHits={delta.get('external_prefix_cache_hits_total',0):.0f}",
            flush=True,
        )

    t1, t2 = res.pop("_texts1"), res.pop("_texts2")
    pairs = [(x, y) for x, y in zip(t1, t2) if x and y]
    res["identical_frac"] = (
        (sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs else None
    )
    res["delta_acc"] = res["pass2"]["exact_match"] - res["pass1"]["exact_match"]
    res["config"] = vars(a)

    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/summary_{a.salt}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(
        f"[{a.salt}] delta_acc={res['delta_acc']:+.4f} "
        f"identical_frac={res['identical_frac']:.3f} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
