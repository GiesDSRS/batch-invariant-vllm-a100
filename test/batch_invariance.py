#!/usr/bin/env python3
"""Same prompt, different batch sizes. Every answer must hash the same.

    pip install openai tqdm
    BASE_URL=... API_KEY=... python batch_invariance.py

Two things are load-bearing: the probe is long, and the fillers retire at
staggered steps. Drop either and this passes on a server that is not invariant.
"""

import concurrent.futures as cf
import hashlib
import os

from openai import OpenAI
from tqdm import tqdm

## Variables
MODEL = "openai/gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")  # ignored unless the server was started with --api-key
BATCHES = (1, 8, 16, 32, 64)                  # 32 matches --max-num-seqs, which saturates one A100
REPEATS = 3
PROBE_WORDS = 9000                            # keep it LONG, short prompts not useful for this test
FILLER_WORDS = 3000
MAX_TOKENS = 600
TIMEOUT = 3600                                # a queued request can wait a long time at high concurrency


client = OpenAI(base_url=BASE_URL,
                api_key=API_KEY,
                timeout=TIMEOUT)


# Synthetic prompts, should be identical on every machine and every run
WORDS = (
    "market revenue guidance segment margin pipeline capacity demand headwind "
    "backlog utilization cadence inventory logistics currency hedging"
).split()




## Functions
def text(n, shift=0):
    return " ".join(WORDS[(i + shift) % len(WORDS)] for i in range(n))


def answer(prompt, limit):

    m = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=limit,
    ).choices[0].message

    # Reasoning models keep the trace in reasoning_content; hashing only
    # content can silently hash the empty string.
    
    return (getattr(m, "reasoning_content", "") or "") + (m.content or "")


def probe_hash(batch, rep):

    jobs = [(PROBE, MAX_TOKENS)]
    jobs += [(text(FILLER_WORDS, i), 400 + i % 8 * 150) for i in range(batch - 1)]

    with cf.ThreadPoolExecutor(batch) as pool, \
         tqdm(total=batch, desc=f"rep{rep} batch {batch:>3}", unit="req") as bar:

        futures = [pool.submit(answer, *job) for job in jobs]
        for _ in cf.as_completed(futures):
            bar.update()

        probe = futures[0].result()
        assert probe.strip(), "empty probe -- raise MAX_TOKENS"
        digest = hashlib.sha256(probe.encode()).hexdigest()[:12]
        bar.set_postfix_str(digest)

    return digest




## Run
PROBE = "Summarize this in detail, listing every theme you find.\n\n" + text(PROBE_WORDS)
hashes = set()

for rep in range(REPEATS):
    for batch in BATCHES:
        hashes.add(probe_hash(batch, rep))

assert len(hashes) == 1, f"not batch invariant: {len(hashes)} distinct outputs"


## Results would look something like this:
# rep0 batch   1: 100%|██████████| 1/1 [00:18<00:00, 18.41s/req, 081d9f8ea5f8]
# rep0 batch   8: 100%|██████████| 8/8 [00:39<00:00,  4.98s/req, 081d9f8ea5f8]
# rep0 batch  16: 100%|██████████| 16/16 [00:49<00:00,  3.12s/req, 081d9f8ea5f8]
# rep0 batch  32: 100%|██████████| 32/32 [01:01<00:00,  1.92s/req, 081d9f8ea5f8]
# rep0 batch  64: 100%|██████████| 64/64 [01:51<00:00,  1.75s/req, 081d9f8ea5f8]
# rep1 batch   1: 100%|██████████| 1/1 [00:17<00:00, 17.94s/req, 081d9f8ea5f8]
# rep1 batch   8: 100%|██████████| 8/8 [00:39<00:00,  4.99s/req, 081d9f8ea5f8]
# rep1 batch  16: 100%|██████████| 16/16 [00:50<00:00,  3.13s/req, 081d9f8ea5f8]
# rep1 batch  32: 100%|██████████| 32/32 [01:02<00:00,  1.96s/req, 081d9f8ea5f8]
# rep1 batch  64: 100%|██████████| 64/64 [01:48<00:00,  1.69s/req, 081d9f8ea5f8]
# rep2 batch   1: 100%|██████████| 1/1 [00:18<00:00, 18.25s/req, 081d9f8ea5f8]
# rep2 batch   8: 100%|██████████| 8/8 [00:41<00:00,  5.16s/req, 081d9f8ea5f8]
# rep2 batch  16: 100%|██████████| 16/16 [00:52<00:00,  3.25s/req, 081d9f8ea5f8]
# rep2 batch  32: 100%|██████████| 32/32 [01:01<00:00,  1.93s/req, 081d9f8ea5f8]
# rep2 batch  64: 100%|██████████| 64/64 [01:46<00:00,  1.66s/req, 081d9f8ea5f8]