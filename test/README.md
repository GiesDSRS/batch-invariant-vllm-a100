# Batch invariance Test

Sends the same long probe prompt at several concurrency levels and checks that every completion hashes identically. Test by running:
```sh
export BASE_URL=http://localhost:8000/v1
export API_KEY=EMPTY
uv run batch_invariance.py
```

The output should look like this:
```
rep0 batch   1: 100%|██████████| 1/1 [00:19<00:00, 19.2s/req, d161bcd77052]
rep0 batch   8: 100%|██████████| 8/8 [01:04<00:00,  8.1s/req, d161bcd77052]
rep0 batch  32: 100%|██████████| 32/32 [03:44<00:00,  7.0s/req, d161bcd77052]
```

Every bar should end with the same hash. Exits 0 when they agree, raises `AssertionError` (exit 1) when they do not, so it works in CI unchanged.

---

# Additional details on the test

This test looks into:

- **Load-bearing** This is what lets the test fail, and it is enforced by the shape of the requests sent.
    - [The probe must be long](#the-probe-must-be-long)
    - [Requests must retire at staggered times](#requests-must-retire-at-staggered-times)
    - [An empty completion must not count as a pass](#an-empty-completion-must-not-count-as-a-pass)

- **Interpretation** This is how much a green run is worth, and it is settled by what it is compared against.
    - [Make it go red](#make-it-go-red)
    - [What a pass does and does not prove](#what-a-pass-does-and-does-not-prove)
    - [Compare against a saved file](#compare-against-a-saved-file)





## Load-bearing

### The probe must be long

The checked request is `PROBE_WORDS` long, ~9,000 words or roughly 14k tokens.

Short prompts are reproducible on broken configurations. Our first determinism check used ~20-token prompts, passed cleanly, and meant nothing: the same config, given a ~12,000-token prompt, returned different answers for the same input in the same batch.

> **Note:** upstream vLLM's own `tests/v1/determinism/test_batch_invariance.py` has this shape. The *checked* request is the needle, `"There once was a "`, seven characters. Only the filler requests are long, so a long checked request is never tested at all.


### Requests must retire at staggered times

Filler requests get varied `max_tokens`, 400 to 1,450, so they drop out underneath the probe while the probe is still generating.

This is subtler, and it was what caught the async-scheduling non-documented effect. If every request in the batch has the same `max_tokens` they all decode the same number of steps and **retire together**, so decode batch composition is constant for the whole run.

The condition that breaks is composition **changing mid-decode**, which is what real traffic would do when outputs range from 100 to 2,000 tokens and the batch churns 10 → 9 → 8 → …

> **Note:** upstream's test pins `max_tokens = 128` for every request, so it never produces this.



### An empty completion must not count as a pass

The script hashes `reasoning_content + content` and asserts the probe came back non-empty.

Reasoning models make this trap easy to fall into. `gpt-oss` puts its chain of thought in `reasoning_content`, and if `max_tokens` runs out before it finishes reasoning, `content` comes back **empty**. Hash that and every concurrency level agrees perfectly, on the sha256 of the empty string: e3b0c44298fc…; if you ever see it, the test measured nothing. Including the reasoning trace makes the check strictly stronger since it is part of the deterministic output.

> **Note:** we shipped this bug into our own test and it reported a confident PASS on zero bytes. Only the **probe** is held to this standard; filler requests that exhaust their budget have still done their job, which is to occupy a slot and then retire.



---


## Interpretation

### Make it go red

To confirm that this test works, run it first **without** `--no-async-scheduling`.

If it passes without that flag, it is likely that the workload is not stressing the scheduler. Raise `BATCHES`, lengthen `PROBE_WORDS`, or widen the filler `max_tokens` spread.



### What a pass does and does not prove

The test is made to prove that for this prompt, on this server, at these concurrency levels, output does not depend on batch composition.

Though, this does not prove invariance across a different vLLM version, different hardware, different flags, or eager versus CUDA graph mode. In particular: **graph numerics differ from eager numerics**. Each is internally reproducible, but they are not identical to each other... pick one mode and do not mix results from both within a study (or further testing).


### Compare against a saved file

Not enforced by this script, but worth noting. If qualifying config changes, dump per-item hashes to a file and diff files. Wall-clock on a single A100 carried roughly **±8% run-to-run variance**.
