# CPU indexed-key reduction

The streamed CPU key loader now reduces each 256-bit coefficient without
128-bit division. Let `x = 2^32`. Goldilocks has `x^2 = x - 1` and `x^6 = 1`,
so the eight little-endian words reduce to `a + b*x`:

```text
a = w0 - w2 - w3 + w5 + w6
b = w1 + w2 - w4 - w5 + w7
```

Both signed sums fit in `i64`. The existing field operations finish the
reduction and return the canonical word. The scalar `coefficient` function
retains its separate ChaCha block implementation and eight remainder steps
as the reference. Seed, nonce addresses, stream generation, coefficient
distribution, key dimensions, and ring products remain unchanged.

## Production measurement

Both release runs use the same stored step-1 inputs, with one complete fresh
witness and sixteen zero running witnesses. They run serially under the
300-second test cap and the current 16 GiB RSS guard. `caffeinate -is` prevents
idle and system sleep for the command while on AC power.

| Measurement | Before | After |
|---|---:|---:|
| Fresh commitment | 95.144954 s | 58.379275583 s |
| Complete Sources phase | 130.097224667 s | 93.323147958 s |
| Peak process RSS | 6,073,974,784 bytes | 6,115,606,528 bytes |

The commitment takes **38.6% less time** (1.63× speed). These are individual
measurements. Preparation takes 34.63 s in both runs. Peak RSS covers the
whole Sources phase; it is not a full lifecycle memory measurement.

Each run recomputes and compares the entire fresh commitment and all running
commitments against the saved claims. All 19 input files are byte-identical,
and the source-check records match. See [the comparison](comparison.json).
This does not generate a new production folding proof or establish the full
CPU/Metal speed ratio.

## Checks

- 12 Ajtai tests pass: stored Lean setup values, RFC block values, all streamed
  lanes, generated seeds and addresses, maximum nonce words, independent ring
  products, batch ordering, and invalid-input rejection. The optional external
  fixture-path test remains ignored; the default stored fixture test ran.
- Three Metal commitment tests pass, including exact indexed coefficients and
  CPU/device commitments across tiles and witness representations.
- Nine supported engine tests pass, including parallel PaperExact/Optimized
  cross-checks, the selected polynomial, CPU/Metal parity, and error rejection.
  The two CUDA comparisons remain ignored because the kernel is absent.
- Release compilation, `cargo fmt --all`, and `git diff --check` pass.

All test and test-build invocations use the 300-second cap from `AGENTS.md`.
No Lean command, new Cargo feature, or environment variable was added. The
engine test log includes deliberate worker failure and invalid-opening cases;
their rejection tests pass.

The saved executables and source patch remain in the private run directory:
`/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h`.
The old executable is `cpu-terminal-profile-test-binary`; the new one is
`wide-reduction-test-binary`. Logs, requests, RSS records, and the exact-input
comparison are retained here.

The 5× full lifecycle target and the memory bound for all supported circuits
remain open. The dense application-table storage has not changed in this pass.
