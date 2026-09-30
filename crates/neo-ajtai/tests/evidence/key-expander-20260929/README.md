# Ajtai key expander cost

This evidence measures the cost of one complete production key pass with four
byte sources. It supports the expander choice in
`docs/reviews/ajtai-key-expander/SECURITY_ARGUMENT.md`. It does not change the
production setup.

## What is measured

One key pass expands every key element of the production prefix: 22 rows by
3,221,095 columns, which is 70,864,090 elements and 3,826,660,860
coefficients. A CPU `extend` expands the key twice (PiDEC children and fresh
carrier). Terminal verification expands it once.

Each element gives 54 coefficients. Every expander reduces 32 little-endian
bytes modulo the Goldilocks prime with the same arithmetic. Only the byte
source differs:

| Expander | Bytes for element (row, column) | Primitive calls |
|---|---|---:|
| `ChaCha20` (replaced) | Lane L: first 32 bytes of ChaCha20 block L, nonce `row ‖ column` | 54 blocks |
| `ChaCha20Contiguous` | Lane L: keystream bytes 32L to 32L+31, same nonce | 27 blocks |
| `Shake128` | SHAKE128(`tag ‖ seed ‖ row ‖ column`), bytes 32L to 32L+31 | 11 permutations |
| `Shake256` | SHAKE256(same input), bytes 32L to 32L+31 | 13 permutations |

The tag is `nightstream-ajtai-shake-bench-v0` (32 bytes). It is a benchmark
value, not a proposed protocol constant.

The production setup now uses SHAKE128, so `bench/` keeps only the two SHAKE
expanders. The ChaCha20 rows are recorded results. The program at commit
`9353dbb17` measured them; its CPU `ChaCha20` path was the production
`coefficient_block` at 5bd5873f4. Each pass folds every coefficient into an
order-independent checksum. The program stops if the CPU and Metal checksums
differ. Before timing, it compares scattered elements with a division-based
reduction.

## Commands

The crate has its own workspace, so it does not change the production crates
or the root lockfile.

```bash
cd crates/neo-ajtai/tests/evidence/key-expander-20260929/bench
cargo build --release
cargo build --release --features asm --target-dir target/asm
timeout --signal=KILL 300 target/release/key-expander-bench both
timeout --signal=KILL 300 target/asm/release/key-expander-bench cpu
```

The `asm` feature uses the ARMv8 SHA3 instructions through `sha3` 0.10.8.
The release profile copies the production workspace profile.

## Machine

Apple M5 Max (6 + 12 CPU cores, 40 GPU cores, 128 GB), macOS 26.5.1,
rustc 1.94.1, 18 rayon threads. Other work also ran on the machine, so the
CPU passes ran as interleaved rounds. The tables show the median of three
rounds. The logs contain every run.

## Results

CPU, one key pass:

| Expander | Portable Keccak | ARMv8 SHA3 instructions |
|---|---:|---:|
| `ChaCha20` (replaced) | 7.46 s | 7.11 s |
| `ChaCha20Contiguous` | 4.92 s | 4.51 s |
| `Shake128` | 9.99 s | 7.50 s |
| `Shake256` | 11.35 s | 8.67 s |

The ChaCha20 rows do not use Keccak. Their difference between the two columns
shows the run-to-run variation.

Metal, one key pass:

| Expander | Time |
|---|---:|
| `ChaCha20` (replaced) | 0.604 s |
| `ChaCha20Contiguous` | 0.322 s |
| `Shake128` | 1.360 s |
| `Shake256` | 1.607 s |

## Reading the results

- On CPU with the SHA3 instructions, SHAKE128 costs about the same as
  ChaCha20 (+5% in the same build). SHAKE256 costs about 22% more.
- On Metal, SHAKE128 costs 2.3 times ChaCha20 and SHAKE256 2.7
  times. Each Metal `extend` expands the key twice, so SHAKE128 adds about
  1.5 s to an `extend` of about 18 s.
- Contiguous ChaCha20 costs about 35% less on CPU and 47% less on Metal.

## Limits

- The Metal Keccak kernel is a direct implementation with array indexing and
  is not tuned. The Metal ChaCha20 kernels are also direct. A tuned Keccak
  kernel can reduce the Metal gap.
- The CPU SHAKE paths hash one element at a time. They do not interleave
  several Keccak states.
- The contiguous ChaCha20 path uses 27 lanes. The production path uses 54,
  which divides into full NEON vectors better.
- A key pass is part of a commitment. The signed accumulation of witness
  columns does not change with the expander and is not included.
