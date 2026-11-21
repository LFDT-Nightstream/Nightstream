# Authorized-amount proof benchmark

The fixture runs one create transaction and 15 warm spends. Each transaction produces two Wasm proofs and one interleaving proof. The ledger verifies every transaction with upstream Akita. The fixture compares the complete encoded spend with the first spend.

## Run on macOS

1. Use a disposable Starstream checkout at `ebc9f3c1c07815f06699749d07d478f9e39d15d2`. Set its five Nightstream workspace dependencies in `Cargo.toml` to `e3b6b4073209c2f24934c5e6e7410bb65913a5fd`.
2. From the Starstream checkout, use `git apply` to apply each fixture patch in order. The first enables three concurrent proof jobs on Metal using the shared worker pools. The second saves complete transaction files and records phase times. [Shared workers](starstream-pr328-shared-metal-workers.patch) · [Timing and outputs](starstream-pr328-warm-spends.patch)
3. Build the release test with the command below. For Metal, first select `ProofEngine::Metal` in `authorized_amount_workflow` in `starstream-ledger/tests/proving.rs` and add `--features starstream-proving-runtime/metal` to the build command. The default uses `ProofEngine::Optimized`.

```sh
binary=$(cargo test -p starstream-ledger --release --test proving --no-run --message-format=json |
  jq -r 'select(.reason=="compiler-artifact" and .target.name=="proving" and .executable!=null) | .executable')
out=$(mktemp -d)
(cd "$out" && /usr/bin/time -l "$binary" authorized_amount_transactions_prove_and_verify --ignored --exact --nocapture)
```

Use the process maximum resident set size in bytes from `/usr/bin/time -l`. Divide by 1,073,741,824 for GiB. Discard the create transaction when calculating warm proving time. The next 15 `full warm clock ns` values include proving and compression for all three jobs and exclude verification.

## Studio check of the pinned integration

Each engine ran one fresh release process on a 96 GiB M5 Ultra. Peak RSS covers the full process lifetime.

| Engine | Peak RSS | Median warm transaction |
|---|---:|---:|
| CPU | 15.700 GiB | 1.879 s |
| Metal with shared worker pools | 15.432 GiB | 1.308 s |

## M1 Max validation

A 10-core M1 Max with 64 GiB of memory ran on AC power. CPU and Metal alternated across five fresh processes each. Each process made one create and 15 warm spends. The table gives the median of the five process medians and the highest process RSS and physical footprint. Both memory measures include setup and verification.

| Engine | Median warm transaction | Range of process medians | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|
| CPU | 6.981 s | 6.966–7.008 s | 15.960 GiB | 12.422 GiB |
| Metal with shared worker pools | 3.305 s | 3.251–3.340 s | 15.685 GiB | 12.484 GiB |

Metal ran the full warm transaction 2.11× as fast as CPU. macOS reported 0 process swaps across the 10 runs. The three jobs overlapped on every warm spend. The job times below include time spent waiting during concurrent execution.

| Proof | CPU proving | CPU compression | Metal proving | Metal compression |
|---|---:|---:|---:|---:|
| Coordinator Wasm | 2.143 s | 4.778 s | 0.430 s | 2.290 s |
| UTXO Wasm | 2.134 s | 4.777 s | 0.337 s | 2.263 s |
| Interleaving | 3.345 s | 3.142 s | 1.737 s | 1.453 s |

All 10 creates and 150 warm spends verified with upstream Akita at `d50f700f`. All saved create and spend files matched the original CPU files byte for byte. An independently built upstream receiver also accepted the M1 output. The prover used Akita at `5a2a2db2c554cdd4dde7dd2f303cb304ce5460be`.

A Wasm job finished last on all 75 CPU spends. On Metal, a Wasm job finished last on 43 spends and interleaving finished last on 32.

The M1 Max also ran the same five-process benchmark with the prior pin, Nightstream `4392899a` and Akita `22556f80`.

| Engine | Prior median | New median | Prior peak RSS | New peak RSS | Change in highest RSS |
|---|---:|---:|---:|---:|---:|
| CPU | 6.951 s | 6.981 s | 16.300 GiB | 15.960 GiB | -348 MiB |
| Metal | 3.287 s | 3.305 s | 16.041 GiB | 15.685 GiB | -365 MiB |

RSS varies between fresh processes. The warm transaction medians increased by 30 ms on CPU and 18 ms on Metal. Peak physical footprint changed from 12.860 to 12.422 GiB on CPU and from 12.427 to 12.484 GiB on Metal.

Check the files in the output directory with `shasum -a 256`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `create.cbor` | 349,978 | `9b4fc016c2a4d115498a3ae6310b5b6cdfae5d683636e1828ea8f1bd51c3d4e9` |
| `spend.cbor` | 349,718 | `72e7cf7e83fe5a3f4d0e2b8d24f597c33508d6b266623a90d5ca9c431be7b3a3` |

## Cache-removal comparison

The controlled CPU comparison used the earlier Nightstream memory-branch revision `fa4d983a7d62c0283390008204e152bcddd6ce39` on the same Studio. Peak RSS fell from 24.86 GiB to 16.40 GiB. An allocation trace showed the 8.4375 GiB prepared transform was absent at the start of the first warm spend. In 15 pairs of adjacent transactions, enabling the portable coefficient fold added a median 131 ms (6.2%). The cached and cache-free builds used the same worker pools and produced identical bytes.

On a 64 GiB M1 Max, the earlier Metal qualification of the streaming transform and shared-worker schedule had a 3.243 s median across nine warm spends in three processes. Maximum process RSS was 16.058 GiB.
