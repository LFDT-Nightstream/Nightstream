# Authorized-amount proof benchmark

The fixture runs one create transaction and 15 warm spends. Each transaction produces two Wasm proofs and one interleaving proof. The ledger verifies every transaction with upstream Akita. The fixture compares the complete encoded spend with the first spend.

## Run on macOS

1. Use a disposable Starstream checkout at `ebc9f3c1c07815f06699749d07d478f9e39d15d2`. Set its five Nightstream workspace dependencies in `Cargo.toml` to `2a1a12d8905de35d73217b501f2773d33f8cc9ea`.
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

Each row is one fresh process on a 96 GiB M5 Ultra. Peak RSS covers the full process lifetime.

| Engine | Peak RSS | Median warm transaction |
|---|---:|---:|
| CPU | 15.545 GiB | 1.879 s |
| Metal with shared worker pools | 15.469 GiB | 1.382 s |

Both engines produced the original transaction bytes. Check the files in the output directory with `shasum -a 256`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `create.cbor` | 349,978 | `9b4fc016c2a4d115498a3ae6310b5b6cdfae5d683636e1828ea8f1bd51c3d4e9` |
| `spend.cbor` | 349,718 | `72e7cf7e83fe5a3f4d0e2b8d24f597c33508d6b266623a90d5ca9c431be7b3a3` |

## Cache-removal comparison

The controlled CPU comparison used the earlier Nightstream memory-branch revision `fa4d983a7d62c0283390008204e152bcddd6ce39` on the same Studio. Peak RSS fell from 24.86 GiB to 16.40 GiB. An allocation trace showed the 8.4375 GiB prepared transform was absent at the start of the first warm spend. In 15 pairs of adjacent transactions, enabling the portable coefficient fold added a median 131 ms (6.2%). The cached and cache-free builds used the same worker pools and produced identical bytes.

On a 64 GiB M1 Max, the earlier Metal qualification of the streaming transform and shared-worker schedule had a 3.243 s median across nine warm spends in three processes. Maximum process RSS was 16.058 GiB.
