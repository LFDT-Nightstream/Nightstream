# neo-midnight-bridge

Experimental bridge: prove Neo FoldRun validity using Midnight's PLONK/KZG verifier stack.

## KZG params (BLS12-381 SRS)

For local tests/benchmarks, this crate can load `midnight-proofs` `ParamsKZG` files from:

`crates/neo-midnight-bridge/testdata/kzg_params/bls_midnight_2p{k}`

### Download pre-generated Midnight params

Midnight publishes pre-generated parameter files under:

`https://midnight-s3-fileshare-dev-eu-west-1.s3.eu-west-1.amazonaws.com/bls_midnight_2p{k}`

Example downloads:

```bash
BASE_URL="https://midnight-s3-fileshare-dev-eu-west-1.s3.eu-west-1.amazonaws.com"
OUT_DIR="crates/neo-midnight-bridge/testdata/kzg_params"

mkdir -p "$OUT_DIR"

# Download k=16,17,18
for k in 16 17 18; do
  curl -L --fail -o "$OUT_DIR/bls_midnight_2p${k}" "$BASE_URL/bls_midnight_2p${k}"
done
```

Notes:
- Files can be large (e.g. `k=21` is ~400MB).
- Midnight-ledger contains SHA-256 hashes for each `bls_midnight_2p{k}` for integrity checking.

### Generate params from a Powers-of-Tau transcript

If you have a Midnight Powers-of-Tau transcript file (raw bytes: G1 powers followed by two G2
points), you can convert it into `ParamsKZG` files via:

```bash
cargo run -p neo-midnight-bridge --example params_from_powers_of_tau -- \
  <powers_of_tau_path> crates/neo-midnight-bridge/testdata/kzg_params [k_max]
```

