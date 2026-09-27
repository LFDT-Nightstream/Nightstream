# Getting started

Use the Rust version pinned in `rust-toolchain.toml`. Build and test with:

```sh
cargo build -p nightstream --release
timeout --signal=KILL 300 cargo test -p nightstream --release
```

The [current API guide](../crates/nightstream/README.md) shows application
compilation, package saving and loading, proving, extension, and verification.
The caller selects an explicit minimum statistical-security level.

Start reading `crates/nightstream/src/circuit.rs`, `application/`, `assembly/`,
`folding/`, and `lifecycle/`. The shared sealed-package consumer is in
`crates/nightstream-fprime/src/package/`. The active formal authority is
`formal/nightstream-fprime`.

Follow [AGENTS.md](../AGENTS.md): release tests, a five-minute non-Lean test
cap, `FoldingMode::Optimized`, and `cargo fmt --all` after Rust edits.
See [golden conformance](../scripts/GOLDEN_CONFORMANCE.md) for fresh two-fold
Rust/Lean checks.
