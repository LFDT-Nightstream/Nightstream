# Wasm circuit proofs in Rocq

This pilot proves straight-line `i32.add`/`i32.sub` trace refinement and selected
zero-test and local-variable steps against SpecTec's Wasm 2.0 semantics.
Field primality and memory-event consistency are assumed; the offline memory
argument is not proved.

## Build

Requires opam and the checked versions of Coq/Rocq 9.0.1, MathComp ssreflect
2.4.0, and RecordUpdate 0.3.6:

```sh
opam switch create nightstream-rocq 5.3.0
eval "$(opam env --switch=nightstream-rocq --set-switch)"
opam repository add rocq-released https://rocq-prover.org/opam/released
opam install coq.9.0.1 coq-mathcomp-ssreflect.2.4.0 coq-record-update.0.3.6
cd formal/wasm-rocq
make
```

For an existing switch, install those packages there and run `make` with that
switch active. SpecTec is not needed to build the checked-in `Wasm2.v`.

Run `make audit` (requires Python 3) to build and print the arithmetic refinement
theorem's type and assumptions. It fails if axiom names differ from `scripts/audit.expected`
or the report cannot be parsed. Review changes before editing that file; changes
to existing axioms' types are not detected.

## Update inputs

From the Nightstream repository root, regenerate the circuit export with:

```sh
cargo run --release -p neo-wasm --bin export_wasm_rocq
```

`Wasm2.v` is copied verbatim from
[`spectec/test-rocq/theories/wasm.v`](https://github.com/Wasm-DSL/spectec/blob/95c256c2c/spectec/test-rocq/theories/wasm.v)
at commit `95c256c2c` (`rocq-backend-proof-final`). It includes upstream proofs
in addition to generated definitions. Upstream lists `spectec/` under
[Apache-2.0](https://github.com/Wasm-DSL/spectec/blob/95c256c2c/LICENSE).
To update it from a local SpecTec clone, run from the Nightstream root:

```sh
git -C /path/to/spectec show 95c256c2c:spectec/test-rocq/theories/wasm.v > formal/wasm-rocq/Wasm2.v
```
