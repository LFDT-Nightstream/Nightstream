# Wasm circuit proofs in Rocq

`run_refinement` proves circuit traces against SpecTec's Wasm 2.0 semantics for
`i32.add/sub`, `local.get/set`, direct guest calls, and zero-result blocks with
`br_if` to any enclosing block. Blocks nest to any depth, calls nest to any depth,
and any supported instruction, calls included, may appear inside a block.
Conditions may be computed, with multiple branches per block; a branch site must
have only the condition above its target block's operands (no unwinding), as the
Rust front end also requires.
`Fragment.v` lists the supported instructions, one constructor each with its
restrictions, and states the branch-site rule (`codeBranches`). Parameters, results, locals and stack values are i32. The theorem
takes plain rows, refines them one at a time, and relates circuit and SpecTec states through
`CircuitConfig` (`ControlDecode.v`): a decoder reads the configuration from memory
at a static control shape, a code path per frame through its enclosing blocks.
The operand height below each label follows from the code's static stack effects;
every function in the store must pass SpecTec validation. A run may stop anywhere
in its base frame, including inside blocks, inside calls, or during a callee's
frame initialization; the final shape resumes the next run. The store is an input:
the circuit does not hold it. A run never halts. Padding, host rows, indirect calls,
loops, `if`, `br`/`br_table`, branches to the function's own label and returning out
of the base frame are excluded.
Selected zero-test steps are also proved.
Field primality and memory-event consistency are assumed; the offline memory
argument is not proved. The program ROM encoder covers this guest-only fragment;
its correspondence with Rust preprocessing is not yet proved.

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

Run `make audit` (requires Python 3) to build and print `run_refinement`
and its assumptions. It fails if axiom names differ from `scripts/audit.expected`
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
