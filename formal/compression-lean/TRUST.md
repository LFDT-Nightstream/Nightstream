# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems that `CompressionLeanTests/Axioms.lean` pins. `lake build` checks that each one uses
only `propext`, `Classical.choice` and `Quot.sound`. The README states, per layer, what each part
assumes; this file lists the layers in one place.

## 1. The Lean kernel and the build cache

The toolchain in `lean-toolchain`. No proof may use `native_decide` (the axiom audit rejects
`Lean.ofReduceBool`), so the compiled evaluator is not trusted. The `.olean` files of Mathlib and of
`nightstream-fprime` come from a build cache that `lake build` does not re-check. Re-checking the
environment with `lean4checker`, or building both from source, removes this layer.

## 2. `nightstream-fprime`

This project imports F′'s definitions and theorems: the terminal package and its rows, the Poseidon2
specification, the Ajtai setup and the paper algebra. A claim here is only as good as F′'s, and
inherits F′'s own trusted layers.

## 3. Named assumptions

The claims bound a probability. They take these as part of the model, not as Lean axioms:

- **WHIR.** An opening is accepted only when one listed candidate agrees with every opened value
  (`Opens`), except with that opening's error. The error figures come from Plonky3's Johnson-bound
  report in Rust (`pcs.rs`) and enter `Budget.lean` as given. No Lean proof checks them. This is the
  weakest numeric layer.
- **Fiat–Shamir.** Poseidon2 acts as a random oracle, and the transcript encoding is injective on
  prefixes (`Restoration.lean`, `FiatShamir.lean`).
- **The read condition.** Inside the final program, the hash acts as a random oracle and the
  openings follow the WHIR model (`Final/Soundness.lean`).
- **Merkle binding.** A setup leaf that reaches the key root and disagrees with the setup gives a
  Poseidon2 collision (`MerkleBinds`).

## 4. Open premises

Two premises of the final theorems are hypotheses, not theorems, until they are proved:
`ProgramEncodes` (`Final/Claim.lean`) and `RunsGive` (`Final/RustSetup.lean`). A claim that takes
one says nothing about a program or a setup that does not meet it.

## 5. The ports

Lean transcribes parts of Rust by hand: the replay of the layer-1 verifier (`Replay/`), the circuit
programs and their lowering (`Circuit/`), the terminal loader (`Package.load`), and the setup build
and key expander (`Final/RustSetup.lean`). A port equals its Rust only as far as these checks reach:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- Ignored Rust tests compare the replay's draws and verdicts, and the programs' instructions, with
  Rust on fixed inputs. They run by hand, not in CI.

## 6. Scope

The final theorems are stated for F′'s reference relation (`Data.logicalWidth`). They describe the
native verifier and the circuit that Rust records; they say nothing about code paths that Lean does
not model.

## Not trusted

- The claims: they are proved.
- The Rust: it is the object being modeled.
- `.lake/`: build output.
