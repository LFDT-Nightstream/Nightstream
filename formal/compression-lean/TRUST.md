# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems of `CompressionLean/Claims.lean`, and `CompressionLeanTests/Axioms.lean` pins their
axioms. `lake build` checks that every theorem of the library uses only `propext`, `Classical.choice`
and `Quot.sound`. The README states, per layer, what each part assumes; this file lists the layers in
one place.

## 1. The Lean kernel and the build cache

The toolchain in `lean-toolchain`. No proof may use `native_decide` (the axiom audit rejects
`Lean.ofReduceBool`), so no proof trusts the compiled evaluator. The executables of section 5 do: they
link the claims to Rust's artifacts. The `.olean` files of Mathlib and of
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

## 4. Premises

A claim that takes a premise says nothing about a key, a package or a setup that does not meet it.
`STATUS.md` lists the named premises of each claim and how each one is met. `ProgramEncodes`
(`Final/Claim.lean`) is not one of them: `Final/Encodes.lean` proves it for the production final
program, so `production_bound` does not take it. These remain open:

- `RunsGive` (`Final/RustSetup.lean`): the runs that Rust's `PackageRows::visit_rows` pushes give
  F′'s matrices. With the port of Rust's setup build, it gives `HonestSetup`
  (`honest_setup_of_build`).
- The facts of the setup that Rust builds (`Setup.Fits`, `Structure.Describes`).
- `ParentShape` for an application other than the Poseidon2 hash chain. For that application,
  `KeyFacts` and `WidthFits` give it (`poseidon2_bound`).

## 5. The ports

Lean transcribes parts of Rust by hand: the replay of the layer-1 verifier (`Replay/`), the circuit
programs and their lowering (`Circuit/`), the terminal loader (`Package.load`), and the setup build
and key expander (`Final/RustSetup.lean`). A port equals its Rust only as far as these checks reach:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- `ports.json` names the Rust item that each port transcribes, with a hash of its tokens. The CI test
  `lean_ports` fails when a ported Rust item changes, until someone re-reads the port and updates the
  hash. It detects a change; it does not show that the port is right. `lake build` fails when a
  declaration that cites Rust has no entry.
- Ignored Rust tests compare the replay's draws and verdicts, and the programs' instructions, with
  Rust on fixed inputs. They run by hand, not in CI (`scripts/crosschecks.sh`).
- The executables check the production artifacts against premises of the claims. `terminalCheck`
  refuses a key without `KeyFacts`, and a terminal package that does not load or whose compact rows
  name a missing template. `lowerCheck` also reads the toy key of `lowering_matches_lean`, which has
  neither production's constants nor an application's width, so it prints whether the key has
  `KeyFacts` and fits the Poseidon2 hash chain (`WidthFits`), and both executables print the width
  result. The production tests `production_layer1_matches_lean` and `final_program_matches_lean`
  require each printed pass. `key_facts_of_check`, `width_fits_of_check` and
  `package_templates_of_check` prove that a pass gives the premise; that the executables ran on the
  production artifacts is the evidence of these Rust tests.

## 6. Scope

The final theorems hold for the relation of an application of any logical width. The terminal package
does not depend on the width: F′'s row builders take it as a parameter but do not use it, and `rfl`
lemmas (`arithmeticRows_at`, `permutationInvocations_at`) state this. `poseidon2_bound` covers the
production key: F′'s relation of the Poseidon2 hash-chain application, at 43,963,750 columns and
814,144 blocks. It still takes the open premises of section 4. The claims describe the native verifier
and the circuit that Rust records; they say nothing about code paths that Lean does not model.

## Not trusted

- The claims: they are proved.
- The Rust: it is the object being modeled.
- `.lake/`: build output.
