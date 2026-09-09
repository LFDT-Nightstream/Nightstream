# Foundation completion goal

Status: the three selected Lean proof and local connection obligations are closed.

Source: working-tree changes based on `4fc02857c3aa4207f3290739cc62d794ba86f5f9`, validated on 2026-09-08. The main repository changes are not committed. The existing Stage 1 owner goal and architecture contract were not changed.

## Acceptance criteria and evidence

| Requested item | Closing evidence |
|---|---|
| `F.field.extension.irreducible` | [sevenProjectiveNonresidue](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/GoldilocksExtension.lean:30) supplies the exact premise for [extensionNoZeroDivisors](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/GoldilocksExtension.lean:49), using the existing arithmetic certificate and the current carriers. |
| `F.strong_set.cardinality` | [productionMember_cardinality](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Phi81StrongSet/Cardinality.lean:21) proves the exact image count; [productionMember_cardinality_bits](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Phi81StrongSet/Cardinality.lean:40) connects it to the 125-bit descriptor; [key_challengeSetSize_cardinality](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:215) connects it to the actual production key. |
| `F.profile.combined_bound` | [production_bigB_below_half_modulus](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Profile.lean:87) proves `2*B<q`; [key_bigB_below_half_modulus](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:225) proves the key uses that bound. [Existing ambient coverage](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiRLC/PaperCorrections.lean:124) is already used by [the extraction algebra](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PaperExtractionAlgebra.lean:1127). |

The field proof uses Mathlib's [Fermat theorem](https://leanprover-community.github.io/mathlib4_docs/Mathlib/FieldTheory/Finite/Basic.html#ZMod.pow_card_sub_one_eq_one), checked against the installed Mathlib v4.30.0 source. `ZMod q` reduces to the existing `Fin q` representation for this modulus. The proof adds no alternative field carrier.

The challenge count follows from the existing injective embedding of `Fin 54 → Fin 5`; it does not enumerate that set. The key's numerical count is unchanged: the new profile definition reduces to the same `5^54` value.

## Validation

All Lean commands used `scripts/validate.sh` with an outer `timeout -s KILL 1500`, as required by the repository's 25-minute cap. No Lean or Rust build ran concurrently.

- Baseline: boundary gate passed; library build passed in 4 seconds; axiom/test build passed in 1 second.
- Focused proof checks: extension module passed in 8 seconds; cardinality module passed in 1 second.
- First affected library rebuild: 3,644 jobs passed in 287 seconds. The new audit file then needed registration in the explicit test-root list; that registration was added.
- Final complete validation: boundary gate passed; library build passed in 1 second; axiom/test build passed in 4 seconds with 3,680 jobs.
- The [foundation axiom audit](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/tests/AxiomsFoundations.lean) is imported by the main audit. Every audited result uses only `propext`, `Classical.choice`, and `Quot.sound`, or no axioms.

Logs: [baseline](/tmp/nightstream-foundations-baseline.log), [field](/tmp/nightstream-foundation-field.log), [cardinality](/tmp/nightstream-foundation-cardinality.log), [first rebuild](/tmp/nightstream-foundations-validation.log), [final validation](/tmp/nightstream-foundations-final.log).

The selected profile remains `b=2`, `k_rho=16`, `B=65536`. No Rust implementation, transcript, matrix layout, serialized schema, or cryptographic assumption was changed. This proof-only slice did not run Rust conformance tests or a complete production proof. The follow-up below adds primitive conformance evidence.

Ajtai/MSIS hardness, commitment binding, low-norm invertibility, and sampling/probability assumptions remain separate. The user authorized these concrete conditions to be proved; this goal did not attempt to prove lattice hardness or finish HyperNova recursion.

The three website records now report local proof and local connection completion. The original review map remains a historical snapshot. Existing definition-only entries are not silently converted to proved theorems.

## Follow-up: two primitive conformance checks

Status: the two requested primitive comparisons pass on the same working tree. The runtime algorithms needed no change. This does not close a protocol phase or the production verifier.

The acceptance criteria were exact agreement with the active Lean transform, exact agreement with the active checked signed split under `b=2`, `k_rho=16`, `B=65536`, and agreement on the selected out-of-range cases.

| Item | Executed evidence |
|---|---|
| `F.transform.rust_connection` | All 54 × 54 entries from `nativeBarEntry` match `superneo_bar_matrix`. All 54 coefficient-basis images match the Rust block and vector functions. |
| `F.decomposition.canonical_sign` | All 131,071 distinct values from −65,535 through +65,535 match `splitScalarChecked` in all 16 digit positions. Matrix ordering, nonzero flags and the virtual-zero branch also pass. Both signs of `B`, `B+1`, and `floor(q/2)` are rejected. |

The [Lean emitter](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/FoundationParityMain.lean) calls the existing semantic definitions. It writes one row or scalar per JSON line. The outputs are 7,467 bytes for the transform and 16,438,527 bytes for the split; the emitter does not retain the complete output in memory. No generated Lean source or new theorem is needed.

The [repeatable check](/Users/nicarq/starstream/develop/nightstream-clean-up/scripts/check_fprime_foundation_parity.sh) generates fresh vectors in a temporary `.lake` directory, then runs the two Rust tests in sequence. It removes the vectors on exit. Each Lean command has the policy cap of 1,500 seconds; each Rust test has the policy cap of 300 seconds. The tests are explicitly selected by this script because ordinary `cargo test` does not supply fresh Lean output on stdin. No new Rust feature or environment setting was added.

The obsolete `phi81_bar_lean_artifact` test and its Cargo registration were replaced by [active_lean_bar_matches_runtime](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-math/tests/phi81_bar_lean_parity.rs:10). The old test could write into the frozen project on failure. The new path reads active Lean output and never uses that project. The signed-digit test is [active_lean_signed_binary_matches_runtime](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-reductions/tests/signed_binary_lean_parity.rs:44).

Validation on 2026-09-08:

- Existing transform regression tests: all five passed; compilation took 5.94 seconds.
- New Lean file check: passed in 4 seconds. The boundary gate passed with the new executable registered.
- First executable build and emission: 5 seconds. Warm emission: 1 second. There was no previous active-project emitter for this comparison.
- Fresh-output script: both Rust tests passed. Initial compilation took 3.46 seconds for the ring test and 9.75 seconds for the split test. Test execution took less than 0.01 and 0.05 seconds respectively.
- After adding the explicit virtual-zero case: both Rust comparisons passed. The signed test command took 4.31 seconds, including recompilation.
- Failure checks: changing matrix entry `(0,0)` from 1 to 0 failed at that exact entry. Changing the first digit of −1 to +1 failed at that exact value and digit. Both commands returned the expected test-failure exit code, 101.
- `cargo fmt --all` and `git diff --check` completed. The formatter reported only its existing stable-toolchain warning about `imports_granularity`.

Logs: [fresh-output script](/tmp/nightstream-foundation-parity.log), [warm emission](/tmp/nightstream-foundation-parity-warm.log), [ring comparison](/tmp/nightstream-foundation-bar.log), [signed comparison](/tmp/nightstream-foundation-split.log), [changed coefficient](/tmp/nightstream-foundation-bar-mutated.log), [changed digit](/tmp/nightstream-foundation-split-mutated.log).

The website now reports the transform connection as connected and both Rust checks as scoped tests passed. Foundations reads Proof 64/75, Link 77/84, Rust 68/70 under the existing counter rules. The proof count still includes 11 definition-only records in its denominator; these are not 11 missing theorems. Cryptographic assumptions and the full pilot/PiCCS/PiRLC/PiDEC integration obligations remain separate.
