# FOLD-PICCS-MIXING-BOUND — finite paper alpha/gamma root bound

```text
property_id: FOLD-PICCS-MIXING-BOUND
claim:
  The existing MixingRootProbabilityContract for the causal paper PiCCS
  experiment is constructed from finite root counting. It is not a premise of
  the concrete theorem.
assumptions:
  - The paper field/ring laws, including NoZeroDivisors for the selected
    extension-field operations.
  - One explicit nonempty duplicate-free finite scalar support.
non_goals:
  - The opaque production Split-NC deriveCore schedule or its stored
    challengeSetSize.
  - Concrete Goldilocks primality or seven-nonresidue certificates.
  - Fiat-Shamir, Poseidon2, Rust, R1CS, IR, encoding, rows, or costs.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:58-93
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md:235-256
```

## Exact experiment

The verifier support is the paper transcript order:

```text
alpha word in S^ell
→ one shared gamma in S
→ later SumCheck challenge word in S^ell
```

The full adversary experiment is the independent product of the prover tape,
post-prefix target tape, and this verifier tape. The target witness is fixed
outside the fresh second execution. No FE/NC strategy-independence or second
gamma is introduced.

`verifierAlphaGamma_marginal` proves that the later SumCheck word cancels from
the alpha/gamma event. Two applications of the finite product marginal prove
that the prover and target tapes cancel as well.

## Root counting

Coefficient nontruth selects one verifier-independent false obligation:

- a nonzero CCS or norm Boolean table; or
- a nonzero carried-evaluation scalar.

For a table controller, `MultilinearRootCounting.zeros_count_le` proves the
direct Cartesian MLE bound

```text
bad alpha words ≤ ell * |S|^(ell - 1).
```

At every remaining alpha, `CoefficientRootCounting.roots_count_le_degree`
uses constant-first synthetic division and the actual nonzero coefficient
list to prove

```text
bad gamma values ≤ jointCoefficientCount - 1.
```

The resulting exact finite count is

```text
(ell + jointCoefficientCount - 1) * |S|^ell
```

inside a support of size `|S|^(ell+1)`, hence

```text
Pr[MixingRoot] ≤
  (ell + jointCoefficientCount - 1) / |S|.
```

This is a direct instantiated bound for the repository's signed coefficient
object. It does not silently assume that a coefficient-nonzero list defines a
nonzero function; that implication is proved from synthetic division and
`NoZeroDivisors`.

## Exact event and composition

`SecurityContracts.mixingRootEvent_eq_true_iff` transports the Boolean event
to the exact named `MixingFailure`. If coefficient truth already holds, that
event has probability zero. Otherwise,
`mixingRoot_probability_le` transports the alpha/gamma count through the
complete causal experiment.

The headline theorem is:

```lean
MixingSoundness.mixingRootProbabilityContract_of_rootCounting
```

It returns the existing `MixingRootProbabilityContract`; that contract is not
an input. `RootCountingSecurity.fixedFirstBadBound_of_rootCounting` combines
it with the independently constructed SumCheck contract, and
`extraction_after_first_success_of_rootCounting` preserves the literal loss
order:

```text
(mixing + SumCheck) + rawMismatch / successFloor.
```

Neither algebraic probability contract is a premise of that final theorem.

## Production boundary

The production Split-NC obstruction remains valid. Its current `deriveCore`
interface exposes no internal finite support or sampling order and stores an
unrelated `challengeSetSize : Nat`. This property therefore closes the
paper/model experiment only. A later refinement must construct the production
challenge carrier and prove that it erases to the production transcript before
using this probability result for Rust or Poseidon2.

```text
conformance_status:
  model-proved (2026-07-25)
retest_commands:
  - cd formal/nightstream-lean &&
      LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsPaperJointMixingSoundness
      ./scripts/validate.sh build
  - cd formal/nightstream-lean &&
      LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsPaperJointMixingSoundness
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && ./scripts/validate.sh axioms
```
