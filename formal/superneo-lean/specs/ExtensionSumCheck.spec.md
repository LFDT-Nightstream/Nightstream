# ExtensionSumCheck

## Purpose

Provide the extension-field analogue of the Definition-6 sum-check protocol
surface needed by Nightstream opening convergence Phase 1.

This module owns the `SuperNeo.KExt` versions of:
- sum-check protocol objects,
- verifier-side acceptance predicates,
- paper-facing statement/transcript consistency,
- final-oracle linkage to extension-field MLE evaluation.

It is the natural successor to `ExtensionMLE`: the current base-field
`SuperNeo/SumCheck.lean` remains the canonical Section 4 closure for `F`,
while this module carries the same theorem-facing protocol boundary over the
quadratic extension carrier used by opening convergence.

## Target Formulas

For an instance with `rounds = ℓ`, claimed value `v ∈ K`, transcript
`(r_0, ..., r_{ℓ-1})`, and univariate round polynomials `p_0, ..., p_{ℓ-1}`:

```text
p_0(0) + p_0(1) = v
```

```text
p_{i+1}(0) + p_{i+1}(1) = p_i(r_i)
```

```text
deg(p_i) ≤ maxDegree
```

For a table witness `T : {0,1}^ℓ -> K`:

```text
Σ_{x ∈ {0,1}^ℓ} T(x) = v
```

and final-oracle consistency means:

```text
p_{ℓ-1}(r_{ℓ-1}) = mleByFoldingK(T, [r_0, ..., r_{ℓ-1}])
```

with the zero-round case reduced to the empty-challenge MLE identity.

## Explicit Type Signatures

| Lean symbol | Type | Role |
|---|---|---|
| `ExtensionSumCheckInstance` | structure | protocol parameters + claimed value in `KExt` |
| `ExtensionSumCheckTranscript` | structure | extension-field challenges + round polynomials |
| `extensionSumcheckEvalPoly` | `Array KExt -> KExt -> KExt` | univariate coefficient-array evaluation |
| `extensionSumcheckAcceptedCore` | `ExtensionSumCheckInstance -> ExtensionSumCheckTranscript -> Prop` | verifier scaffold acceptance |
| `ExtensionSumCheckStatement` | structure | paper-facing hypercube-sum statement |
| `extensionSumcheckFinalOracleConsistent` | `ExtensionSumCheckInstance -> ExtensionSumCheckStatement inst -> ExtensionSumCheckTranscript -> Prop` | final-oracle linkage to `mleByFoldingK` |
| `extensionSumcheckAcceptedForTable` | `ExtensionSumCheckInstance -> Array KExt -> ExtensionSumCheckTranscript -> Prop` | fixed-table verifier acceptance |

## Paper Anchors

Source: `./formal/superneo-lean/SuperNeo.pdf.md`

- Definition 6 (sum-check protocol), Section 4.
- Opening convergence Phase 1: the round challenges and claimed values live in
  the quadratic extension carrier `K`.

## Contract Surface

| Lean symbol | Kind | Guarantee |
|---|---|---|
| `extensionSumcheckRoundPolyShape_degreeLe` | theorem | normalized coefficient-array shape implies paper degree bound |
| `extensionSumcheckVerifierAccepted_of_accepted` | theorem | scaffold acceptance implies paper-facing verifier acceptance |
| `extensionSumcheckFinalOracleConsistent_iff_withTable` | theorem | statement-indexed and table-indexed final-oracle consistency are equivalent |
| `ExtensionSumCheckClaim.accepted` | theorem | packaged claim witnesses reconstruct core acceptance |
| `extensionSumcheckAccepted_of_acceptedForTable` | theorem | fixed-table acceptance implies verifier acceptance |
| `extensionSumcheckAcceptedClosed_of_acceptedForTable` | theorem | fixed-table acceptance plus hypercube-sum equality reconstructs the existential statement witness |

## Out of Scope

- Full extension-field soundness/completeness closure.
- Extension-field Schwartz-Zippel or Lund bound formalization.
- Instantiation into the proof-system-level `ProofSystem/SumCheck/General.lean`.
