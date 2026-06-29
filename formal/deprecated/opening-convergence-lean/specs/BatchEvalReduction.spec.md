# BatchEvalReduction Specification

## Purpose

Prove that Phase 1 (point unification via eta/gamma-linearization +
rho-batched sumcheck + u_i* scalar outputs) is a sound batch evaluation
reduction: if the verifier accepts, then with high probability all unified
claims are true evaluation claims at the verifier-derived point r*.

## Target Formulas

### Theorem 3: ClaimedSumCorrectness

For batched polynomial P(x) = sum_i rho^i * eq(r_i, x) * g_i(x):

```
sum_{x in {0,1}^ell} P(x) = sum_i rho^i * g_i(r_i) = sum_i rho^i * u_i = T
```

This is verifier-computable before the sumcheck begins.

### Theorem 4: CoefficientLinearization

Given eta sampled uniformly from K and packed column j with AJTAI_D
coefficient polynomials:

```
w_i*^{(j)} = sum_t eta^t * v_i*[j].coeffs[t]
```

If the claimed coefficients differ from the true MLE evaluations of the
coefficient polynomials, then the eta-linearized values agree only on a
bounded bad challenge set:

```
BadEta = { eta in K | coeffLinearize(eta, claimed) = coeffLinearize(eta, true) }
|BadEta| / |K| <= AJTAI_D / |K|
```

This is the Lean-facing form of the Schwartz-Zippel soundness statement.

### Theorem 5: GammaLinearization

For m > 1 packed columns, the gamma-consistency check:

```
u_i* = sum_j gamma^j * w_i*^{(j)}
```

If any column disagrees, the gamma-linearized values agree only on a
bounded bad challenge set:

```
BadGamma = { gamma in K | gammaLinearize(gamma, claimed) = gammaLinearize(gamma, true) }
|BadGamma| / |K| <= m / |K|
```

This is the Lean-facing form of the Schwartz-Zippel soundness statement.

### Theorem 5b: RhoLinearization

For the final cross-claim batching step, the rho-consistency check:

```
sum_i rho^(i+1) * claimed_i = sum_i rho^(i+1) * true_i
```

If any weighted claim disagrees, the rho-linearized values agree only on a
bounded bad challenge set:

```
BadRho = { rho in K | sum_i rho^(i+1) * claimed_i = sum_i rho^(i+1) * true_i }
|BadRho| / |K| <= N / |K|
```

This is the Lean-facing form of the final Schwartz-Zippel batching bound
used in the Phase 1 error term.

### Theorem 6: Phase1Soundness

Composition of sumcheck soundness + coefficient linearization + gamma
linearization + rho-batching. Split into:

- **Phase1SoundnessCore** (deterministic): the theorem surface now carries
  explicit checked values and explicit challenge-goodness assumptions:
  - `etaConsistency`
  - `gammaConsistency`
  - `rhoConsistency`
  - `sumcheckTerminalCorrect`
  - `etaAvoidsMismatchBadSets`
  - `gammaAvoidsMismatchBadSets`
  - `rhoAvoidsMismatchBadSets`

  Under those explicit assumptions, the theorem concludes exact semantic
  correctness of the unified payloads:

```
forall i, j, t:
  unified_payloads[i].column_evals[j].coeffs[t]
    = mleEval(true_coeff_polys[i][j][t], r*)
```

- **Phase1SoundnessFailureBound** (probabilistic bookkeeping): the current
  Lean surface freezes the Phase 1 bad-event quantity as the union-bound
  sum of the five paper-level failure components:

```
epsilon = (2*ell + N*ell + AJTAI_D + m + N) / |K|
```

Components: sumcheck (2*ell/|K|), eq-poly nonzero (N*ell/|K|), eta
(AJTAI_D/|K|), gamma (m/|K|), rho (N/|K|).

This is a frozen union-bound bookkeeping quantity, not yet a fully
instantiated transcript-coin probability space.

### Bridge Lemma: SameObjectPayloadUniqueness

If two accepted unified claims share the same opened object, and the
opened-object boundary is functional at `r*`, then Phase 1 semantic
correctness forces their payloads to be equal. The Lean surface exposes
this functionality as an explicit hypothesis:

```
openedObject(a) = openedObject(b) -> truePayloadAt(a) = truePayloadAt(b)
```

This bridges Phase 1 output to Phase 2 input (`Phase2Group.hIdentical`
hypothesis) without pretending that object-to-payload functionality has
already been proved inside this module.

## Explicit Type Signatures

| Lean symbol | Type | Lives in |
|---|---|---|
| `points` | `Fin N -> (Fin ell -> K)` | Per-claim evaluation points |
| `scalarValues` | `Fin N -> K` | u_i = scalarize(payload_i) |
| `gPolys` | `Fin N -> Fin (2^ell) -> K` | Per-claim scalarized polynomial |
| `rho` | `K` | Batching challenge |
| `eta` | `K` | Coefficient linearization challenge |
| `gamma` | `K` | Column linearization challenge |
| `rStar` | `Fin ell -> K` | Verifier-derived unified point |
| `coeffPolys` | `Fin AJTAI_D -> Fin (2^ell) -> K` | Coefficient polynomials for one packed column |
| `wsClaimed` | `Fin m -> K` | Coefficient-linearized claimed values per column |
| `Phase1Accepted` | Structure | All verifier checks passed |
| `trueColumnLinearized` | `Phase1Accepted -> Fin N -> Fin m -> K` | True eta-linearized column scalar |
| `trueClaimLinearized` | `Phase1Accepted -> Fin N -> K` | True gamma-linearized claim scalar |
| `sumcheckTerminalCorrect` | `Phase1Accepted -> Prop` | Sumcheck/no-eq bad events excluded at the terminal scalar |
| `Phase1UnifiedPayloadCorrect` | Prop | Exact coefficient-level correctness at `r*` |
| `coefficientBadSet` | `Finset K` | Eta challenges that accidentally hide a bad coefficient vector |
| `gammaBadSet` | `Finset K` | Gamma challenges that accidentally hide a bad column vector |
| `rhoBadSet` | `Finset K` | Rho challenges that accidentally hide a bad claim vector |
| `etaAvoidsMismatchBadSets` | `Prop` | Eta avoids every mismatch root set for the accepted instance |
| `gammaAvoidsMismatchBadSets` | `Prop` | Gamma avoids every mismatch root set for the accepted instance |
| `rhoAvoidsMismatchBadSets` | `Prop` | Rho avoids every mismatch root set for the accepted instance |
| `phase1BadEventProbability` | `Q` | Frozen union-bound bookkeeping quantity for one accepted bucket |

## Linearization Chain

```
PackedColumnEval.coeffs[t]   (D=54 coefficients per packed column)
  -- eta linearization -->
w_i*^{(j)} : K              (one scalar per column j)
  -- gamma linearization -->
u_i* : K                    (one scalar per claim i)
  -- rho batching -->
T : K                       (single claimed sum for sumcheck)
```

## Paper Anchors

- SuperNeo Section 5: Sumcheck protocol
- SuperNeo Section 6: Batch evaluation via random linear combination
- Jolt Section 4.3: Batched opening with rho-powers

## Module Mapping

| Existing module | Import | What it provides |
|---|---|---|
| `SuperNeo.SumCheck` | Sumcheck soundness | Degree-check + consistency |
| `SuperNeo.MLE` | `mleEval`, linearity | MLE evaluation |
| `SuperNeo.EqPoly` | `eqPoly`, properties | Kronecker delta on Boolean cube |
| `SuperNeo.SchwartzZippel` | Root bound | Nonzero poly eval probability |

## Contract Surface

| Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|
| `claimedSumCorrectness` | Theorem | P0 | Verifier-computable T equals hypercube sum of P |
| `coefficientLinearization` | Theorem | P0 | `coefficientBadSet` density is at most `AJTAI_D / \|K\|` |
| `gammaLinearization` | Theorem | P0 | `gammaBadSet` density is at most `m / \|K\|` |
| `phase1SoundnessCore` | Theorem | P0 | Explicit checked equalities + good challenges imply payload correctness |
| `phase1SoundnessFailureBound` | Theorem | P0 | `phase1BadEventProbability <= epsilon` |
| `phase1Soundness` | Theorem | P0 | Core correctness plus failure-bound inequality, under explicit good-event hypotheses |
| `sameObjectPayloadUniqueness` | Theorem | Bridge | Same object + explicit true-payload functionality imply same payload |
| `Phase1Accepted` | Structure | Input | All Phase 1 verifier checks passed |
| `phase1ErrorBound` | Definition | Bound | (2*ell + N*ell + D + m + N) / \|K\| |
