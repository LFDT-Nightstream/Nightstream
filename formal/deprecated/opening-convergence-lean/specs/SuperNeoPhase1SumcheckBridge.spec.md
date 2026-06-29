# SuperNeo Phase 1 SumCheck Bridge

## Goal

Freeze the exact theorem that connects the concrete `superneo-lean`
extension-field SumCheck final-oracle surface to the abstract
`OpeningConvergence.BatchEvalReduction.sumcheckTerminalCorrect` boundary.

This module does not try to re-prove SumCheck soundness. It only states and
proves the bridge:

- if a concrete extension-field statement/transcript pair is final-oracle
  consistent,
- if the opening-convergence terminal scalar is the concrete verifier-side
  terminal value extracted from that transcript,
- and if the statement table's guarded MLE evaluation is known to equal the
  true Phase 1 batched scalar sum,

then the abstract Phase 1 `sumcheckTerminalCorrect` predicate holds.

## Theorem Target

For `K = SuperNeo.KExt`, define:

- `pointArray(accepted, i) := Array.ofFn (accepted.points i)`
- `rStarArray(accepted) := Array.ofFn accepted.rStar`
- `trueColumnAtPoint(accepted, i, j, x) := Σ_t eta^t * MLE(f_i^{(j,t)})(x)`
- `trueClaimAtPoint(accepted, i, x) :=`
  - `trueColumnAtPoint(accepted, i, 0, x)` when `m_i = 1`
  - `Σ_j gamma^j * trueColumnAtPoint(accepted, i, j, x)` otherwise
- `phase1BatchedPolynomial(accepted, x) := Σ_i rho^i * eq(r_i, x) * trueClaimAtPoint(accepted, i, x)`
- `trueClaimTable(accepted)[x] := phase1BatchedPolynomial(accepted, bits(x))`
- `trueClaimSum(accepted) := Σ_i rho^i * eq(r_i, r*) * trueClaimLinearized_i`
- `terminalValue(inst, tr) :=`
  - `inst.claimedValue` when `inst.rounds = 0`
  - `extensionSumcheckEvalPoly(lastRoundPoly, lastChallenge)` otherwise

Then prove:

- the canonical-object shape lemmas:
  - `pointArray_size`
  - `rStarArray_size`
  - `trueClaimTable_size`
- `phase1BatchedPolynomial_rStar_eq_trueClaimSum`
- `sumcheckTerminalCorrect_of_extensionFinalOracle`

under the hypotheses:

- `extensionSumcheckStatementTranscriptConsistent inst stmt tr`
- `accepted.terminalSumcheckValue = terminalValue inst tr`
- the appropriate `mleEvalK` bridge from `stmt.table` to
  `trueClaimSum accepted`

the conclusion is:

- `BatchEvalReduction.sumcheckTerminalCorrect accepted`

## Why This Matters

The current opening-convergence package still carries `sumcheckTerminalCorrect`
as a hypothesis. This bridge narrows that boundary to the concrete
extension-field protocol objects already formalized in `superneo-lean`.

The remaining semantic closure target in this module is now explicit:

- identify the exact relationship between:
  - the actual degree-2 polynomial value
    `phase1BatchedPolynomial(accepted, r*) = trueClaimSum accepted`
  - and the current table/MLE final-oracle path over
    `trueClaimTable accepted`

This distinction matters because the frozen Phase 1 polynomial
`P(x) = Σ_i rho^i * eq(r_i, x) * g_i(x)` is degree-2 in each coordinate,
while a table/MLE final oracle reasons about the multilinear extension of the
Boolean-cube table of `P`. Any bridge that identifies those two must be
proved explicitly; it is not definitionally free.
