import OpeningConvergence.BatchEvalReductionInterface
import SuperNeo.Primitives.ExtensionFieldInterface
import SuperNeo.Primitives.ExtensionMLEInterface
import SuperNeo.Primitives.ExtensionSumCheckInterface

/-!
# Module 8: SuperNeoPhase1SumcheckBridge — Interface

Owns the theorem that bridges the concrete extension-field SumCheck
final-oracle surface from `superneo-lean` to the abstract Phase 1
`sumcheckTerminalCorrect` predicate used by opening convergence.

## Spec
See `specs/SuperNeoPhase1SumcheckBridge.spec.md`
-/

namespace OpeningConvergence.SuperNeoPhase1SumcheckBridge

abbrev K := SuperNeo.ExtensionFieldInterface.KExt

open OpeningConvergence.BatchEvalReduction

/-- Canonical array encoding of one original claim point. -/
noncomputable def pointArray
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N) : Array K :=
  Array.ofFn (accepted.points i)

/-- Canonical array encoding of the verifier-derived unified point `r*`. -/
noncomputable def rStarArray
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Array K :=
  Array.ofFn accepted.rStar

/-- The true eta-linearized column value at an arbitrary evaluation point. -/
noncomputable def trueColumnAtPoint
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema))
    (x : Fin ell → K) : K :=
  Finset.sum Finset.univ fun t : Fin AJTAI_D =>
    accepted.eta ^ (t : Nat) * mleEval (accepted.trueCoeffPolys i j t) x

/-- The true gamma-linearized claim scalar at an arbitrary evaluation point. -/
noncomputable def trueClaimAtPoint
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (x : Fin ell → K) : K :=
  let m := packedColumnCount (accepted.unifiedPayloads i).schema
  let ws : Fin m → K := fun j => trueColumnAtPoint accepted i j x
  if h : m = 1 then
    ws ⟨0, by omega⟩
  else
    gammaLinearize accepted.gamma ws

/-- The true eta-linearized column value on one Boolean-cube row. -/
noncomputable def trueColumnOnCube
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema))
    (x : Fin (2 ^ ell)) : K :=
  trueColumnAtPoint accepted i j (boolCubeBits x)

/-- The true gamma-linearized claim scalar on one Boolean-cube row. -/
noncomputable def trueClaimOnCube
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (x : Fin (2 ^ ell)) : K :=
  trueClaimAtPoint accepted i (boolCubeBits x)

/-- The actual frozen Phase 1 batched polynomial at an arbitrary verifier
    point. This is the paper-facing degree-2 object. -/
noncomputable def phase1BatchedPolynomial
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (x : Fin ell → K) : K :=
  Finset.sum Finset.univ fun i : Fin N =>
    accepted.rho ^ ((i : Nat) + 1) *
      eqPoly (accepted.points i) x *
      trueClaimAtPoint accepted i x

/-- Canonical Phase 1 batched table over the Boolean cube.

This is the Boolean-cube restriction of the actual Phase 1 batched
polynomial `phase1BatchedPolynomial`.
-/
noncomputable def trueClaimTable
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Array K :=
  Array.ofFn fun x : Fin (2 ^ ell) =>
    phase1BatchedPolynomial accepted (boolCubeBits x)

/-- The canonical point-array encoding has width `ell`. -/
@[simp] theorem pointArray_size
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N) :
    (pointArray accepted i).size = ell := by
  simp [pointArray]

/-- The canonical `r*` array encoding has width `ell`. -/
@[simp] theorem rStarArray_size
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) :
    (rStarArray accepted).size = ell := by
  simp [rStarArray]

/-- The canonical Phase 1 batched table has Boolean-cube width `2^ell`. -/
@[simp] theorem trueClaimTable_size
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) :
    (trueClaimTable accepted).size = 2 ^ ell := by
  simp [trueClaimTable]

/-- The true rho-weighted Phase 1 scalar sum at the verifier-derived point. -/
noncomputable def trueClaimSum
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : K :=
  Finset.sum Finset.univ fun i : Fin N =>
    accepted.rho ^ ((i : Nat) + 1) * pointWeight accepted i * trueClaimLinearized accepted i

/-- Evaluating one column-polynomial package at `r*` agrees with the frozen
    linearized column value used elsewhere in Phase 1. -/
theorem trueColumnAtPoint_rStar_eq_trueColumnLinearized
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema)) :
    trueColumnAtPoint accepted i j accepted.rStar =
      trueColumnLinearized accepted i j := by
  unfold trueColumnAtPoint trueColumnLinearized
  rfl

/-- Evaluating the actual Phase 1 batched polynomial at the verifier-derived
    point `r*` yields the weighted scalar sum used by
    `sumcheckTerminalCorrect`. -/
theorem phase1BatchedPolynomial_rStar_eq_trueClaimSum
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) :
    phase1BatchedPolynomial accepted accepted.rStar = trueClaimSum accepted := by
  unfold phase1BatchedPolynomial trueClaimSum pointWeight trueClaimAtPoint trueClaimLinearized
  refine Finset.sum_congr rfl ?_
  intro i _
  by_cases hOne : packedColumnCount (accepted.unifiedPayloads i).schema = 1
  · simp [hOne, trueColumnAtPoint_rStar_eq_trueColumnLinearized]
  · simp [hOne, trueColumnAtPoint_rStar_eq_trueColumnLinearized]

/-- The concrete verifier-side terminal value extracted from an extension-field
    SumCheck transcript. -/
noncomputable def terminalValue
    (inst : SuperNeo.ExtensionSumCheckInterface.ExtensionSumCheckInstance)
    (tr : SuperNeo.ExtensionSumCheckInterface.ExtensionSumCheckTranscript) : K :=
  if _hZero : inst.rounds = 0 then
    inst.claimedValue
  else
    SuperNeo.ExtensionSumCheckInterface.extensionSumcheckEvalPoly
      tr.roundPolys[inst.rounds - 1]!
      tr.challenges[inst.rounds - 1]!

/-- Concrete extension-field final-oracle consistency plus an explicit
    folding-evaluation bridge implies the abstract Phase 1
    `sumcheckTerminalCorrect` predicate. -/
theorem sumcheckTerminalCorrect_of_extensionFinalOracle
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (inst : SuperNeo.ExtensionSumCheckInterface.ExtensionSumCheckInstance)
    (stmt : SuperNeo.ExtensionSumCheckInterface.ExtensionSumCheckStatement inst)
    (tr : SuperNeo.ExtensionSumCheckInterface.ExtensionSumCheckTranscript)
    (hConsistent :
      SuperNeo.ExtensionSumCheckInterface.extensionSumcheckStatementTranscriptConsistent
        inst stmt tr)
    (hTerminal :
      accepted.terminalSumcheckValue = terminalValue inst tr)
    (hEvalBridge :
      if _hZero : inst.rounds = 0 then
        SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table #[] = trueClaimSum accepted
      else
        SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table tr.challenges = trueClaimSum accepted)
    :
    sumcheckTerminalCorrect accepted := by
  unfold sumcheckTerminalCorrect
  by_cases hZero : inst.rounds = 0
  · have hOracle :
        SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table #[] = inst.claimedValue := by
        simpa [SuperNeo.ExtensionSumCheckInterface.extensionSumcheckFinalOracleConsistent, hZero]
          using hConsistent.2.2
    have hTableSize : stmt.table.size = 1 := by
      simpa [hZero] using stmt.tableSize
    have hEvalToFold :
        SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table #[] =
          SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table #[] := by
      exact SuperNeo.ExtensionMLEInterface.mleEvalK_eq_mleByFoldingK_of_size hTableSize
    calc
      accepted.terminalSumcheckValue = terminalValue inst tr := hTerminal
      _ = inst.claimedValue := by simp [terminalValue, hZero]
      _ = SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table #[] := hOracle.symm
      _ = SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table #[] := hEvalToFold.symm
      _ = trueClaimSum accepted := by
            simpa [hZero] using hEvalBridge
      _ = Finset.sum Finset.univ
            (fun i : Fin N =>
              accepted.rho ^ ((i : Nat) + 1) *
                pointWeight accepted i * trueClaimLinearized accepted i) := by
            simp [trueClaimSum]
  · have hOracle :
        SuperNeo.ExtensionSumCheckInterface.extensionSumcheckEvalPoly
            tr.roundPolys[inst.rounds - 1]!
            tr.challenges[inst.rounds - 1]!
          =
        SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table tr.challenges := by
        simpa [SuperNeo.ExtensionSumCheckInterface.extensionSumcheckFinalOracleConsistent, hZero]
          using hConsistent.2.2
    have hChallengeSize : tr.challenges.size = inst.rounds := by
      exact hConsistent.1.2.2.1.1
    have hEvalToFold :
        SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table tr.challenges =
          SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table tr.challenges := by
      exact SuperNeo.ExtensionMLEInterface.mleEvalK_eq_mleByFoldingK_of_size
        (by simpa [hChallengeSize] using stmt.tableSize)
    calc
      accepted.terminalSumcheckValue = terminalValue inst tr := hTerminal
      _ =
          SuperNeo.ExtensionSumCheckInterface.extensionSumcheckEvalPoly
            tr.roundPolys[inst.rounds - 1]!
            tr.challenges[inst.rounds - 1]! := by
              simp [terminalValue, hZero]
      _ = SuperNeo.ExtensionMLEInterface.mleByFoldingK stmt.table tr.challenges := hOracle
      _ = SuperNeo.ExtensionMLEInterface.mleEvalK stmt.table tr.challenges := hEvalToFold.symm
      _ = trueClaimSum accepted := by
            simpa [hZero] using hEvalBridge
      _ = Finset.sum Finset.univ
            (fun i : Fin N =>
              accepted.rho ^ ((i : Nat) + 1) *
                pointWeight accepted i * trueClaimLinearized accepted i) := by
            simp [trueClaimSum]

end OpeningConvergence.SuperNeoPhase1SumcheckBridge
