import SuperNeo.Primitives.ExtensionMLE

/-!
Extension-field SumCheck scaffold.

This module mirrors the Definition-6 theorem-facing protocol surface of
`SuperNeo.SumCheckCore`/`SuperNeo.SumCheckPaper`, but over `SuperNeo.KExt`.
It owns the protocol objects and verifier-side acceptance boundary needed by
opening convergence Phase 1.
-/

namespace SuperNeo

structure ExtensionSumCheckInstance where
  rounds : Nat
  maxDegree : Nat
  domainSize : Nat
  claimedValue : KExt

structure ExtensionSumCheckTranscript where
  challenges : Array KExt
  roundPolys : Array (Array KExt)

/-- Evaluate a univariate polynomial (coefficient form, low degree first). -/
def extensionSumcheckEvalPoly (poly : Array KExt) (x : KExt) : KExt :=
  poly.foldr (fun c acc => c + x * acc) 0

/-- Basic transcript well-formedness against an instance. -/
def extensionSumcheckRoundConsistent
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  tr.challenges.size = inst.rounds ∧
  tr.roundPolys.size = inst.rounds

/-- Each round polynomial has the expected coefficient length. -/
def extensionSumcheckRoundPolyShape
  (inst : ExtensionSumCheckInstance)
  (poly : Array KExt) : Prop :=
  poly.size = inst.maxDegree + 1

/-- Every round polynomial satisfies the expected shape. -/
def extensionSumcheckRoundShapes
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    extensionSumcheckRoundPolyShape inst tr.roundPolys[i.1]

/-- Paper-facing degree bound for coefficient-array round polynomials. -/
def extensionSumcheckRoundPolyDegreeLe
  (inst : ExtensionSumCheckInstance)
  (poly : Array KExt) : Prop :=
  poly.size ≤ inst.maxDegree + 1

theorem extensionSumcheckRoundPolyShape_degreeLe
  {inst : ExtensionSumCheckInstance}
  {poly : Array KExt}
  (h : extensionSumcheckRoundPolyShape inst poly) :
  extensionSumcheckRoundPolyDegreeLe inst poly := by
  simpa [extensionSumcheckRoundPolyDegreeLe, extensionSumcheckRoundPolyShape] using Nat.le_of_eq h

/-- Paper-facing per-round degree bound extracted from the transcript encoding. -/
def extensionSumcheckRoundDegrees
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    extensionSumcheckRoundPolyDegreeLe inst tr.roundPolys[i.1]

/--
Round-transition consistency:
- each next-round polynomial satisfies the `0/1` sum equation against the
  previous round challenge evaluation.
-/
def extensionSumcheckFoldConsistent
  (tr : ExtensionSumCheckTranscript) : Prop :=
  tr.challenges.size = tr.roundPolys.size ∧
  ∀ i : Nat,
    i + 1 < tr.roundPolys.size →
      extensionSumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
          extensionSumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
        extensionSumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!)

/-- Initial round-sum consistency against the claimed value. -/
def extensionSumcheckInitialRoundConsistent
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  if _hZero : tr.roundPolys.size = 0 then
    True
  else
    extensionSumcheckEvalPoly (tr.roundPolys[0]!) 0 +
      extensionSumcheckEvalPoly (tr.roundPolys[0]!) 1 =
        inst.claimedValue

/-- Static parameter consistency required by the extension-field SumCheck instance. -/
def extensionSumcheckParameterConsistent (inst : ExtensionSumCheckInstance) : Prop :=
  inst.maxDegree ≤ inst.domainSize

/--
Internal completeness restriction of the current table-based extension-field
SumCheck model.
-/
def extensionSumcheckDegreeCompatible (inst : ExtensionSumCheckInstance) : Prop :=
  inst.rounds = 0 ∨ 0 < inst.maxDegree

/-- Core scaffold acceptance checks (without endpoint-oracle consistency). -/
def extensionSumcheckAcceptedCore
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckParameterConsistent inst ∧
  extensionSumcheckDegreeCompatible inst ∧
  extensionSumcheckRoundConsistent inst tr ∧
  extensionSumcheckRoundShapes inst tr ∧
  extensionSumcheckInitialRoundConsistent inst tr ∧
  extensionSumcheckFoldConsistent tr

/-- Paper-facing verifier checks: transcript shape, degree bound, initial, and fold. -/
def extensionSumcheckVerifierAccepted
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckRoundConsistent inst tr ∧
  extensionSumcheckRoundDegrees inst tr ∧
  extensionSumcheckInitialRoundConsistent inst tr ∧
  extensionSumcheckFoldConsistent tr

theorem extensionSumcheckVerifierAccepted_of_accepted
  {inst : ExtensionSumCheckInstance}
  {tr : ExtensionSumCheckTranscript}
  (hAcc : extensionSumcheckAcceptedCore inst tr) :
  extensionSumcheckVerifierAccepted inst tr := by
  refine ⟨hAcc.2.2.1, ?_, hAcc.2.2.2.2.1, hAcc.2.2.2.2.2⟩
  intro i
  exact extensionSumcheckRoundPolyShape_degreeLe (hAcc.2.2.2.1 i)

/-- Hypercube table sum surface used in the paper-facing extension-field statement. -/
def extensionSumcheckTableSum (table : Array KExt) : KExt :=
  table.foldr (fun v acc => v + acc) 0

/-- Paper-facing extension-field SumCheck statement object. -/
structure ExtensionSumCheckStatement (inst : ExtensionSumCheckInstance) where
  parameterConsistent : extensionSumcheckParameterConsistent inst
  degreeCompatible : extensionSumcheckDegreeCompatible inst
  table : Array KExt
  tableSize : table.size = 2 ^ inst.rounds
  hypercubeSumEqClaimed : extensionSumcheckTableSum table = inst.claimedValue

/-- Final-round oracle consistency for a statement witness and transcript. -/
def extensionSumcheckFinalOracleConsistent
  (inst : ExtensionSumCheckInstance)
  (stmt : ExtensionSumCheckStatement inst)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckRoundConsistent inst tr ∧
  if _hZero : inst.rounds = 0 then
    mleByFoldingK stmt.table #[] = inst.claimedValue
  else
    extensionSumcheckEvalPoly (tr.roundPolys[inst.rounds - 1]!) (tr.challenges[inst.rounds - 1]!) =
      mleByFoldingK stmt.table tr.challenges

/-- Table-indexed final oracle consistency with transcript-shape safety. -/
def extensionSumcheckFinalOracleConsistentWithTable
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  table.size = 2 ^ inst.rounds ∧
  extensionSumcheckRoundConsistent inst tr ∧
  if _hZero : inst.rounds = 0 then
    mleByFoldingK table #[] = inst.claimedValue
  else
    extensionSumcheckEvalPoly (tr.roundPolys[inst.rounds - 1]!) (tr.challenges[inst.rounds - 1]!) =
      mleByFoldingK table tr.challenges

theorem extensionSumcheckFinalOracleConsistent_iff_withTable
  {inst : ExtensionSumCheckInstance}
  {stmt : ExtensionSumCheckStatement inst}
  {tr : ExtensionSumCheckTranscript} :
  extensionSumcheckFinalOracleConsistent inst stmt tr ↔
    extensionSumcheckFinalOracleConsistentWithTable inst stmt.table tr := by
  constructor
  · intro h
    exact ⟨stmt.tableSize, h⟩
  · intro h
    exact h.2

/-- Verifier acceptance predicate for the extension-field SumCheck protocol. -/
def extensionSumcheckAccepted
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckAcceptedCore inst tr

/-- Constructively closed acceptance surface. -/
def extensionSumcheckAcceptedClosed
  (inst : ExtensionSumCheckInstance)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckAccepted inst tr ∧
  ∃ stmt : ExtensionSumCheckStatement inst, extensionSumcheckFinalOracleConsistent inst stmt tr

/-- Verifier acceptance for a fixed table witness. -/
def extensionSumcheckAcceptedForTable
  (inst : ExtensionSumCheckInstance)
  (table : Array KExt)
  (tr : ExtensionSumCheckTranscript) : Prop :=
  extensionSumcheckAccepted inst tr ∧
  extensionSumcheckFinalOracleConsistentWithTable inst table tr

/-- Verifier-facing SumCheck claim object. -/
structure ExtensionSumCheckClaim (inst : ExtensionSumCheckInstance) where
  transcript : ExtensionSumCheckTranscript
  parameterConsistent : extensionSumcheckParameterConsistent inst
  degreeCompatible : extensionSumcheckDegreeCompatible inst
  roundConsistent : extensionSumcheckRoundConsistent inst transcript
  roundShapes : extensionSumcheckRoundShapes inst transcript
  initialRound : extensionSumcheckInitialRoundConsistent inst transcript
  foldConsistent : extensionSumcheckFoldConsistent transcript

theorem ExtensionSumCheckClaim.accepted
  {inst : ExtensionSumCheckInstance}
  (c : ExtensionSumCheckClaim inst) :
  extensionSumcheckAcceptedCore inst c.transcript := by
  exact ⟨c.parameterConsistent, c.degreeCompatible, c.roundConsistent, c.roundShapes,
    c.initialRound, c.foldConsistent⟩

theorem extensionSumcheckAccepted_of_acceptedForTable
  {inst : ExtensionSumCheckInstance}
  {table : Array KExt}
  {tr : ExtensionSumCheckTranscript}
  (hAccTable : extensionSumcheckAcceptedForTable inst table tr) :
  extensionSumcheckAccepted inst tr := by
  exact hAccTable.1

theorem extensionSumcheckAcceptedClosed_of_acceptedForTable
  {inst : ExtensionSumCheckInstance}
  {table : Array KExt}
  {tr : ExtensionSumCheckTranscript}
  (hAccTable : extensionSumcheckAcceptedForTable inst table tr)
  (hSum : extensionSumcheckTableSum table = inst.claimedValue) :
  extensionSumcheckAcceptedClosed inst tr := by
  rcases hAccTable with ⟨hCore, hFinalTable⟩
  have hParam : extensionSumcheckParameterConsistent inst := hCore.1
  have hDegree : extensionSumcheckDegreeCompatible inst := hCore.2.1
  have hTableSize : table.size = 2 ^ inst.rounds := hFinalTable.1
  let stmt : ExtensionSumCheckStatement inst :=
    { parameterConsistent := hParam
      degreeCompatible := hDegree
      table := table
      tableSize := hTableSize
      hypercubeSumEqClaimed := hSum }
  have hFinal :
      extensionSumcheckFinalOracleConsistent inst stmt tr := by
    exact hFinalTable.2
  refine ⟨hCore, ?_⟩
  exact ⟨stmt, hFinal⟩

end SuperNeo
