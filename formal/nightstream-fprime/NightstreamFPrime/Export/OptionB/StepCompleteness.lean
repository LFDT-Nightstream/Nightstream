import NightstreamFPrime.Export.OptionB.SlotCompleteness
import NightstreamFPrime.Export.OptionB.StepRelation

/-! Completeness of the exported step relations (CW-10.2), the converse of
`StepRelation.light_freshHolds` and `StepRelation.refresh_freshHolds`. Take an environment in
which the step circuit's rows hold, as the assembler gives it for an honest step
(`Layout.OptionB.LightAssemblerCompleteness.complete`, `RefreshAssemblerCompleteness.complete`).
Set its public columns to `encHash` of its output digest (`withPublic`). The circuit reads no
public column, so its rows still hold, and the decode constraints hold (`pieces_complete`).
`SlotCompleteness.rowsZero_of_holds` then gives an assignment that satisfies the relation at the
key's widened slot layout, with small coordinates, and `Plan.rowsZero_implies_freshHolds` gives the
fresh CCS statement with public input `encHash digest` (`fresh_complete`). `light_complete` and
`refresh_complete` read the digest from the step's output.

Four premises are computed checks on the package, as `CircuitRelation.packable` is: the blocks
pack, every piece reads only columns below the first fresh column
(`PieceRowsCompleteness.pieceBelow`), no piece reads a quiet column (`SlotCompleteness.bitsSafe`),
and no row of the circuit reads a public column (`publicFree`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.StepCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.OptionB.SlotRuns
open NightstreamFPrime.Export.OptionB.SlotCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- No row of the circuit reads a public column. -/
def publicFree (constraints : List Expr) : Bool :=
  constraints.all fun expression => (exprVars expression).all fun column =>
    decide (publicColumns ≤ column)

/-- `env` with its public columns set to `encHash digest`. -/
def withPublic (digest : Digest) (env : Env) : Env :=
  fun column => if inside : column < publicColumns then
    encHash (publicFits := publicFits) digest ⟨column, inside⟩ else env column

theorem publicInput_withPublic (digest : Digest) (env : Env) :
    LogicalPlan.publicInput (publicFits := publicFits)
        (withPublic (publicFits := publicFits) digest env) =
      encHash (publicFits := publicFits) digest := by
  funext column
  show (if inside : column.val < publicColumns then _ else _) = _
  rw [dite_eq_left (show column.val < publicColumns from column.isLt)]

/-- The circuit's rows with the decode constraints of its output digest hold, piece by piece,
once the public columns carry `encHash digest`. -/
theorem pieces_complete (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (env : Env) (holds : holdsFlat env (Circuit.ops circuit.main offset)) (digest : Digest)
    (digestWords : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (digestAbove : ∀ word, publicColumns ≤ digestColumn word)
    (free : publicFree (flatConstraints (Circuit.ops circuit.main offset)) = true) :
    ∀ piece ∈ CircuitRelation.pieces circuit offset digestColumn,
      ConstraintsHold (withPublic (publicFits := publicFits) digest env)
        (piece.constraints (PilotData.canonicalConstraints ())) := by
  have all : ConstraintsHold (withPublic (publicFits := publicFits) digest env)
      (CircuitRelation.constraints circuit offset digestColumn) := by
    intro expression member
    rcases List.mem_append.mp member with flat | decode
    · have above := List.all_eq_true.mp free expression flat
      rw [eval_of_vars (right := env) expression fun column read => by
        have bound := List.all_eq_true.mp above column read
        simp only [decide_eq_true_eq] at bound
        simp [withPublic, not_lt.mpr bound]]
      exact holds expression flat
    · obtain ⟨word, rfl⟩ := List.mem_ofFn.mp decode
      rw [LogicalPlan.decodeConstraint, Expr.eval_sub, Expr.eval_var,
        LogicalPlan.decodeExpr_eval (publicFits := publicFits), publicInput_withPublic,
        decodeHashWord_encHash]
      simp [withPublic, not_lt.mpr (digestAbove word), digestWords]
  intro piece member expression inPiece
  apply all
  show expression ∈ CircuitRelation.constraints circuit offset digestColumn
  rw [← flatMap_split (PilotData.canonicalConstraints ())
    (CircuitRelation.constraints circuit offset digestColumn)]
  exact List.mem_flatMap.mpr ⟨piece, member, inPiece⟩

theorem quiet_public {pieces : List Piece} {next : Nat}
    (packed : packable publicColumns pieces = true) (publicBelow : publicColumns ≤ next)
    (batches : List WitnessBatch) (env : Env) {column : Nat} (inside : column < publicColumns) :
    completeEnv pieces next (quiet publicColumns pieces next batches env) column = env column := by
  rw [completeEnv_low (by have := next_le_sboxStart pieces next; omega),
    executePieces_below _ _ _ _ _ (by omega)]
  have notQuiet : (quietMarks publicColumns pieces next batches).getD column false = false := by
    rw [quietMarks_getD]
    simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
    left; left; right
    omega
  simp [liveEnv, packable_public packed inside, quiet, notQuiet]

/-- The fresh CCS statement of an honest step: the commitment to the honest assignment and the
public input `encHash digest` hold under the relation of the circuit's key, read through the
package's slot layout widened to the key width. -/
theorem fresh_complete (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (env : Env) (holds : holdsFlat env (Circuit.ops circuit.main offset)) (digest : Digest)
    (digestWords : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (digestAbove : ∀ word, publicColumns ≤ digestColumn word)
    (free : publicFree (flatConstraints (Circuit.ops circuit.main offset)) = true)
    (firstFresh width : Nat)
    (fits : Fits publicColumns (CircuitRelation.pieces circuit offset digestColumn) firstFresh width)
    (packed : packable publicColumns (CircuitRelation.pieces circuit offset digestColumn) = true)
    (below : (CircuitRelation.pieces circuit offset digestColumn).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (safe : bitsSafe publicColumns (CircuitRelation.pieces circuit offset digestColumn) firstFresh
      batches = true)
    (wideFits : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (CircuitRelation.pieces circuit offset digestColumn)
        firstFresh batches)) ≤ logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fits
            (WideLayout.widen (SlotLayout.layout _) wideFits)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
              (Phi81CarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) digest })
        (Phi81CarrierLayout.extendAssignment 0 assignment) := by
  have piecesHold := pieces_complete (publicFits := publicFits) circuit offset digestColumn env
    holds digest digestWords digestAbove free
  have marker : (withPublic (publicFits := publicFits) digest env) 0 = 1 := by
    have atMarker := congrFun (publicInput_withPublic (publicFits := publicFits) digest env)
      encHashMarkerIndex
    rw [encHash_marker] at atMarker
    simpa [LogicalPlan.publicInput, encHashMarkerIndex] using atMarker
  have publicNorm : ∀ column, column < publicColumns → centeredMagnitude ((withPublic (publicFits := publicFits) digest env) column) < 2 := by
    intro column inside
    have atColumn := congrFun (publicInput_withPublic (publicFits := publicFits) digest env)
      ⟨column, inside⟩
    simp only [LogicalPlan.publicInput] at atColumn
    rw [atColumn]
    exact encHash_norm (publicFits := publicFits) digest ⟨column, inside⟩
  obtain ⟨rows, bounded⟩ := SlotCompleteness.rowsZero_of_holds fits packed
    (fun piece member => valid_split _ _ piece member)
    (fun piece member => List.all_eq_true.mp below piece member) publicBelow batches safe wideFits
    (withPublic (publicFits := publicFits) digest env) piecesHold marker publicNorm
  refine ⟨_, Layout.ProductionRelation.Plan.rowsZero_implies_freshHolds _ cubeFits ajtai _ _ rows
    bounded ?_⟩
  funext column
  have inside : column.val < logicalWidth :=
    Nat.lt_of_lt_of_le column.isLt (le_trans (SlotLayout.publicFits _) wideFits)
  have same : (PaperAlgebra.FullShape logicalWidth publicFits).publicColumn column =
      Phi81CarrierLayout.embedLogical (⟨column.val, inside⟩ : Fin logicalWidth) := by
    apply Fin.ext
    rfl
  rw [Phi81Relation.projectPublicInput, same, Phi81CarrierLayout.extendAssignment_embedLogical]
  have narrow : column.val < ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (CircuitRelation.pieces circuit offset digestColumn) firstFresh batches)) :=
    Nat.lt_of_lt_of_le column.isLt (SlotLayout.publicFits _)
  simp only [WideLayout.extend, honestAssignment]
  rw [dite_eq_left narrow,
    show (⟨column.val, narrow⟩ : Fin _) =
      ProductionAssignment.publicColumn ⟨column.val, column.isLt⟩ from rfl,
    ProductionAssignment.logicalAssignment_publicColumn]
  rw [quiet_public packed publicBelow batches _ column.isLt]
  exact congrFun (publicInput_withPublic (publicFits := publicFits) digest env) column

/-! ### The light and refresh steps -/

section Steps

variable (program : Stage1.Application.Program)

theorem digest_getD {env : Env} {digest : Digest} {start : Nat}
    (digestEq : digest = List.ofFn fun lane : Fin 4 => (Expr.var (start + lane.val)).eval env)
    (word : Fin 4) : env (start + word.val) = digest.getD word.val 0 := by
  subst digestEq
  fin_cases word <;> rfl

theorem digestColumn_above (word : Fin 4) :
    publicColumns ≤ LightAssembler.digestsStart StepRelation.base program.witnessWordCount + 16 +
      word.val := by
  have baseEq : StepRelation.base = publicColumns := rfl
  simp only [LightAssembler.digestsStart, LightAssembler.slotsStart, LightAssembler.witnessStart,
    baseEq]
  omega

/-- The pieces of the exported light circuit. -/
abbrev lightPieces (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    List Piece :=
  CircuitRelation.pieces (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai)
    (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)
    (StepRelation.lightDigestColumn program)

/-- The pieces of the exported refresh circuit. -/
abbrev refreshPieces (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    List Piece :=
  CircuitRelation.pieces
    (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai)
    (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)
    (StepRelation.refreshDigestColumn program)

/-- An honest light step's assignment (`LightAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the light key. -/
theorem light_complete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (env : Env)
    (holds : holdsFlat env (Circuit.ops
      (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai).main
      (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)))
    (represents : LightStepSoundness.StepRepresents program
      (StepRelation.lightInterface (logicalWidth := logicalWidth) (publicFits := publicFits) program)
      ajtai circuitKeys 0 input output proof env)
    (free : publicFree (flatConstraints (Circuit.ops
      (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai).main
      (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount))) = true)
    (firstFresh width : Nat)
    (fits : Fits publicColumns (lightPieces program relation ajtai) firstFresh width)
    (packed : packable publicColumns (lightPieces program relation ajtai) = true)
    (below : (lightPieces program relation ajtai).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (safe : bitsSafe publicColumns (lightPieces program relation ajtai) firstFresh batches = true)
    (wideFits : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (lightPieces program relation ajtai) firstFresh batches)) ≤
        logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fits
            (WideLayout.widen (SlotLayout.layout _) wideFits)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
              (Phi81CarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) output.x })
        (Phi81CarrierLayout.extendAssignment 0 assignment) :=
  fresh_complete _ _ _ env holds output.x (digest_getD represents.outputDigest)
    (digestColumn_above program) free firstFresh width fits packed below publicBelow batches safe
    wideFits cubeFits ajtai

/-- An honest refresh step's assignment (`RefreshAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the refresh key. -/
theorem refresh_complete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8) (env : Env)
    (holds : holdsFlat env (Circuit.ops
      (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai).main
      (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)))
    (represents : RefreshStepSoundness.StepRepresents program
      (StepRelation.refreshInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
        program) ajtai circuitKeys 1 input output proof env)
    (free : publicFree (flatConstraints (Circuit.ops
      (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai).main
      (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount))) = true)
    (firstFresh width : Nat)
    (fits : Fits publicColumns (refreshPieces program relation ajtai) firstFresh width)
    (packed : packable publicColumns (refreshPieces program relation ajtai) = true)
    (below : (refreshPieces program relation ajtai).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (safe : bitsSafe publicColumns (refreshPieces program relation ajtai) firstFresh batches = true)
    (wideFits : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (refreshPieces program relation ajtai) firstFresh
        batches)) ≤ logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fits
            (WideLayout.widen (SlotLayout.layout _) wideFits)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
              (Phi81CarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) output.x })
        (Phi81CarrierLayout.extendAssignment 0 assignment) :=
  fresh_complete _ _ _ env holds output.x (digest_getD represents.outputDigest)
    (digestColumn_above program) free firstFresh width fits packed below publicBelow batches safe
    wideFits cubeFits ajtai

end Steps

end NightstreamFPrime.Export.OptionB.StepCompleteness
