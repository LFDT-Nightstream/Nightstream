import NightstreamFPrime.Export.BoundedDepth.SlotCompleteness
import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening

/-! Completeness of the exported step relations (CW-10.2), the converse of
`StepRelation.light_freshHolds` and `StepRelation.refresh_freshHolds`. Take an environment in
which the step circuit's rows hold, as the assembler gives it for an honest step
(`Layout.BoundedDepth.LightAssemblerCompleteness.complete`, `RefreshAssemblerCompleteness.complete`).
Its public columns carry `encHash` of its output digest, as the assembler's source columns put
it there, so the circuit's rows and the decode constraints hold in it (`constraints_complete`,
`pieces_complete`).
`SlotCompleteness.rowsZero_of_holds` then gives an assignment that satisfies the relation at the
key's widened slot layout, with small coordinates, and `Plan.rowsZero_implies_freshHolds` gives the
fresh CCS statement with public input `encHash digest` (`fresh_complete`). The assignment extends
the package's logical width with zeros (`WideLayout.extend`), so it is zero past that width.
`light_complete` and `refresh_complete` read the digest from the step's output.

One premise is a computed check on the package, as `CircuitRelation.packable` is: the blocks
pack. That every piece reads only columns below the first fresh column
(`PieceRowsCompleteness.pieceBelow`) follows when every row of the circuit reads only columns
before the end of its locals, as the assembler gives it for an honest step, and the first fresh
column is at or after that end (`lightPieces_readOwnColumns`, `refreshPieces_readOwnColumns`).
The emitter also checks that no row of the circuit reads a public column (`publicFree`); the
proofs do not need it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.StepCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.BoundedDepth.SlotRuns
open NightstreamFPrime.Export.BoundedDepth.SlotCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- No row of the circuit reads a public column. -/
def publicFree (constraints : List Expr) : Bool :=
  constraints.all fun expression => (SlotLayout.exprColumns expression).all fun column =>
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

/-- An environment whose public columns carry `encHash digest` is its own `withPublic`. -/
theorem withPublic_eq (digest : Digest) (env : Env)
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) digest ⟨column, inside⟩) :
    withPublic (publicFits := publicFits) digest env = env := by
  funext column
  unfold withPublic
  split
  · rename_i inside
    exact (publicEq column inside).symm
  · rfl

/-- Rows that hold in an environment whose public columns carry `encHash digest`, with the
decode constraints of the digest that the words `digestColumn` carry, hold in its `withPublic`. -/
theorem constraints_complete (ops : List Op) (digestColumn : Fin 4 → Nat) (env : Env)
    (holds : holdsFlat env ops) (digest : Digest)
    (digestWords : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (digestAbove : ∀ word, publicColumns ≤ digestColumn word)
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) digest ⟨column, inside⟩) :
    ConstraintsHold (withPublic (publicFits := publicFits) digest env)
      (flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn) := by
  intro expression member
  rcases List.mem_append.mp member with flat | decode
  · rw [withPublic_eq digest env publicEq]
    exact holds expression flat
  · obtain ⟨word, rfl⟩ := List.mem_ofFn.mp decode
    rw [LogicalPlan.decodeConstraint, Expr.eval_sub, Expr.eval_var,
      LogicalPlan.decodeExpr_eval (publicFits := publicFits), publicInput_withPublic,
      decodeHashWord_encHash]
    simp [withPublic, not_lt.mpr (digestAbove word), digestWords]

/-- The circuit's rows with the decode constraints of its output digest hold, piece by piece,
when the public columns carry `encHash digest`. -/
theorem pieces_complete (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (env : Env) (holds : holdsFlat env (Circuit.ops circuit.main offset)) (digest : Digest)
    (digestWords : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (digestAbove : ∀ word, publicColumns ≤ digestColumn word)
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) digest ⟨column, inside⟩) :
    ∀ piece ∈ CircuitRelation.pieces circuit offset digestColumn,
      ConstraintsHold (withPublic (publicFits := publicFits) digest env)
        (piece.constraints (PilotData.canonicalConstraints ())) := by
  have all : ConstraintsHold (withPublic (publicFits := publicFits) digest env)
      (CircuitRelation.constraints circuit offset digestColumn) :=
    constraints_complete (publicFits := publicFits) _ digestColumn env holds digest digestWords
      digestAbove publicEq
  intro piece member expression inPiece
  apply all
  show expression ∈ CircuitRelation.constraints circuit offset digestColumn
  rw [← flatMap_split (PilotData.canonicalConstraints ()) (marks (Circuit.ops circuit.main offset))
    (CircuitRelation.constraints circuit offset digestColumn)]
  exact List.mem_flatMap.mpr ⟨piece, member, inPiece⟩

/-- The fresh CCS statement of an honest step: the commitment to the honest assignment and the
public input `encHash digest` hold under the relation of the circuit's key, read through the
package's slot layout widened to the key width. The committed witness is zero past the package's
logical width. -/
theorem fresh_complete (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (env : Env) (holds : holdsFlat env (Circuit.ops circuit.main offset)) (digest : Digest)
    (digestWords : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (digestAbove : ∀ word, publicColumns ≤ digestColumn word)
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) digest ⟨column, inside⟩)
    (firstFresh width : Nat)
    (fits : Fits publicColumns (CircuitRelation.pieces circuit offset digestColumn) firstFresh width)
    (packed : packable publicColumns (CircuitRelation.pieces circuit offset digestColumn) = true)
    (below : (CircuitRelation.pieces circuit offset digestColumn).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (wideFits : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (CircuitRelation.pieces circuit offset digestColumn)
        firstFresh batches)) ≤ logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fits
            (WideLayout.widen (SlotLayout.layout _) wideFits)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => RingRelation.PiRLCAlgebra.Commitment.commit ajtai
              (RingCarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) digest })
        (RingCarrierLayout.extendAssignment 0 assignment) ∧
      ∀ column : Fin (RingCarrierLayout.carrierWidth logicalWidth),
        ProductionAssignment.logicalWidth (SlotLayout.runSlots width
          (StepPackage.slotRuns publicColumns (CircuitRelation.pieces circuit offset digestColumn)
            firstFresh batches)) ≤ column.val →
        RingCarrierLayout.extendAssignment 0 assignment column = 0 := by
  have piecesHold := pieces_complete (publicFits := publicFits) circuit offset digestColumn env
    holds digest digestWords digestAbove publicEq
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
    (fun piece member => valid_split _ _ _ piece member)
    (fun piece member => List.all_eq_true.mp below piece member) publicBelow batches wideFits
    (withPublic (publicFits := publicFits) digest env) piecesHold marker publicNorm
  refine ⟨_, Layout.ProductionRelation.Plan.rowsZero_implies_freshHolds _ cubeFits ajtai _ _ rows
    bounded ?_, ?_⟩
  swap
  · intro column above
    by_cases inside : column.val < logicalWidth
    · rw [show column = RingCarrierLayout.embedLogical (⟨column.val, inside⟩ : Fin logicalWidth)
        from Fin.ext rfl, RingCarrierLayout.extendAssignment_embedLogical]
      unfold WideLayout.extend
      exact dite_eq_right (not_lt.mpr above)
    · exact RingCarrierLayout.extendAssignment_tail_zero 0 _ column (not_lt.mp inside)
  funext column
  have inside : column.val < logicalWidth :=
    Nat.lt_of_lt_of_le column.isLt (le_trans (SlotLayout.publicFits _) wideFits)
  have same : (PaperAlgebra.FullShape logicalWidth publicFits).publicColumn column =
      RingCarrierLayout.embedLogical (⟨column.val, inside⟩ : Fin logicalWidth) := by
    apply Fin.ext
    rfl
  rw [RingRelation.projectPublicInput, same, RingCarrierLayout.extendAssignment_embedLogical]
  have narrow : column.val < ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (CircuitRelation.pieces circuit offset digestColumn) firstFresh batches)) :=
    Nat.lt_of_lt_of_le column.isLt (SlotLayout.publicFits _)
  simp only [WideLayout.extend, honestAssignment]
  rw [dite_eq_left narrow,
    show (⟨column.val, narrow⟩ : Fin _) =
      ProductionAssignment.publicColumn ⟨column.val, column.isLt⟩ from rfl,
    ProductionAssignment.logicalAssignment_publicColumn]
  rw [completeEnv_public packed publicBelow _ column.isLt]
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

/-- The operations of the light circuit at its shared end. -/
def lightOps (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : List Op :=
  Circuit.ops (LightStepLayout.main program relation 0
      (LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        StepRelation.base program.witnessWordCount))
    (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)

/-- The operations of the refresh circuit at its shared end. -/
def refreshOps (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : List Op :=
  Circuit.ops (RefreshStepLayout.main program relation 1
      (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits)
        StepRelation.base program.witnessWordCount))
    (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)

/-- The pieces that the emitter writes for a step circuit: the pieces of the circuit's
constraints, then of the decode constraints of its output digest. -/
def piecesOf (ops : List Op) (digestColumn : Fin 4 → Nat) : List Piece :=
  split (PilotData.canonicalConstraints ()) (marks ops)
    (flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn)

/-- The emitter's light pieces are the pieces of `StepRelation.light_stepHolds` and
`light_complete`, for every commitment key. -/
theorem piecesOf_lightOps (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    piecesOf (lightOps program relation) (StepRelation.lightDigestColumn program) =
      lightPieces program relation ajtai :=
  rfl

/-- The emitter's refresh pieces are the pieces of `StepRelation.refresh_stepHolds` and
`refresh_complete`, for every commitment key. -/
theorem piecesOf_refreshOps (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    piecesOf (refreshOps program relation) (StepRelation.refreshDigestColumn program) =
      refreshPieces program relation ajtai :=
  rfl

theorem recipeConstraints_last :
    ∀ (start count : Nat) (recipes : List Expr), recipes.length = count + 1 →
      ∃ rest, Expr.var (start + count) - rest ∈ recipeConstraints start recipes
  | _, _, [], length => by simp at length
  | start, 0, recipe :: _, _ => ⟨recipe, by simp [recipeConstraints]⟩
  | start, count + 1, _ :: recipes, length => by
      obtain ⟨rest, member⟩ := recipeConstraints_last (start + 1) count recipes
        (by simpa using length)
      refine ⟨rest, ?_⟩
      simp only [recipeConstraints, List.mem_cons]
      right
      rw [show start + (count + 1) = start + 1 + count by omega]
      exact member

/-- The last constraint of the permutation template starts with its last local, column
`16 + 1095`. -/
theorem canonical_last :
    ∃ rest, Expr.var (16 + 1095) - rest ∈ PilotData.canonicalConstraints () :=
  recipeConstraints_last 16 1095 _
    (Gadgets.Poseidon2.Permutation.compile_schedule_recipe_count 16 PilotData.canonicalState)

/-- The first constraint of the permutation template: its first local is the first lane of the
initial linear layer. -/
theorem canonical_head :
    Expr.var 16 - Gadgets.Poseidon2.Layer.externalE PilotData.canonicalState ⟨0, by decide⟩ ∈
      PilotData.canonicalConstraints () := by
  obtain ⟨tail, schedule⟩ : ∃ tail, Gadgets.Poseidon2.Permutation.schedule =
      Gadgets.Poseidon2.Permutation.Step.initialLayer :: tail := ⟨_, rfl⟩
  unfold PilotData.canonicalConstraints PilotData.canonicalRecipes
  rw [schedule]
  simp only [Gadgets.Poseidon2.Permutation.compile, Gadgets.Poseidon2.Permutation.stepRecipes,
    List.ofFn_succ, List.cons_append, recipeConstraints]
  exact List.mem_cons_self

/-- Each of the 16 lanes is read by a constraint of the permutation template. -/
theorem canonical_lanes : ∀ lane, lane < 16 → ∃ pattern ∈ PilotData.canonicalConstraints (),
    PieceRowsCompleteness.readsColumn lane pattern = true := by
  have reads : ∀ lane, lane < 16 → PieceRowsCompleteness.readsColumn lane
      (Expr.var 16 - Gadgets.Poseidon2.Layer.externalE PilotData.canonicalState ⟨0, by decide⟩) =
        true := by
    decide
  exact fun lane inside => ⟨_, canonical_head, reads lane inside⟩

theorem decodeFold_below (word : Fin 4) {bound : Nat} (bits : 257 ≤ bound) :
    ∀ (indices : List Nat) (initial : Expr), initial.VarsBelow bound →
      (indices.foldl (fun value bit => value + Expr.const (Poseidon2.ofNat (2 ^ bit)) *
        Expr.var (LogicalPlan.bitColumn word bit)) initial).VarsBelow bound
  | [], _, below => below
  | bit :: rest, initial, below => by
      simp only [List.foldl_cons]
      apply decodeFold_below word bits rest
      refine ⟨below, trivial, ?_⟩
      show LogicalPlan.bitColumn word bit < bound
      have := word.isLt
      have := Nat.mod_lt bit (show 64 > 0 by decide)
      simp only [LogicalPlan.bitColumn]
      omega

/-- The decode constraints read the digest columns and the bits of the public input. -/
theorem decodeConstraints_below (digestColumn : Fin 4 → Nat) {bound : Nat}
    (digest : ∀ word, digestColumn word < bound) (bits : 257 ≤ bound) :
    ∀ constraint ∈ LogicalPlan.decodeConstraints digestColumn, constraint.VarsBelow bound := by
  intro constraint member
  obtain ⟨word, rfl⟩ := List.mem_ofFn.mp member
  exact Expr.VarsBelow.sub _ _ _ (digest word) (decodeFold_below word bits _ _ trivial)

/-- Every light piece reads only columns before `firstFresh` when every row of the light circuit
does and `firstFresh` is after the shared words and the public columns. -/
theorem lightPieces_readOwnColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {firstFresh : Nat}
    (scope : ∀ constraint ∈ flatConstraints (lightOps program relation),
      constraint.VarsBelow firstFresh)
    (sharedBelow : LightAssembler.sharedEnd StepRelation.base program.witnessWordCount ≤ firstFresh)
    (publicBelow : publicColumns ≤ firstFresh) :
    (lightPieces program relation ajtai).all (pieceBelow firstFresh) = true := by
  rw [← piecesOf_lightOps program relation ajtai]
  refine split_below _ _ _ canonical_last canonical_lanes fun constraint member => ?_
  rcases List.mem_append.mp member with member | member
  · exact scope constraint member
  · refine decodeConstraints_below _ (fun word => ?_) (le_trans (by decide) publicBelow)
      constraint member
    have := word.isLt
    refine lt_of_lt_of_le ?_ sharedBelow
    simp only [StepRelation.lightDigestColumn, LightAssembler.sharedEnd,
      LightAssembler.digestSelectStart, LightAssembler.outputStart, LightAssembler.roundsStart,
      LightAssembler.freshStart]
    omega

/-- Every refresh piece reads only columns before `firstFresh` when every row of the refresh
circuit does and `firstFresh` is after the shared words and the public columns. -/
theorem refreshPieces_readOwnColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {firstFresh : Nat}
    (scope : ∀ constraint ∈ flatConstraints (refreshOps program relation),
      constraint.VarsBelow firstFresh)
    (sharedBelow :
      RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount ≤ firstFresh)
    (publicBelow : publicColumns ≤ firstFresh) :
    (refreshPieces program relation ajtai).all (pieceBelow firstFresh) = true := by
  rw [← piecesOf_refreshOps program relation ajtai]
  refine split_below _ _ _ canonical_last canonical_lanes fun constraint member => ?_
  rcases List.mem_append.mp member with member | member
  · exact scope constraint member
  · refine decodeConstraints_below _ (fun word => ?_) (le_trans (by decide) publicBelow)
      constraint member
    have := word.isLt
    refine lt_of_lt_of_le ?_ sharedBelow
    rw [RefreshAssembler.sharedEnd_eq]
    simp only [StepRelation.refreshDigestColumn, LightAssembler.digestsStart,
      LightAssembler.slotsStart, LightAssembler.witnessStart, LightAssembler.slotWordCount]
    omega

/-- An honest light step's assignment (`LightAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the light key,
with a witness that is zero past the package's logical width. -/
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
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) output.x ⟨column, inside⟩)
    (firstFresh width : Nat)
    (fitsPackage : Fits publicColumns (lightPieces program relation ajtai) firstFresh width)
    (blocksPack : packable publicColumns (lightPieces program relation ajtai) = true)
    (readsOwnColumns : (lightPieces program relation ajtai).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (fitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (lightPieces program relation ajtai) firstFresh batches)) ≤
        logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fitsPackage
            (WideLayout.widen (SlotLayout.layout _) fitsKeyWidth)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => RingRelation.PiRLCAlgebra.Commitment.commit ajtai
              (RingCarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) output.x })
        (RingCarrierLayout.extendAssignment 0 assignment) ∧
      ∀ column : Fin (RingCarrierLayout.carrierWidth logicalWidth),
        ProductionAssignment.logicalWidth (SlotLayout.runSlots width
          (StepPackage.slotRuns publicColumns (lightPieces program relation ajtai) firstFresh
            batches)) ≤ column.val →
        RingCarrierLayout.extendAssignment 0 assignment column = 0 :=
  fresh_complete _ _ _ env holds output.x (digest_getD represents.outputDigest)
    (digestColumn_above program) publicEq firstFresh width fitsPackage blocksPack readsOwnColumns publicBelow batches
    fitsKeyWidth cubeFits ajtai

/-- An honest refresh step's assignment (`RefreshAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the refresh key,
with a witness that is zero past the package's logical width. -/
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
    (publicEq : ∀ column (inside : column < publicColumns),
      env column = encHash (publicFits := publicFits) output.x ⟨column, inside⟩)
    (firstFresh width : Nat)
    (fitsPackage : Fits publicColumns (refreshPieces program relation ajtai) firstFresh width)
    (blocksPack : packable publicColumns (refreshPieces program relation ajtai) = true)
    (readsOwnColumns : (refreshPieces program relation ajtai).all (pieceBelow firstFresh) = true)
    (publicBelow : publicColumns ≤ firstFresh) (batches : List WitnessBatch)
    (fitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots width
      (StepPackage.slotRuns publicColumns (refreshPieces program relation ajtai) firstFresh
        batches)) ≤ logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ assignment : Assignment F logicalWidth,
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns _ firstFresh width fitsPackage
            (WideLayout.widen (SlotLayout.layout _) fitsKeyWidth)).logicalRelation
              (publicFits := publicFits) cubeFits)
          { commitments := fun _ => RingRelation.PiRLCAlgebra.Commitment.commit ajtai
              (RingCarrierLayout.extendAssignment 0 assignment)
            publicInputs := fun _ => encHash (publicFits := publicFits) output.x })
        (RingCarrierLayout.extendAssignment 0 assignment) ∧
      ∀ column : Fin (RingCarrierLayout.carrierWidth logicalWidth),
        ProductionAssignment.logicalWidth (SlotLayout.runSlots width
          (StepPackage.slotRuns publicColumns (refreshPieces program relation ajtai) firstFresh
            batches)) ≤ column.val →
        RingCarrierLayout.extendAssignment 0 assignment column = 0 :=
  fresh_complete _ _ _ env holds output.x (digest_getD represents.outputDigest)
    (digestColumn_above program) publicEq firstFresh width fitsPackage blocksPack readsOwnColumns publicBelow batches
    fitsKeyWidth cubeFits ajtai

end Steps

end NightstreamFPrime.Export.BoundedDepth.StepCompleteness
