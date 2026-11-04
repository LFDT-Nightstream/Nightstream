import NightstreamFPrime.Export.Generic.NativeBlockCompleteness

/-! Completeness of a step circuit's logical relation (CW-5.8), the converse of
`CircuitRelation.rowsZero_implies_spec`. The prover fills the honest environment's fresh columns
(`filled`): when the blocks pack, the lowering's fresh columns of the pieces outside the blocks
(`PieceRowsCompleteness.fillPieces`), then the S-box columns
(`NativeBlockCompleteness.sboxFill`); otherwise the fresh columns of every piece. It encodes the
result in the slots (`assignment`). When every piece reads below the first fresh column and every
column of the relation other than a dead local has a slot (`Covers`), both branches of
`CircuitRelation.plan` have zero rows, and the assignment decodes to the honest environment on the
public columns and on every live column below the first fresh column (`rowsZero_of_pieces`). From
a circuit's rows, an output digest that its public columns encode and its digest columns carry,
this gives `rowsZero_of_holdsFlat`. -/

namespace NightstreamFPrime.Export.Generic.CircuitRelationCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.NativeBlock
open NightstreamFPrime.Export.Generic.NativeBlockCompleteness
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### Blocks of the pieces -/

theorem blocksOf_mem : ∀ {pieces : List Piece} {block : Block}, block ∈ blocksOf pieces →
    Piece.permutation block.start block.lanes ∈ pieces
  | [], _, member => by simp [blocksOf] at member
  | .permutation start lanes :: rest, block, member => by
      simp only [blocksOf, List.mem_cons] at member
      rcases member with rfl | member
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (blocksOf_mem member)
  | .constraint _ :: rest, _, member => List.mem_cons_of_mem _ (blocksOf_mem member)
  | .ringProduct _ _ _ :: rest, _, member => List.mem_cons_of_mem _ (blocksOf_mem member)

theorem otherPieces_mem {pieces : List Piece} {piece : Piece} (member : piece ∈ otherPieces pieces) :
    piece ∈ pieces ∧ isBlock piece = false := by
  have filtered := List.mem_filter.mp member
  exact ⟨filtered.1, by simpa using filtered.2⟩

/-- A piece of `split` gives constraints of the split list. -/
theorem split_mem {canonical constraints : List Expr} {piece : Piece}
    (member : piece ∈ split canonical constraints) {expression : Expr}
    (inPiece : expression ∈ piece.constraints canonical) : expression ∈ constraints := by
  rw [← flatMap_split canonical constraints]
  exact List.mem_flatMap.mpr ⟨piece, member, inPiece⟩

/-! ### Lowered rows that read no dead local -/

theorem avoids_varsSatisfy {dead : Nat → Bool} :
    ∀ expression : Expr, avoids dead expression = true →
      expression.VarsSatisfy fun column => dead column = false
  | .var column, avoid => by
      show dead column = false
      simpa [avoids] using avoid
  | .const _, _ => trivial
  | .add left right, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      exact ⟨avoids_varsSatisfy left avoid.1, avoids_varsSatisfy right avoid.2⟩
  | .mul left right, avoid => by
      simp only [avoids, Bool.and_eq_true] at avoid
      exact ⟨avoids_varsSatisfy left avoid.1, avoids_varsSatisfy right avoid.2⟩

theorem rowAvoids_of_varsSatisfy {dead : Nat → Bool} {row : R1CS.Row}
    (support : row.VarsSatisfy fun column => dead column = false) : rowAvoids dead row = true := by
  simp only [rowAvoids, combinationAvoids, Bool.and_eq_true, List.all_eq_true]
  refine ⟨⟨fun term member => ?_, fun term member => ?_⟩, fun term member => ?_⟩
  · simpa using support.1 term member
  · simpa using support.2.1 term member
  · simpa using support.2.2 term member

theorem getD_avoids {dead : Nat → Bool} {lanes : List Expr}
    (avoid : ∀ lane ∈ lanes, avoids dead lane = true) (index : Nat) :
    avoids dead (lanes.getD index 0) = true := by
  by_cases inside : index < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact avoid _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

/-- The rows of a piece outside the blocks that reads no dead local read no dead local, when the
dead locals lie below the piece's fresh columns. -/
theorem lowerPiece_avoid {dead : Nat → Bool} (piece : Piece) (next : Nat)
    (notBlock : isBlock piece = false) (avoid : avoidsDead dead piece = true)
    (deadBelow : ∀ column, dead column = true → column < next) :
    ∀ row ∈ (lowerPiece (PilotData.canonicalConstraints ()) piece next).1,
      rowAvoids dead row = true := by
  intro row member
  apply rowAvoids_of_varsSatisfy
  have fresh : ∀ column, next ≤ column → dead column = false := by
    intro column high
    cases isDead : dead column
    · rfl
    · exact absurd (deadBelow column isDead) (by omega)
  cases piece with
  | permutation => simp [isBlock] at notBlock
  | constraint expression =>
      have scope := R1CS.lowerConstraints_rows_varsSatisfy [expression] next _ (fun current inside => by
        simp only [List.mem_singleton] at inside
        subst inside
        exact avoids_varsSatisfy _ avoid) row member
      exact scope.mono row fun column source => by
        rcases source with live | ⟨high, _⟩
        · exact live
        · exact fresh column high
  | ringProduct cells start lanes =>
      simp only [avoidsDead, Bool.and_eq_true, List.all_eq_true, List.mem_range] at avoid
      have laneAvoid := fun index => avoids_varsSatisfy _ (getD_avoids avoid.1 index)
      exact ringRows_varsSatisfy cells start next lanes (fun lane => laneAvoid _)
        (fun _ _ lane => laneAvoid _) (fun _ _ lane => laneAvoid _)
        (fun cell inside lane => by simpa [outputColumn] using avoid.2 _ (slot_lt lane inside))
        (fun cell inside lane => fresh _ (by simp only [quotientColumn]; omega)) row member

theorem lowerPiecesAux_avoid {dead : Nat → Bool} {bound : Nat}
    (deadBelow : ∀ column, dead column = true → column < bound) :
    ∀ (pieces : List Piece) (next : Nat) (rowsReversed : List R1CS.Row), bound ≤ next →
      (∀ piece ∈ pieces, isBlock piece = false ∧ avoidsDead dead piece = true) →
      (∀ row ∈ rowsReversed, rowAvoids dead row = true) →
      ∀ row ∈ (lowerPiecesAux (PilotData.canonicalConstraints ()) pieces next rowsReversed).1,
        rowAvoids dead row = true
  | [], _, rowsReversed, _, _, rows => by
      intro row member
      exact rows row (by simpa [lowerPiecesAux] using member)
  | piece :: rest, next, rowsReversed, le, pieces, rows => by
      simp only [lowerPiecesAux]
      apply lowerPiecesAux_avoid deadBelow rest _ _
        (le_trans le (lowerPiece_next _ piece next))
        (fun other member => pieces other (List.mem_cons_of_mem _ member))
      intro row member
      rw [List.reverseAux_eq, List.mem_append, List.mem_reverse] at member
      rcases member with member | member
      · exact lowerPiece_avoid piece next (pieces _ List.mem_cons_self).1
          (pieces _ List.mem_cons_self).2 (fun column isDead => by
            have := deadBelow column isDead
            omega) row member
      · exact rows row member

/-! ### The prover's assignment -/

/-- The columns the slots need not cover: the dead locals when the blocks pack. -/
def deadColumn (publicColumns : Nat) (pieces : List Piece) (column : Nat) : Bool :=
  packable publicColumns pieces && deadIn (blocksOf pieces) column

/-- The first column after the relation's columns: after the S-box columns when the blocks pack,
after the lowering's fresh columns otherwise. -/
def columnEnd (publicColumns : Nat) (pieces : List Piece) (next : Nat) : Nat :=
  if packable publicColumns pieces then sboxStart pieces next + (blocksOf pieces).length * 150
  else (lowerPieces (PilotData.canonicalConstraints ()) pieces next).2

/-- Every column from the public columns to `columnEnd`, other than a dead local, has a slot. -/
def Covers {width : Nat} (slots : List (LowNormAssignment.Slot width)) (publicColumns : Nat)
    (pieces : List Piece) (next : Nat) : Prop :=
  ∀ column, LogicalPlan.publicCount ≤ column → column < columnEnd publicColumns pieces next →
    deadColumn publicColumns pieces column = false → ∃ slot ∈ slots, slot.source.val = column

/-- The honest environment with its fresh columns filled. -/
def filled (publicColumns : Nat) (pieces : List Piece) (next : Nat) (env : Env) : Env :=
  if packable publicColumns pieces then
    sboxFill (blocksOf pieces) (sboxStart pieces next)
      (fillPieces (PilotData.canonicalConstraints ()) env (otherPieces pieces) next)
  else fillPieces (PilotData.canonicalConstraints ()) env pieces next

/-- The prover's assignment: the filled environment encoded in the slots. -/
def assignment {width : Nat} (slots : List (LowNormAssignment.Slot width)) (publicColumns : Nat)
    (pieces : List Piece) (next : Nat) (env : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  encode slots (filled publicColumns pieces next env)

theorem publicInside_of_packed {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (fits : Fits publicColumns pieces next width) (packed : packable publicColumns pieces = true)
    (publicBelow : LogicalPlan.publicCount ≤ next) : LogicalPlan.publicCount ≤ width := by
  have sboxColumns := (fits.blocks packed).sboxColumns
  have nextLe : next ≤ sboxStart pieces next :=
    lowerPiecesAux_le (PilotData.canonicalConstraints ()) (otherPieces pieces) next []
  omega

/-! ### Completeness -/

/-- Plan completeness on pieces: when every piece's constraints hold in `env` and read below the
first fresh column `next`, and every column of the relation other than a dead local has a slot,
the prover's assignment makes the plan's rows zero, for both branches of `CircuitRelation.plan`.
It decodes to `env` on the public columns and on every live column below `next`. -/
theorem rowsZero_of_pieces (publicColumns : Nat) (pieces : List Piece)
    (valid : ∀ piece ∈ pieces, piece.Valid) (next width : Nat)
    (fits : Fits publicColumns pieces next width) (slots : List (LowNormAssignment.Slot width))
    (covers : Covers slots publicColumns pieces next)
    (publicBelow : LogicalPlan.publicCount ≤ next)
    (publicInside : LogicalPlan.publicCount ≤ width) (env : Env) (one : env 0 = 1)
    (below : ∀ piece ∈ pieces, ∀ expression ∈ piece.constraints (PilotData.canonicalConstraints ()),
      expression.VarsBelow next)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    (plan publicColumns pieces next width fits (layout slots)).RowsZero
        (assignment slots publicColumns pieces next env) ∧
      (∀ column, column < LogicalPlan.publicCount →
        LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns pieces next env)
          column = env column) ∧
      ∀ column, column < next → column < width →
        deadColumn publicColumns pieces column = false →
          LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns pieces next env)
            column = env column := by
  have zeroPublic : 0 < LogicalPlan.publicCount := by decide
  by_cases packed : packable publicColumns pieces = true
  · let canonical := PilotData.canonicalConstraints ()
    let blocks := blocksOf pieces
    let honest := fillPieces canonical env (otherPieces pieces) next
    have othersValid : ∀ piece ∈ otherPieces pieces, piece.Valid :=
      fun piece member => valid piece (otherPieces_mem member).1
    have othersBelow : ∀ piece ∈ otherPieces pieces, ∀ expression ∈ piece.constraints canonical,
        expression.VarsBelow next := fun piece member => below piece (otherPieces_mem member).1
    have othersHold : ∀ piece ∈ otherPieces pieces, ConstraintsHold env (piece.constraints canonical) :=
      fun piece member => holds piece (otherPieces_mem member).1
    obtain ⟨loweredRows, agree⟩ := lowerPieces_complete canonical (otherPieces pieces) next env
      othersValid othersBelow othersHold
    have low : ∀ column, column < next → honest column = env column :=
      fun column below => agree column (Or.inl below)
    have blocksBelow : ∀ block ∈ blocks, ∀ expression ∈
        (Piece.permutation block.start block.lanes).constraints canonical,
        expression.VarsBelow next := fun block member => below _ (blocksOf_mem member)
    have nextLe : next ≤ sboxStart pieces next :=
      lowerPiecesAux_le canonical (otherPieces pieces) next []
    have check := packed
    simp only [packable, Bool.and_eq_true, List.all_eq_true] at check
    have avoid : (lowerPieces canonical (otherPieces pieces) next).1.all
        (rowAvoids (deadIn blocks)) = true :=
      List.all_eq_true.mpr (lowerPiecesAux_avoid (fun column dead => dead_below blocksBelow dead)
        (otherPieces pieces) next [] (le_refl _)
        (fun piece member => ⟨(otherPieces_mem member).2,
          check.2 piece (otherPieces_mem member).1⟩)
        (fun _ member => by simp at member))
    have blockFits := fits.blocks packed
    have endEq : columnEnd publicColumns pieces next = sboxStart pieces next + blocks.length * 150 := by
      unfold columnEnd
      rw [ite_eq_left packed]
    have deadEq : ∀ column, deadColumn publicColumns pieces column = deadIn blocks column := by
      intro column
      simp only [deadColumn, packed, Bool.true_and, blocks]
    obtain ⟨sboxRows, rowsHold, publicEq, decode⟩ := blocks_rows_complete slots
      (packable_framed packed) (sboxStart pieces next) avoid
      (lowerPieces_rows_below canonical (otherPieces pieces) next othersBelow)
      (fun block member expression inside => Expr.VarsBelow.mono _
        (blocksBelow block member expression inside) nextLe)
      (le_trans publicBelow nextLe) blockFits.sboxColumns
      (fun column notPublic high live => covers column notPublic (by rw [endEq]; exact high)
        (by rw [deadEq]; exact live))
      (by have := blockFits.rowCount; omega) honest (by rw [low 0 (by omega)]; exact one)
      loweredRows (fun block member => constraintsHold_of_agree_below env honest _ next
        (blocksBelow block member) low (holds _ (blocksOf_mem member)))
    have assignmentEq : assignment slots publicColumns pieces next env =
        blockAssignment slots blocks (sboxStart pieces next) honest := by
      unfold assignment filled blockAssignment
      rw [ite_eq_left packed]
    rw [assignmentEq]
    have oneEq : blockAssignment slots blocks (sboxStart pieces next) honest (layout slots).one = 1 := by
      unfold blockAssignment
      rw [encode_one, sboxFill_outside (Or.inl (by omega)), low 0 (by omega)]
      exact one
    refine ⟨?_, fun column isPublic => (publicEq column isPublic).trans (low column (by omega)),
      fun column below _ live => ?_⟩
    · unfold plan
      rw [dite_eq_left packed, Plan.append_rowsZero_iff]
      refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows (layout slots) _ oneEq).mpr ?_,
        sboxRows⟩
      unfold relationRows
      rw [ite_eq_left packed]
      exact rowsHold
    · rw [deadEq] at live
      rw [decode column (by omega) live, sboxFill_outside (Or.inl (by omega)), low column below]
  · have notPacked : packable publicColumns pieces = false := by simpa using packed
    let canonical := PilotData.canonicalConstraints ()
    let honest := fillPieces canonical env pieces next
    obtain ⟨loweredRows, agree⟩ := lowerPieces_complete canonical pieces next env valid below holds
    have low : ∀ column, column < next → honest column = env column :=
      fun column below => agree column (Or.inl below)
    have nextLe : next ≤ (lowerPieces canonical pieces next).2 :=
      lowerPiecesAux_le canonical pieces next []
    have assignmentEq : assignment slots publicColumns pieces next env = encode slots honest := by
      unfold assignment filled
      rw [ite_eq_right (by simpa using packed)]
    rw [assignmentEq]
    have decode : ∀ column, column < width → column < (lowerPieces canonical pieces next).2 →
        LogicalPlan.sourceEnv (layout slots) (encode slots honest) column = honest column := by
      intro column inside high
      apply sourceEnv_encode slots honest inside
      by_cases isPublic : column < LogicalPlan.publicCount
      · exact Or.inl isPublic
      · refine Or.inr (covers column (Nat.le_of_not_gt isPublic) ?_ ?_)
        · unfold columnEnd
          rw [ite_eq_right (by simpa using packed)]
          exact high
        · simp [deadColumn, notPacked]
    have rowsBelow := lowerPieces_rows_below canonical pieces next below
    have rowsAt : R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots) (encode slots honest))
        (lowerPieces canonical pieces next).1 := by
      apply R1CS.rowsHold_of_agree _
        (fun column => column < width ∧ column < (lowerPieces canonical pieces next).2) honest
      · intro row member
        have bounded := fits.rows.rowBounded (show row ∈ relationRows publicColumns pieces next width by
          unfold relationRows
          rw [ite_eq_right (by simpa using packed)]
          exact member)
        have rowBelow := rowsBelow row member
        exact ⟨fun term inside => ⟨bounded.1 term inside, rowBelow.1 term inside⟩,
          fun term inside => ⟨bounded.2.1 term inside, rowBelow.2.1 term inside⟩,
          fun term inside => ⟨bounded.2.2 term inside, rowBelow.2.2 term inside⟩⟩
      · intro column allowed
        exact decode column allowed.1 allowed.2
      · exact loweredRows
    have oneEq : encode slots honest (layout slots).one = 1 := by
      rw [encode_one, low 0 (by omega)]
      exact one
    refine ⟨?_, fun column isPublic => ?_, fun column below inside _ => ?_⟩
    · unfold plan
      rw [dite_eq_right packed]
      refine (LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows (layout slots) _ oneEq).mpr ?_
      unfold relationRows
      rw [ite_eq_right (by simpa using packed)]
      exact rowsAt
    · rw [sourceEnv_encode_public slots honest (by omega) isPublic, low column (by omega)]
    · rw [decode column inside (by omega), low column below]

/-- Plan completeness on the constraints that `split` cuts into the pieces. -/
theorem rowsZero_of_constraints (publicColumns : Nat) (constraints : List Expr) (next width : Nat)
    (fits : Fits publicColumns (split (PilotData.canonicalConstraints ()) constraints) next width)
    (slots : List (LowNormAssignment.Slot width))
    (covers : Covers slots publicColumns (split (PilotData.canonicalConstraints ()) constraints) next)
    (publicBelow : LogicalPlan.publicCount ≤ next)
    (publicInside : LogicalPlan.publicCount ≤ width) (env : Env) (one : env 0 = 1)
    (below : ∀ expression ∈ constraints, expression.VarsBelow next)
    (holds : ConstraintsHold env constraints) :
    (plan publicColumns _ next width fits (layout slots)).RowsZero
        (assignment slots publicColumns (split (PilotData.canonicalConstraints ()) constraints)
          next env) ∧
      (∀ column, column < LogicalPlan.publicCount →
        LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns
          (split (PilotData.canonicalConstraints ()) constraints) next env) column = env column) ∧
      ∀ column, column < next → column < width →
        deadColumn publicColumns (split (PilotData.canonicalConstraints ()) constraints) column =
          false →
          LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns
            (split (PilotData.canonicalConstraints ()) constraints) next env) column =
            env column :=
  rowsZero_of_pieces publicColumns _ (valid_split _ constraints) next width fits slots covers
    publicBelow publicInside env one
    (fun _ member _ inside => below _ (split_mem member inside))
    (fun _ member _ inside => holds _ (split_mem member inside))

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The decode constraints hold when the public columns encode `digest` and the digest columns
carry it. -/
theorem decodeConstraints_hold (env : Env) (digestColumn : Fin 4 → Nat) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    (words : ∀ word, env (digestColumn word) = digest.getD word.val 0) :
    ConstraintsHold env (LogicalPlan.decodeConstraints digestColumn) := by
  intro expression member
  obtain ⟨word, rfl⟩ := List.mem_ofFn.mp member
  rw [LogicalPlan.decodeConstraint, Expr.eval_sub, Expr.eval_var,
    LogicalPlan.decodeExpr_eval (publicFits := publicFits), encoded, decodeHashWord_encHash, words,
    sub_self]

theorem decodeExpr_below (word : Fin 4) {bound : Nat} (inside : 257 ≤ bound) :
    (LogicalPlan.decodeExpr word).VarsBelow bound := by
  unfold LogicalPlan.decodeExpr
  have step : ∀ (bits : List Nat) (initial : Expr), initial.VarsBelow bound →
      (bits.foldl (fun value bit => value + Expr.const (Poseidon2.ofNat (2 ^ bit)) *
        Expr.var (LogicalPlan.bitColumn word bit)) initial).VarsBelow bound := by
    intro bits
    induction bits with
    | nil => exact fun _ below => below
    | cons bit rest ih =>
        intro initial below
        apply ih
        refine ⟨below, trivial, ?_⟩
        show LogicalPlan.bitColumn word bit < bound
        have := word.isLt
        have := Nat.mod_lt bit (show 0 < 64 by decide)
        simp only [LogicalPlan.bitColumn]
        omega
  exact step _ _ trivial

/-- The decode constraints read the digest columns and the public columns 1 to 256. -/
theorem decodeConstraints_below (digestColumn : Fin 4 → Nat) {bound : Nat}
    (digestBelow : ∀ word, digestColumn word < bound) (inside : 257 ≤ bound) :
    ∀ expression ∈ LogicalPlan.decodeConstraints digestColumn, expression.VarsBelow bound := by
  intro expression member
  obtain ⟨word, rfl⟩ := List.mem_ofFn.mp member
  exact Expr.VarsBelow.sub _ _ _ (digestBelow word) (decodeExpr_below word inside)

theorem one_of_encoded (env : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest) :
    env 0 = 1 := by
  have marker := congrFun encoded encHashMarkerIndex
  rw [encHash_marker] at marker
  exact marker

/-- Plan completeness of a step circuit, the converse of `CircuitRelation.rowsZero_implies_spec`:
the circuit's rows in an honest environment whose public columns encode `digest` and whose digest
columns carry it, with every constraint below the first fresh column `next`, give the prover's
assignment, whose plan rows are zero and whose public input encodes `digest`. -/
theorem rowsZero_of_holdsFlat (circuit : FormalCircuit) (offset next : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat}
    (fits : Fits CircuitRelation.publicColumns (pieces circuit offset digestColumn) next width)
    (slots : List (LowNormAssignment.Slot width))
    (covers : Covers slots CircuitRelation.publicColumns (pieces circuit offset digestColumn) next)
    (publicBelow : LogicalPlan.publicCount ≤ next)
    (publicInside : LogicalPlan.publicCount ≤ width) (env : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    (words : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (below : ∀ expression ∈ CircuitRelation.constraints circuit offset digestColumn,
      expression.VarsBelow next)
    (rows : holdsFlat env (Circuit.ops circuit.main offset)) :
    (plan _ _ next width fits (layout slots)).RowsZero
        (assignment slots CircuitRelation.publicColumns (pieces circuit offset digestColumn) next
          env) ∧
      LogicalPlan.publicInput (publicFits := publicFits) (LogicalPlan.sourceEnv (layout slots)
          (assignment slots CircuitRelation.publicColumns (pieces circuit offset digestColumn) next
            env)) = encHash digest ∧
      ∀ column, column < next → column < width →
        deadColumn CircuitRelation.publicColumns (pieces circuit offset digestColumn) column =
          false →
          LogicalPlan.sourceEnv (layout slots) (assignment slots CircuitRelation.publicColumns
            (pieces circuit offset digestColumn) next env) column = env column := by
  have holds : ConstraintsHold env (CircuitRelation.constraints circuit offset digestColumn) :=
    (constraintsHold_append _ _ _).mpr ⟨rows,
      decodeConstraints_hold (publicFits := publicFits) env digestColumn digest encoded words⟩
  obtain ⟨rowsZero, publicEq, decode⟩ := rowsZero_of_constraints CircuitRelation.publicColumns
    (CircuitRelation.constraints circuit offset digestColumn) next width fits slots covers
    publicBelow publicInside env (one_of_encoded (publicFits := publicFits) env digest encoded)
    below holds
  refine ⟨rowsZero, ?_, decode⟩
  rw [← encoded]
  funext column
  exact publicEq column.val column.isLt

end NightstreamFPrime.Export.Generic.CircuitRelationCompleteness
