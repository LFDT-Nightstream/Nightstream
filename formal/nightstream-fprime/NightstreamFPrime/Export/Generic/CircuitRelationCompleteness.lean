import NightstreamFPrime.Export.Generic.NativeBlockCompleteness

/-! Completeness of a step circuit's logical relation (CW-5.8), the converse of
`CircuitRelation.rowsZero_implies_spec`. The prover fills the honest environment's fresh columns
(`filled`): when the blocks pack, the lowering's fresh columns of the pieces outside the blocks
(`PieceRowsCompleteness.fillPieces`), then the S-box columns
(`NativeBlockCompleteness.sboxFill`); otherwise the fresh columns of every piece. It encodes the
result in the slots (`assignment`). When every piece reads below the first fresh column and every
column that the relation reads has a slot (`Covers`), both branches of `CircuitRelation.plan` have
zero rows, and the assignment decodes to the honest environment on the public columns and on the
column of every slot below the first fresh column (`rowsZero_of_pieces`). A column that no row
reads needs no slot. From a circuit's rows, an output digest that its public columns encode and its
digest columns carry, this gives `rowsZero_of_holdsFlat`. -/

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

/-! ### The prover's assignment -/

/-- Each column that the relation reads below its last column is public or has a slot: when the
blocks pack, the rows of the other pieces, the blocks' lanes and output locals and the S-box
columns (`NativeBlockCompleteness.ReadsCovered`); otherwise the rows of every piece. -/
structure Covers {width : Nat} (slots : List (LowNormAssignment.Slot width)) (publicColumns : Nat)
    (pieces : List Piece) (next : Nat) : Prop where
  packed : packable publicColumns pieces = true →
    ReadsCovered slots (lowerPieces (PilotData.canonicalConstraints ()) (otherPieces pieces) next).1
      (blocksOf pieces) (sboxStart pieces next)
  unpacked : packable publicColumns pieces = false →
    ReadsCovered slots (lowerPieces (PilotData.canonicalConstraints ()) pieces next).1 []
      (lowerPieces (PilotData.canonicalConstraints ()) pieces next).2

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
first fresh column `next`, and every column that the relation reads has a slot, the prover's
assignment makes the plan's rows zero, for both branches of `CircuitRelation.plan`. It decodes to
`env` on the public columns and on the column of every slot below `next`. -/
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
      ∀ slot ∈ slots, slot.source.val < next →
        LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns pieces next env)
          slot.source.val = env slot.source.val := by
  have zeroPublic : 0 < LogicalPlan.publicCount := by decide
  have decodeSlot : ∀ honest : Env, ∀ slot ∈ slots,
      LogicalPlan.sourceEnv (layout slots) (encode slots honest) slot.source.val =
        honest slot.source.val :=
    fun honest slot member => sourceEnv_encode slots honest slot.source.isLt
      (Or.inr ⟨slot, member, rfl⟩)
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
    have blockFits := fits.blocks packed
    obtain ⟨sboxRows, rowsHold, publicEq⟩ := blocks_rows_complete slots
      (packable_framed packed) (sboxStart pieces next)
      (lowerPieces_rows_below canonical (otherPieces pieces) next othersBelow)
      (fun block member expression inside => Expr.VarsBelow.mono _
        (blocksBelow block member expression inside) nextLe)
      (le_trans publicBelow nextLe) blockFits.sboxColumns (covers.packed packed)
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
      fun slot member below => ?_⟩
    · unfold plan
      rw [dite_eq_left packed, Plan.append_rowsZero_iff]
      refine ⟨(LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows (layout slots) _ oneEq).mpr ?_,
        sboxRows⟩
      unfold relationRows
      rw [ite_eq_left packed]
      exact rowsHold
    · unfold blockAssignment
      rw [decodeSlot _ slot member, sboxFill_outside (Or.inl (by omega)), low _ below]
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
    have rowsBelow := lowerPieces_rows_below canonical pieces next below
    have cover := covers.unpacked notPacked
    have rowsAt : R1CS.RowsHold (LogicalPlan.sourceEnv (layout slots) (encode slots honest))
        (lowerPieces canonical pieces next).1 :=
      R1CS.rowsHold_of_agree _
        (fun column => column < (lowerPieces canonical pieces next).2 ∧ Slotted slots column) honest _
        (fun row member => row_scope (rowsBelow row member) fun column columnMember high =>
          cover column (by simpa using high) (Or.inl ⟨row, member, columnMember⟩))
        (fun column allowed => sourceEnv_encode slots honest (allowed.2.lt_width publicInside)
          allowed.2)
        loweredRows
    have oneEq : encode slots honest (layout slots).one = 1 := by
      rw [encode_one, low 0 (by omega)]
      exact one
    refine ⟨?_, fun column isPublic => ?_, fun slot member below => ?_⟩
    · unfold plan
      rw [dite_eq_right packed]
      refine (LogicalPlan.rowsZero_iff_rowsHold _ width fits.rows (layout slots) _ oneEq).mpr ?_
      unfold relationRows
      rw [ite_eq_right (by simpa using packed)]
      exact rowsAt
    · rw [sourceEnv_encode_public slots honest (by omega) isPublic, low column (by omega)]
    · rw [decodeSlot _ slot member, low _ below]

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
      ∀ slot ∈ slots, slot.source.val < next →
        LogicalPlan.sourceEnv (layout slots) (assignment slots publicColumns
          (split (PilotData.canonicalConstraints ()) constraints) next env) slot.source.val =
          env slot.source.val :=
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
      ∀ slot ∈ slots, slot.source.val < next →
        LogicalPlan.sourceEnv (layout slots) (assignment slots CircuitRelation.publicColumns
          (pieces circuit offset digestColumn) next env) slot.source.val =
          env slot.source.val := by
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
