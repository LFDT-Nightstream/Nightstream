import NightstreamFPrime.Export.Generic.CircuitRelationCompleteness

/-! The norm bound of the prover's assignment (CW-5.11). Π_CCS checks that each fresh witness
coordinate lies in `{-1, 0, 1}` (`LogicalPlan.Layout`). The prover's assignment
(`CircuitRelationCompleteness.assignment`) changes the honest environment only from the first
fresh column (`filled_below`), so it lies within the norm bound when the honest public columns do
and each bit or centered slot reads a column below the first fresh column whose honest value is
valid for the slot's kind (`SlotsValid`, `assignment_norm`). Each column of `encHash` holds 0 or 1
(`encHash_bit`), so a slot of any kind is valid on a public column of an honest step.

A key can have more logical columns than a step's slots. `paddedSlots` adds slots that read the
public column 269, which `encHash` sets to 0 (`encHash_tail`): the padded encoding is zero past
the slots' coordinates, as such a key expects (`paddedSlots_logicalWidth`).

A bit slot needs no premise about the honest value when a row forces its column to 0 or 1: where
the rows hold, the column holds 0 or 1 (`encode_norm_of_forced`). -/

namespace NightstreamFPrime.Export.Generic.CircuitRelationNorm

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.NativeBlockCompleteness
open NightstreamFPrime.Export.Generic.SlotLayoutCompleteness
open NightstreamFPrime.Export.Generic.CircuitRelationCompleteness
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### The filled environment below the first fresh column -/

/-- The filled environment agrees with the honest one below the first fresh column. -/
theorem filled_below (publicColumns : Nat) (pieces : List Piece) (next : Nat) (env : Env)
    {column : Nat} (below : column < next) :
    filled publicColumns pieces next env column = env column := by
  unfold filled
  split
  · have nextLe : next ≤ sboxStart pieces next :=
      lowerPiecesAux_le (PilotData.canonicalConstraints ()) (otherPieces pieces) next []
    rw [sboxFill_outside (Or.inl (lt_of_lt_of_le below nextLe))]
    exact fillPieces_outside _ _ env next (Or.inl below)
  · exact fillPieces_outside _ _ env next (Or.inl below)

/-! ### The norm bound -/

/-- Each slot other than a field slot reads a column below `next` whose value in `env` is valid
for the slot's kind. -/
def SlotsValid {width : Nat} (slots : List (LowNormAssignment.Slot width)) (next : Nat)
    (env : Env) : Prop :=
  ∀ slot ∈ slots, slot.kind ≠ .field →
    slot.source.val < next ∧ LowNormSlot.Valid slot.kind (env slot.source.val)

/-- The prover's assignment lies within the norm bound when the honest public columns do and the
slots are valid for the honest environment. -/
theorem assignment_norm {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (publicColumns : Nat) (pieces : List Piece) (next : Nat) (env : Env)
    (publicBelow : LogicalPlan.publicCount ≤ next)
    (publicNorm : ∀ column, column < LogicalPlan.publicCount →
      centeredMagnitude (env column) < 2)
    (valid : SlotsValid slots next env) (column : Fin (ProductionAssignment.logicalWidth slots)) :
    centeredMagnitude (assignment slots publicColumns pieces next env column) < 2 := by
  apply encode_norm slots
  · intro cell
    have isPublic : cell.val < LogicalPlan.publicCount := cell.isLt
    rw [filled_below publicColumns pieces next env (lt_of_lt_of_le isPublic publicBelow)]
    exact publicNorm cell.val isPublic
  · intro slot member
    by_cases field : slot.kind = .field
    · rw [field]
      trivial
    · obtain ⟨below, slotValid⟩ := valid slot member field
      rw [filled_below publicColumns pieces next env below]
      exact slotValid

/-! ### Slots that read the public input -/

section Public

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Each column of `encHash` holds 0 or 1: the marker, a digest bit, or a tail zero. -/
theorem encHash_bit (digest : Digest) (column : Fin (ringDegree * PaperAlgebra.publicRingColumns)) :
    LowNormSlot.Valid .bit (encHash (publicFits := publicFits) digest column) := by
  obtain ⟨index, bound⟩ := column
  show encodedHashCells digest ⟨index, bound⟩ = 0 ∨
    encodedHashCells digest ⟨index, bound⟩ = 1
  unfold encodedHashCells
  cases index with
  | zero => exact Or.inr rfl
  | succ index =>
      dsimp only
      rw [List.getD_cons_succ]
      by_cases bitRegion : index < 256
      · rw [List.getD_eq_getElem (l := serializeDigestBits digest) (d := 0)
          (by simpa using bitRegion)]
        simp only [serializeDigestBits, List.getElem_map, List.getElem_range, digestBitWord]
        rcases Nat.mod_two_eq_zero_or_one
            ((digest.getD (index / 64) 0).val / 2 ^ (index % 64)) with bit | bit <;>
          rw [bit]
        · exact Or.inl rfl
        · exact Or.inr rfl
      · left
        apply List.getD_eq_default
        simp only [serializeDigestBits_length]
        omega

/-- A public column whose values encode a digest holds a value valid for every kind of slot. -/
theorem encoded_valid {env : Env} {digest : Digest}
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    {column : Nat} (isPublic : column < LogicalPlan.publicCount) (kind : LowNormSlot.Kind) :
    LowNormSlot.Valid kind (env column) := by
  have value : env column = encHash (publicFits := publicFits) digest ⟨column, isPublic⟩ :=
    congrFun encoded ⟨column, isPublic⟩
  rw [value]
  cases kind with
  | bit => exact encHash_bit (publicFits := publicFits) digest _
  | centered => exact encHash_norm (publicFits := publicFits) digest _
  | field => trivial

end Public

/-! ### Padding to a key width -/

/-- A bit slot that reads the public column 269, which `encHash` sets to 0. -/
def zeroSlot {width : Nat} (inside : LogicalPlan.publicCount ≤ width) :
    LowNormAssignment.Slot width :=
  ⟨⟨LogicalPlan.publicCount - 1, by
    have positive : 0 < LogicalPlan.publicCount := by decide
    omega⟩, .bit⟩

/-- `slots` followed by `count` zero slots. -/
def paddedSlots {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat) :
    List (LowNormAssignment.Slot width) :=
  slots ++ List.replicate count (zeroSlot inside)

theorem paddedSlots_logicalWidth {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat) :
    ProductionAssignment.logicalWidth (paddedSlots slots inside count) =
      ProductionAssignment.logicalWidth slots + count := by
  have zeroWidth : (zeroSlot inside).width = 1 := rfl
  simp only [ProductionAssignment.logicalWidth, LowNormAssignment.logicalWidth, paddedSlots,
    List.map_append, List.sum_append, List.map_replicate, zeroWidth, List.sum_replicate,
    smul_eq_mul, Nat.mul_one, Nat.add_assoc]

theorem paddedSlots_covers {width : Nat} {slots : List (LowNormAssignment.Slot width)}
    {publicColumns : Nat} {pieces : List Piece} {next : Nat}
    (covers : Covers slots publicColumns pieces next) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat) : Covers (paddedSlots slots inside count) publicColumns pieces next :=
  ⟨fun packed => (covers.packed packed).mono fun _ member => List.mem_append_left _ member,
    fun notPacked => (covers.unpacked notPacked).mono fun _ member => List.mem_append_left _ member⟩

/-- The zero slots are valid for an environment that is 0 at column 269, below `next`. -/
theorem paddedSlots_valid {width : Nat} {slots : List (LowNormAssignment.Slot width)} {next : Nat}
    {env : Env} (valid : SlotsValid slots next env) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat) (publicBelow : LogicalPlan.publicCount ≤ next)
    (zero : env (LogicalPlan.publicCount - 1) = 0) :
    SlotsValid (paddedSlots slots inside count) next env := by
  intro slot member nonField
  rcases List.mem_append.mp member with member | member
  · exact valid slot member nonField
  · rw [List.eq_of_mem_replicate member]
    have positive : 0 < LogicalPlan.publicCount := by decide
    refine ⟨by show LogicalPlan.publicCount - 1 < next; omega, ?_⟩
    show env (LogicalPlan.publicCount - 1) = 0 ∨ env (LogicalPlan.publicCount - 1) = 1
    exact Or.inl zero

/-! ### Bit slots that a row forces -/

/-- Where the plan's rows are zero, so are the rows of `CircuitRelation.relationRows` alone. -/
theorem relationRows_of_rowsZero {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (fits : Fits publicColumns pieces next width) {logicalWidth : Nat}
    (layout : LogicalPlan.Layout width logicalWidth) (assignment : Assignment F logicalWidth)
    (rowsZero : (plan publicColumns pieces next width fits layout).RowsZero assignment) :
    (LogicalPlan.plan (relationRows publicColumns pieces next width) width fits.rows
      layout).RowsZero assignment := by
  unfold plan at rowsZero
  by_cases packed : packable publicColumns pieces = true
  · rw [dite_eq_left packed, Plan.append_rowsZero_iff] at rowsZero
    exact rowsZero.1
  · rw [dite_eq_right packed] at rowsZero
    exact rowsZero

/-- An encoding whose ordinary plan rows are zero is in the norm bound when each public value is
valid for every kind of slot and each slot other than a field slot reads a public column or is a
bit slot whose column a row forces to 0 or 1 (`NativeBlock.forced_valid`). -/
theorem encode_norm_of_forced {width : Nat} (rows : List R1CS.Row)
    (fits : LogicalPlan.Fits rows width) (slots : List (LowNormAssignment.Slot width)) (env : Env)
    (rowsZero : (LogicalPlan.plan rows width fits (SlotLayout.layout slots)).RowsZero (encode slots env))
    (one : env 0 = 1)
    (publicValid : ∀ column, column < LogicalPlan.publicCount →
      ∀ kind, LowNormSlot.Valid kind (env column))
    (kinds : ∀ slot ∈ slots, slot.kind ≠ .field → slot.source.val < LogicalPlan.publicCount ∨
      (slot.kind = .bit ∧ ∃ row ∈ rows, SlotLayout.forcedColumn? row = some slot.source.val))
    (column : Fin (ProductionAssignment.logicalWidth slots)) :
    centeredMagnitude (encode slots env column) < 2 := by
  have oneEq : encode slots env (SlotLayout.layout slots).one = 1 := by
    rw [encode_one]
    exact one
  have rowsHold := (LogicalPlan.rowsZero_iff_rowsHold rows width fits (SlotLayout.layout slots) _ oneEq).mp
    rowsZero
  have sourceOne : LogicalPlan.sourceEnv (SlotLayout.layout slots) (encode slots env) 0 = 1 := by
    rw [sourceEnv_encode_public slots env fits.positive (by decide)]
    exact one
  apply encode_norm slots env fun cell => publicValid cell.val cell.isLt .centered
  intro slot member
  by_cases field : slot.kind = .field
  · rw [field]
    trivial
  · rcases kinds slot member field with isPublic | ⟨bit, row, rowMember, forced⟩
    · exact publicValid _ isPublic slot.kind
    · have valid := NativeBlock.forced_valid forced sourceOne (rowsHold row rowMember)
      rw [sourceEnv_encode slots env slot.source.isLt (Or.inr ⟨slot, member, rfl⟩)] at valid
      rw [bit]
      exact valid

end NightstreamFPrime.Export.Generic.CircuitRelationNorm
