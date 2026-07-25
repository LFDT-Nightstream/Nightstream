import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkCanonicalRefinement

/-!
Contract: artifact-independent arbitrary-batch lifting of the selected plain
terminal latest-link row block.

Owns:
- one nonoptional typed receipt for every emitted row;
- one uniquely owned fresh public column per receipt;
- exact row and column costs for every batch size;
- soundness and completeness of the repeated 270-row relation.

Does not own: host nonemptiness and shape rejection, production column
placement, a generated multi-claim artifact, producer output-bit semantics,
or the optional application suffix.

The constant-one column and the 256 producer `x_out` bit columns are external
inputs to this block.  Each claim contributes exactly 270 public columns and
270 recurring rows.  The block allocates no committed or auxiliary columns.
-/

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

set_option maxRecDepth 32768

/-- Canonical four-lane digest value consumed by `enc_inst`. -/
abbrev LinkDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- One scalar obligation inside a claim's selected 270-row block. -/
inductive LocalOwner where
  | affineOne
  | linked (bit : Fin 256)
  | paddingZero (padding : Fin 13)
deriving Repr, DecidableEq

/-- Local physical row/column position of one obligation. -/
def LocalOwner.index : LocalOwner → Fin 270
  | .affineOne => ⟨0, by omega⟩
  | .linked bit => ⟨bit.val + 1, by have := bit.isLt; omega⟩
  | .paddingZero padding =>
      ⟨padding.val + 257, by have := padding.isLt; omega⟩

/-- Decode every local physical position into its unique obligation owner. -/
def ownerAt (position : Fin 270) : LocalOwner :=
  if zero : position.val = 0 then
    .affineOne
  else if linked : position.val ≤ 256 then
    .linked ⟨position.val - 1, by omega⟩
  else
    .paddingZero ⟨position.val - 257, by
      have := position.isLt
      omega⟩

theorem ownerAt_index (owner : LocalOwner) :
    ownerAt owner.index = owner := by
  cases owner with
  | affineOne =>
      rfl
  | linked bit =>
      simp only [ownerAt, LocalOwner.index]
      split
      · rename_i zero
        have := bit.isLt
        omega
      · split
        · rename_i linked
          congr
        · rename_i notLinked
          have := bit.isLt
          omega
  | paddingZero padding =>
      simp only [ownerAt, LocalOwner.index]
      split
      · rename_i zero
        have := padding.isLt
        omega
      · split
        · rename_i linked
          have := padding.isLt
          omega
        · congr

theorem index_ownerAt (position : Fin 270) :
    (ownerAt position).index = position := by
  apply Fin.ext
  simp only [ownerAt]
  split
  · rename_i zero
    simp [LocalOwner.index, zero]
  · rename_i nonzero
    split
    · rename_i linked
      simp only [LocalOwner.index]
      omega
    · rename_i notLinked
      simp only [LocalOwner.index]
      have := position.isLt
      omega

/-- Nonoptional row receipt: claim owner plus scalar obligation owner. -/
structure Receipt (batchSize : Nat) where
  claim : Fin batchSize
  owner : LocalOwner
deriving Repr, DecidableEq

theorem Receipt.eq_of_fields {batchSize : Nat}
    (left right : Receipt batchSize)
    (claim : left.claim = right.claim)
    (owner : left.owner = right.owner) :
    left = right := by
  cases left
  cases right
  cases claim
  cases owner
  rfl

/-- Physical row position owned by a receipt. -/
def Receipt.slot {batchSize : Nat}
    (receipt : Receipt batchSize) : Fin (batchSize * 270) :=
  ⟨receipt.claim.val * 270 + receipt.owner.index.val, by
    have claimLt := receipt.claim.isLt
    have ownerLt := receipt.owner.index.isLt
    omega⟩

/-- Receipt decoded from a physical row position. -/
def receiptAt {batchSize : Nat}
    (slot : Fin (batchSize * 270)) : Receipt batchSize where
  claim := ⟨slot.val / 270, by
    have slotLt := slot.isLt
    omega⟩
  owner := ownerAt ⟨slot.val % 270, Nat.mod_lt _ (by omega)⟩

theorem receiptAt_slot {batchSize : Nat}
    (receipt : Receipt batchSize) :
    receiptAt receipt.slot = receipt := by
  cases receipt with
  | mk claim owner =>
      apply Receipt.eq_of_fields
      · apply Fin.ext
        simp only [receiptAt, Receipt.slot]
        have ownerLt := owner.index.isLt
        omega
      · simp only [receiptAt, Receipt.slot]
        have positionEqual :
            (⟨(claim.val * 270 + owner.index.val) % 270,
              Nat.mod_lt _ (by omega)⟩ : Fin 270) =
              owner.index := by
          apply Fin.ext
          have ownerLt := owner.index.isLt
          simp [Nat.add_mod, Nat.mod_eq_of_lt ownerLt]
        rw [positionEqual]
        exact ownerAt_index owner

theorem slot_receiptAt {batchSize : Nat}
    (slot : Fin (batchSize * 270)) :
    (receiptAt slot).slot = slot := by
  apply Fin.ext
  simp only [receiptAt, Receipt.slot]
  rw [index_ownerAt]
  simpa [Nat.mul_comm] using Nat.div_add_mod slot.val 270

/-- First public column allocated for a claim. -/
def claimBase (claim : Nat) : Nat :=
  freshOneCol + claim * 270

def freshOneColAt (claim : Nat) : Nat :=
  claimBase claim

def freshBitColAt (claim bit : Nat) : Nat :=
  claimBase claim + 1 + bit

def freshPaddingColAt (claim padding : Nat) : Nat :=
  claimBase claim + 257 + padding

/-- The exact row emitted for one typed receipt. -/
def emit {batchSize : Nat} (receipt : Receipt batchSize) : Row :=
  match receipt.owner with
  | .affineOne =>
      ⟨[(freshOneColAt receipt.claim.val, 1), (0, goldilocksP - 1)],
       [(0, 1)], []⟩
  | .linked bit =>
      ⟨[(freshBitColAt receipt.claim.val bit.val, 1),
         (lastXOutBitCol bit.val, goldilocksP - 1)],
       [(0, 1)], []⟩
  | .paddingZero padding =>
      ⟨[(freshPaddingColAt receipt.claim.val padding.val, 1)],
       [(0, 1)], []⟩

/-- Canonical repeated physical row list in claim-major/local-major order. -/
def rows (batchSize : Nat) : List Row :=
  List.ofFn fun slot : Fin (batchSize * 270) =>
    emit (receiptAt slot)

def rowCount (batchSize : Nat) : Nat :=
  batchSize * 270

def publicColumnCount (batchSize : Nat) : Nat :=
  batchSize * 270

def committedColumnCount (_batchSize : Nat) : Nat :=
  0

def auxiliaryColumnCount (_batchSize : Nat) : Nat :=
  256

/-- Constant-one, shared producer bits, and claim-owned public columns. -/
def columnCount (batchSize : Nat) : Nat :=
  1 + auxiliaryColumnCount batchSize + publicColumnCount batchSize

theorem rows_length (batchSize : Nat) :
    (rows batchSize).length = rowCount batchSize := by
  simp [rows, rowCount]

theorem cost_conservation (batchSize : Nat) :
    columnCount batchSize =
      1 + auxiliaryColumnCount batchSize +
        committedColumnCount batchSize + publicColumnCount batchSize := by
  simp [columnCount, committedColumnCount]

/-- The one-claim specialization is exactly the current generated artifact,
not merely equicardinal with it. -/
theorem rows_one_eq_artifact :
    rows 1 = FPrimeTerminalLink.rows := by
  decide

/-- Physical row index owned by a receipt. -/
def physicalIndex {batchSize : Nat}
    (receipt : Receipt batchSize) : Fin (rows batchSize).length :=
  Fin.cast (rows_length batchSize).symm receipt.slot

theorem row_at_physicalIndex {batchSize : Nat}
    (receipt : Receipt batchSize) :
    (rows batchSize).get (physicalIndex receipt) = emit receipt := by
  simp only [rows, physicalIndex, List.get_eq_getElem,
    List.getElem_ofFn]
  congr
  exact receiptAt_slot receipt

/-- No two receipts own the same physical row position. -/
theorem physicalIndex_injective (batchSize : Nat) :
    Function.Injective
      (physicalIndex : Receipt batchSize → Fin (rows batchSize).length) := by
  intro left right equal
  have slotEqual : left.slot = right.slot := by
    apply Fin.ext
    simpa [physicalIndex] using congrArg Fin.val equal
  calc
    left = receiptAt left.slot := (receiptAt_slot left).symm
    _ = receiptAt right.slot := by rw [slotEqual]
    _ = right := receiptAt_slot right

/-- Every physical row position is owned by a receipt. -/
theorem physicalIndex_surjective (batchSize : Nat) :
    Function.Surjective
      (physicalIndex : Receipt batchSize → Fin (rows batchSize).length) := by
  intro index
  let slot : Fin (batchSize * 270) :=
    Fin.cast (rows_length batchSize) index
  refine ⟨receiptAt slot, ?_⟩
  apply Fin.ext
  change (receiptAt slot).slot.val = index.val
  rw [slot_receiptAt]
  rfl

/-- The public column allocated by a row receipt. Producer bits are external
read-only inputs and therefore are not claimed by this map. -/
def publicColumn {batchSize : Nat}
    (receipt : Receipt batchSize) : Nat :=
  claimBase receipt.claim.val + receipt.owner.index.val

theorem publicColumn_bounds {batchSize : Nat}
    (receipt : Receipt batchSize) :
    freshOneCol ≤ publicColumn receipt ∧
      publicColumn receipt < freshOneCol + publicColumnCount batchSize := by
  have claimLt := receipt.claim.isLt
  have ownerLt := receipt.owner.index.isLt
  simp only [publicColumn, claimBase, publicColumnCount]
  omega

/-- No two row instructions own the same allocated public column. -/
theorem publicColumn_injective (batchSize : Nat) :
    Function.Injective
      (publicColumn : Receipt batchSize → Nat) := by
  intro left right equal
  have quotient :
      left.claim.val = right.claim.val := by
    simp only [publicColumn, claimBase] at equal
    have leftOwner := left.owner.index.isLt
    have rightOwner := right.owner.index.isLt
    omega
  have ownerIndex :
      left.owner.index = right.owner.index := by
    apply Fin.ext
    simp only [publicColumn, claimBase] at equal
    omega
  have ownerEqual : left.owner = right.owner := by
    calc
      left.owner = ownerAt left.owner.index :=
        (ownerAt_index left.owner).symm
      _ = ownerAt right.owner.index := by rw [ownerIndex]
      _ = right.owner := ownerAt_index right.owner
  cases left
  cases right
  simp only at quotient ownerEqual
  cases Fin.ext quotient
  cases ownerEqual
  rfl

/-- Every column in the claim-owned public interval has an instruction
receipt; there are no allocated public columns outside emission receipts. -/
theorem publicColumn_surjective_interval
    (batchSize column : Nat)
    (lower : freshOneCol ≤ column)
    (upper : column < freshOneCol + publicColumnCount batchSize) :
    ∃ receipt : Receipt batchSize, publicColumn receipt = column := by
  let offset := column - freshOneCol
  have offsetLt : offset < batchSize * 270 := by
    simp only [publicColumnCount] at upper
    omega
  let slot : Fin (batchSize * 270) := ⟨offset, offsetLt⟩
  refine ⟨receiptAt slot, ?_⟩
  simp only [publicColumn, claimBase, receiptAt]
  rw [index_ownerAt]
  simp only [slot]
  have decomposition :
      offset / 270 * 270 + offset % 270 = offset := by
    simpa [Nat.mul_comm] using Nat.div_add_mod offset 270
  dsimp only [offset]
  omega

/-- Semantic meaning of the repeated row block. -/
structure Holds (batchSize : Nat) (z : Nat → Nat) : Prop where
  affineOne :
    ∀ claim : Fin batchSize,
      z (freshOneColAt claim.val) = 1
  linked :
    ∀ (claim : Fin batchSize) (bit : Fin 256),
      z (freshBitColAt claim.val bit.val) =
        z (lastXOutBitCol bit.val)
  paddingZero :
    ∀ (claim : Fin batchSize) (padding : Fin 13),
      z (freshPaddingColAt claim.val padding.val) = 0

private theorem equality_of_link_row {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1) {left right : Nat}
    (holds : RowHolds z
      ⟨[(left, 1), (right, goldilocksP - 1)], [(0, 1)], []⟩) :
    z left = z right := by
  have leftLt := canonical left
  have rightLt := canonical right
  simp only [RowHolds, lcEval, List.foldl, one, goldilocksP] at holds leftLt rightLt
  omega

private theorem zero_of_row {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1) {column : Nat}
    (holds : RowHolds z
      ⟨[(column, 1)], [(0, 1)], []⟩) :
    z column = 0 := by
  have columnLt := canonical column
  simp only [RowHolds, lcEval, List.foldl, one, goldilocksP] at holds columnLt
  omega

private theorem equality_row_of_equal {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1) {left right : Nat}
    (equal : z left = z right) :
    RowHolds z
      ⟨[(left, 1), (right, goldilocksP - 1)], [(0, 1)], []⟩ := by
  have leftLt := canonical left
  have rightLt := canonical right
  simp only [RowHolds, lcEval, List.foldl, one, goldilocksP]
  omega

private theorem zero_row_of_zero {z : Nat → Nat}
    (one : z 0 = 1) {column : Nat}
    (zero : z column = 0) :
    RowHolds z
      ⟨[(column, 1)], [(0, 1)], []⟩ := by
  simp [RowHolds, lcEval, one, zero]

private theorem emit_mem_rows {batchSize : Nat}
    (receipt : Receipt batchSize) :
    emit receipt ∈ rows batchSize := by
  rw [← row_at_physicalIndex receipt]
  exact List.get_mem (rows batchSize) (physicalIndex receipt)

/-- Satisfying every emitted row yields every claim's affine, link, and
padding obligations. -/
theorem sound {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (satisfies : Satisfies (rows batchSize) z) :
    Holds batchSize z := by
  refine {
    affineOne := ?_
    linked := ?_
    paddingZero := ?_
  }
  · intro claim
    let receipt : Receipt batchSize :=
      ⟨claim, LocalOwner.affineOne⟩
    have row := satisfies (emit receipt) (emit_mem_rows receipt)
    change RowHolds z
      ⟨[(freshOneColAt claim.val, 1), (0, goldilocksP - 1)],
       [(0, 1)], []⟩ at row
    exact (equality_of_link_row canonical one row).trans one
  · intro claim bit
    let receipt : Receipt batchSize :=
      ⟨claim, LocalOwner.linked bit⟩
    have row := satisfies (emit receipt) (emit_mem_rows receipt)
    change RowHolds z
      ⟨[(freshBitColAt claim.val bit.val, 1),
         (lastXOutBitCol bit.val, goldilocksP - 1)],
       [(0, 1)], []⟩ at row
    exact equality_of_link_row canonical one row
  · intro claim padding
    let receipt : Receipt batchSize :=
      ⟨claim, LocalOwner.paddingZero padding⟩
    have row := satisfies (emit receipt) (emit_mem_rows receipt)
    change RowHolds z
      ⟨[(freshPaddingColAt claim.val padding.val, 1)],
       [(0, 1)], []⟩ at row
    exact zero_of_row canonical one row

private theorem emit_holds {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (holds : Holds batchSize z)
    (receipt : Receipt batchSize) :
    RowHolds z (emit receipt) := by
  cases receipt with
  | mk claim owner =>
      cases owner with
      | affineOne =>
          exact equality_row_of_equal canonical one
            ((holds.affineOne claim).trans one.symm)
      | linked bit =>
          exact equality_row_of_equal canonical one
            (holds.linked claim bit)
      | paddingZero padding =>
          exact zero_row_of_zero one
            (holds.paddingZero claim padding)

/-- Every semantic batch obligation constructs satisfaction of the exact
receipt-owned row list. -/
theorem complete {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (holds : Holds batchSize z) :
    Satisfies (rows batchSize) z := by
  intro row member
  simp only [rows, List.mem_ofFn] at member
  rcases member with ⟨slot, rowEqual⟩
  subst row
  exact emit_holds canonical one holds (receiptAt slot)

theorem satisfies_iff_holds {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1) :
    Satisfies (rows batchSize) z ↔ Holds batchSize z :=
  ⟨sound canonical one, complete canonical one⟩

/-- Typed plain-carrier view of one claim's exactly owned public columns. -/
def claimOfAssignment {batchSize : Nat}
    (z : Nat → Nat) (claim : Fin batchSize) :
    CanonicalPlainCarrierLink.Claim where
  mIn := CanonicalPlainCarrierLink.carrierWidth
  x :=
    { one := z (freshOneColAt claim.val)
      body := fun lane bit =>
        z (freshBitColAt claim.val (lane.val * 64 + bit.val))
      padding := fun padding =>
        z (freshPaddingColAt claim.val padding.val) }

/-- The shared producer-bit input columns carry the canonical digest encoder.
This remains an explicit surrounding-owner placement obligation. -/
def ProducerAligned
    (digest : LinkDigest)
    (z : Nat → Nat) : Prop :=
  ∀ lane bit,
    z (lastXOutBitCol (lane.val * 64 + bit.val)) =
      CanonicalPlainCarrierLink.encodedBit digest lane bit

/-- Batch row validity makes every typed claim checker accept. -/
theorem checks_of_holds
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (holds : Holds batchSize z)
    (producerAligned : ProducerAligned digest z) :
    ∀ claim : Fin batchSize,
      CanonicalPlainCarrierLink.check
          digest (claimOfAssignment z claim) = true := by
  intro claim
  apply
    (CanonicalPlainCarrierLink.check_eq_true_iff
        digest (claimOfAssignment z claim)).2
  apply CanonicalPlainCarrierLink.Claim.eq_of_fields
  · rfl
  · apply CanonicalPlainCarrierLink.Carrier.eq_of_fields
    · exact holds.affineOne claim
    · funext lane bit
      have flatLt : lane.val * 64 + bit.val < 256 := by
        have laneLt := lane.isLt
        have bitLt := bit.isLt
        omega
      exact
        (holds.linked claim
          ⟨lane.val * 64 + bit.val, flatLt⟩).trans
          (producerAligned lane bit)
    · funext padding
      exact holds.paddingZero claim padding

/-- Exact rows plus producer alignment reduce every claim in the batch to the
typed canonical checker. -/
theorem checks_of_satisfies
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (satisfies : Satisfies (rows batchSize) z)
    (producerAligned : ProducerAligned digest z) :
    ∀ claim : Fin batchSize,
      CanonicalPlainCarrierLink.check
          digest (claimOfAssignment z claim) = true :=
  checks_of_holds digest (sound canonical one satisfies) producerAligned

/-- Conversely, acceptance of every typed claim reconstructs the semantic
batch relation and therefore every emitted row. -/
theorem satisfies_of_checks
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (checks :
      ∀ claim : Fin batchSize,
        CanonicalPlainCarrierLink.check
            digest (claimOfAssignment z claim) = true)
    (producerAligned : ProducerAligned digest z) :
    Satisfies (rows batchSize) z := by
  apply complete canonical one
  refine {
    affineOne := ?_
    linked := ?_
    paddingZero := ?_
  }
  · intro claim
    have claimEqual :=
      (CanonicalPlainCarrierLink.check_eq_true_iff
          digest (claimOfAssignment z claim)).1 (checks claim)
    simpa [
      claimOfAssignment,
      CanonicalPlainCarrierLink.encodeClaim,
      CanonicalPlainCarrierLink.encodeCarrier
    ] using
      congrArg
        (fun typed : CanonicalPlainCarrierLink.Claim => typed.x.one)
        claimEqual
  · intro claim bit
    let lane : Fin 4 := ⟨bit.val / 64, by
      have bitLt := bit.isLt
      omega⟩
    let offset : Fin 64 := ⟨bit.val % 64, Nat.mod_lt _ (by omega)⟩
    have flatEqual : lane.val * 64 + offset.val = bit.val := by
      simp only [lane, offset]
      simpa [Nat.mul_comm] using Nat.div_add_mod bit.val 64
    have claimEqual :=
      (CanonicalPlainCarrierLink.check_eq_true_iff
          digest (claimOfAssignment z claim)).1 (checks claim)
    have freshEncoded :
        z (freshBitColAt claim.val (lane.val * 64 + offset.val)) =
          CanonicalPlainCarrierLink.encodedBit digest lane offset := by
      simpa [
        claimOfAssignment,
        CanonicalPlainCarrierLink.encodeClaim,
        CanonicalPlainCarrierLink.encodeCarrier
      ] using
        congrArg
          (fun typed : CanonicalPlainCarrierLink.Claim =>
            typed.x.body lane offset)
          claimEqual
    have aligned := producerAligned lane offset
    rw [flatEqual] at freshEncoded aligned
    exact freshEncoded.trans aligned.symm
  · intro claim padding
    have claimEqual :=
      (CanonicalPlainCarrierLink.check_eq_true_iff
          digest (claimOfAssignment z claim)).1 (checks claim)
    simpa [
      claimOfAssignment,
      CanonicalPlainCarrierLink.encodeClaim,
      CanonicalPlainCarrierLink.encodeCarrier
    ] using
      congrArg
        (fun typed : CanonicalPlainCarrierLink.Claim =>
          typed.x.padding padding)
        claimEqual

/-- Exact arbitrary-batch artifact-independent R1CS refinement. -/
theorem satisfies_iff_checks
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned : ProducerAligned digest z) :
    Satisfies (rows batchSize) z ↔
      ∀ claim : Fin batchSize,
        CanonicalPlainCarrierLink.check
            digest (claimOfAssignment z claim) = true :=
  ⟨fun satisfies =>
      checks_of_satisfies digest canonical one satisfies producerAligned,
    fun checks =>
      satisfies_of_checks digest canonical one checks producerAligned⟩

/-- Every batch claim accepted by the exact rows is precisely the zero
completion of a logical HyperNova public input. -/
theorem satisfies_iff_logicalPaperLinks
    (digest : LinkDigest)
    {batchSize : Nat} {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned : ProducerAligned digest z) :
    Satisfies (rows batchSize) z ↔
      ∀ claim : Fin batchSize,
        ∃ logical,
          CanonicalPublicInputLink.check digest logical = true ∧
          claimOfAssignment z claim =
            CanonicalPlainCarrierLink.completeClaim logical := by
  rw [satisfies_iff_checks digest canonical one producerAligned]
  apply forall_congr'
  intro claim
  exact CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink
    digest (claimOfAssignment z claim)

end Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
