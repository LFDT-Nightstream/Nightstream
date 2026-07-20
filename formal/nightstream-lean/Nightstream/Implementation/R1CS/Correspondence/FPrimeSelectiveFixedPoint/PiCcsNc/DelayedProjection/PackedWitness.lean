import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane

/-!
Exact packed-witness/assignment refinement contract for production `Pi_CCS`
and delayed `y_zcol` authority.

Assurance tier: model-level until generated production witness-decoder rows
and the Rust verifier instantiate this contract.

Owns: Rust's full packed-witness coordinate rule
`Z[(lane, block)] = assignment[block * 54 + lane]`; equality of Rust's
`ceil(logicalWidth / 54)` matrix width and the independent completed-carrier
block count; mutually inverse packing and unpacking; canonical fresh-tail
zero padding; and construction of `Sources.Data.runningAssignments` directly
from full packed witnesses.

Does not own: the bounded `CeClaim.X` decoder, sparse allocation rows,
combined-NC acceptance, transcript scheduling, recursive-state continuity,
Ajtai internals, proof that a claimed commitment opens to the supplied `Z`,
costs, or row-removal permission. Commitment failure remains the explicit
`RunningWitnessBindingFailure` boundary below.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority class | Rust owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.witness.layout` | `Z[lane, block] = z[block * 54 + lane]` over the complete witness | direct dataflow | `CcsInstance::from_low_norm_assignment`, `CcsWitness.Z`, `CeWitness.Z` |
| `f_prime.pi_ccs_nc.witness.inverse` | packing and unpacking are mutual inverses | derived | same |
| `f_prime.pi_ccs_nc.witness.fresh_padding` | a fresh final partial block is zero outside the logical assignment | computed | zero-initialized packed matrix |
| `f_prime.pi_ccs_nc.witness.running_sources` | combined NC reads full packed witnesses, never `CeClaim.y_zcol` | direct dataflow | production combined-NC witness handoff |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-- Rust's full packed witness matrix: Phi81 coefficient lane first, then
ring block. Its block count is computed from the original CCS width, exactly
as `pack_assignment_into_ring_matrix` computes `structure.m.div_ceil(D)`. -/
abbrev Matrix (shape : SemanticShape) :=
  Fin ringDegree -> Fin (Phi81ColumnLayout.blockCount shape.logicalWidth) -> F

/-- The semantic completed carrier and Rust packed matrix have exactly the
same number of ring blocks. -/
theorem semanticBlockCount_eq_rustBlockCount (shape : SemanticShape) :
    Phi81ColumnLayout.blockCount shape.carrierWidth =
      Phi81ColumnLayout.blockCount shape.logicalWidth := by
  simpa [SemanticShape.carrierWidth] using
    Phi81CarrierLayout.blockCount_carrierWidth shape.logicalWidth

/-- Cast a semantic completed-carrier block to the corresponding Rust matrix
column. This changes only the type-level bound. -/
def rustBlockOfSemantic
    {shape : SemanticShape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    Fin (Phi81ColumnLayout.blockCount shape.logicalWidth) :=
  Fin.cast (semanticBlockCount_eq_rustBlockCount shape) block

/-- Inverse cast from a Rust matrix column to a semantic carrier block. -/
def semanticBlockOfRust
    {shape : SemanticShape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.logicalWidth)) :
    Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  Fin.cast (semanticBlockCount_eq_rustBlockCount shape).symm block

@[simp] theorem semanticBlockOfRust_rustBlockOfSemantic
    {shape : SemanticShape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    semanticBlockOfRust (rustBlockOfSemantic block) = block := by
  apply Fin.ext
  rfl

@[simp] theorem rustBlockOfSemantic_semanticBlockOfRust
    {shape : SemanticShape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.logicalWidth)) :
    rustBlockOfSemantic (semanticBlockOfRust block) = block := by
  apply Fin.ext
  rfl

/-- Decode a full Rust packed witness into the one-dimensional authoritative
assignment consumed by independent Split-NC semantics. -/
def unpack
    {shape : SemanticShape}
    (witness : Matrix shape) : Assignment F shape.carrierWidth :=
  fun column =>
    witness (Phi81ColumnLayout.decode column).2
      (rustBlockOfSemantic (Phi81ColumnLayout.decode column).1)

/-- Pack a complete semantic assignment into Rust's `(lane, block)` matrix
order. -/
def pack
    {shape : SemanticShape}
    (assignment : Assignment F shape.carrierWidth) : Matrix shape :=
  fun lane block =>
    assignment
      (BlockLane.carrierColumn
        (semanticBlockOfRust block) lane)

/-- Unpacking a packed complete assignment recovers every flat coordinate. -/
@[simp] theorem unpack_pack
    {shape : SemanticShape}
    (assignment : Assignment F shape.carrierWidth) :
    unpack (pack assignment) = assignment := by
  funext column
  unfold unpack pack
  rw [semanticBlockOfRust_rustBlockOfSemantic]
  rw [BlockLane.carrierColumn_decode]

/-- Packing an unpacked full witness recovers every Rust matrix cell. -/
@[simp] theorem pack_unpack
    {shape : SemanticShape}
    (witness : Matrix shape) :
    pack (unpack witness) = witness := by
  funext lane block
  unfold pack unpack BlockLane.carrierColumn
  have decoded :
      Phi81ColumnLayout.decode
          (Phi81CarrierLayout.carrierColumn
            (semanticBlockOfRust block) lane) =
        (semanticBlockOfRust block, lane) := by
    exact Phi81CarrierLayout.decode_carrierColumn
      (logicalWidth := shape.logicalWidth)
        (semanticBlockOfRust block) lane
  have decodedLane :
      (Phi81ColumnLayout.decode
        (Phi81CarrierLayout.carrierColumn
          (semanticBlockOfRust block) lane)).2 = lane :=
    congrArg Prod.snd decoded
  have decodedBlock :
      (Phi81ColumnLayout.decode
        (Phi81CarrierLayout.carrierColumn
          (semanticBlockOfRust block) lane)).1 =
        semanticBlockOfRust block :=
    congrArg Prod.fst decoded
  have rustDecodedBlock :
      rustBlockOfSemantic
          (Phi81ColumnLayout.decode
            (Phi81CarrierLayout.carrierColumn
              (semanticBlockOfRust block) lane)).1 = block := by
    calc
      rustBlockOfSemantic
          (Phi81ColumnLayout.decode
            (Phi81CarrierLayout.carrierColumn
              (semanticBlockOfRust block) lane)).1 =
          rustBlockOfSemantic (semanticBlockOfRust block) :=
        congrArg rustBlockOfSemantic decodedBlock
      _ = block := rustBlockOfSemantic_semanticBlockOfRust block
  calc
    witness
        (Phi81ColumnLayout.decode
          (Phi81CarrierLayout.carrierColumn
            (semanticBlockOfRust block) lane)).2
        (rustBlockOfSemantic
          (Phi81ColumnLayout.decode
            (Phi81CarrierLayout.carrierColumn
              (semanticBlockOfRust block) lane)).1) =
      witness lane
        (rustBlockOfSemantic
          (Phi81ColumnLayout.decode
            (Phi81CarrierLayout.carrierColumn
              (semanticBlockOfRust block) lane)).1) :=
        congrArg
          (fun decodedLaneValue =>
            witness decodedLaneValue
              (rustBlockOfSemantic
                (Phi81ColumnLayout.decode
                  (Phi81CarrierLayout.carrierColumn
                    (semanticBlockOfRust block) lane)).1))
          decodedLane
    _ = witness lane block := congrArg (witness lane) rustDecodedBlock

/-- Explicit cellwise coordinate alignment. This is the exact layout fact a
commitment-opening refinement must establish; equality of digests is not a
substitute. -/
def CoordinatesAligned
    {shape : SemanticShape}
    (witness : Matrix shape)
    (assignment : Assignment F shape.carrierWidth) : Prop :=
  ∀ lane block,
    witness lane block =
      assignment
        (BlockLane.carrierColumn
          (semanticBlockOfRust block) lane)

/-- Cellwise Rust/Lean alignment is equivalent to equality after unpacking. -/
theorem coordinatesAligned_iff_unpack_eq
    {shape : SemanticShape}
    (witness : Matrix shape)
    (assignment : Assignment F shape.carrierWidth) :
    CoordinatesAligned witness assignment <->
      unpack witness = assignment := by
  constructor
  · intro aligned
    have packedEq : pack assignment = witness := by
      funext lane block
      exact (aligned lane block).symm
    rw [← packedEq, unpack_pack]
  · intro unpacked lane block
    calc
      witness lane block = pack (unpack witness) lane block := by
        rw [pack_unpack]
      _ = pack assignment lane block := by rw [unpacked]
      _ = assignment
          (BlockLane.carrierColumn
            (semanticBlockOfRust block) lane) := rfl

/-- Pack an original-width fresh assignment after the independent semantics
has canonically completed its final partial Phi81 block with zeros. -/
def packFresh
    {shape : SemanticShape}
    (assignment : Assignment F shape.logicalWidth) : Matrix shape :=
  pack (Phi81CarrierLayout.extendAssignment 0 assignment)

/-- The unpacked fresh witness is exactly the canonical completed assignment. -/
@[simp] theorem unpack_packFresh
    {shape : SemanticShape}
    (assignment : Assignment F shape.logicalWidth) :
    unpack (packFresh assignment) =
      Phi81CarrierLayout.extendAssignment 0 assignment := by
  exact unpack_pack _

/-- Every logical fresh coordinate survives the packed-witness round trip. -/
theorem unpack_packFresh_logical
    {shape : SemanticShape}
    (assignment : Assignment F shape.logicalWidth)
    (column : Fin shape.logicalWidth) :
    unpack (packFresh assignment)
        (Phi81CarrierLayout.embedLogical column) = assignment column := by
  rw [unpack_packFresh]
  exact Phi81CarrierLayout.extendAssignment_embedLogical 0 assignment column

/-- Every completed fresh coordinate beyond the original width is the
zero-initialized Rust matrix value. -/
theorem unpack_packFresh_tail_zero
    {shape : SemanticShape}
    (assignment : Assignment F shape.logicalWidth)
    (column : Fin shape.carrierWidth)
    (tail : shape.logicalWidth <= column.val) :
    unpack (packFresh assignment) column = 0 := by
  rw [unpack_packFresh]
  exact Phi81CarrierLayout.extendAssignment_tail_zero 0 assignment column tail

/-- Full running assignments decoded directly from the corresponding packed
CE/CCS witness matrices. -/
def decodedRunningAssignments
    {shape : SemanticShape}
    (witnesses : Fin shape.runningCount -> Matrix shape) :
    Fin shape.runningCount -> Assignment F shape.carrierWidth :=
  fun source => unpack (witnesses source)

/-- Replace only the running-source family of a semantic input with the full
packed witnesses supplied to production combined NC. -/
def decodedData
    {shape : SemanticShape}
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape) : Data shape :=
  { template with runningAssignments := decodedRunningAssignments witnesses }

@[simp] theorem decodedData_runningAssignments
    {shape : SemanticShape}
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (source : Fin shape.runningCount) :
    (decodedData template witnesses).runningAssignments source =
      unpack (witnesses source) := by
  rfl

/-- Each decoded full witness is coordinate-aligned with the running source
consumed by the independent semantics, by construction rather than by a
caller-supplied authority premise. -/
theorem decodedData_coordinatesAligned
    {shape : SemanticShape}
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (source : Fin shape.runningCount) :
    CoordinatesAligned (witnesses source)
      ((decodedData template witnesses).runningAssignments source) := by
  apply (coordinatesAligned_iff_unpack_eq _ _).2
  rfl

/-- Exact external failure boundary: at least one packed witness cell is not
the corresponding authoritative running-assignment coordinate. A later
Ajtai reduction must derive this branch from commitment binding failure; this
file does not assume or re-prove that primitive. -/
def RunningWitnessBindingFailure
    {shape : SemanticShape}
    (data : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape) : Prop :=
  ∃ source, ∃ lane, ∃ block,
    witnesses source lane block ≠
      data.runningAssignments source
        (BlockLane.carrierColumn
          (semanticBlockOfRust block) lane)

/-- Data constructed from full packed witnesses cannot itself exhibit the
coordinate-binding failure. -/
theorem decodedData_not_runningWitnessBindingFailure
    {shape : SemanticShape}
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape) :
    ¬ RunningWitnessBindingFailure (decodedData template witnesses)
      witnesses := by
  rintro ⟨source, lane, block, mismatch⟩
  exact mismatch (decodedData_coordinatesAligned template witnesses source
    lane block)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness
