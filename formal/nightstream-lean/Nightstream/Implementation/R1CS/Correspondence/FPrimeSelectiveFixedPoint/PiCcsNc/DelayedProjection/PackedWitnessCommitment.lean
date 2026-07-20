import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren

/-!
Packed-witness coordinates at the production Ajtai opening boundary.

Assurance tier: artifact-checked geometry and model-level commitment
correspondence.  Native key serialization and Ajtai binding remain explicit
external boundaries.

Owns: the exact finite Ajtai equation over Rust's `(lane, block)` witness
matrix; equality of that equation with the independently typed commitment of
`PackedWitness.unpack`; the exact flattened `Commitment.data[row*54+lane]`
view for bounded κ=4 and production κ=18; and derivation of raw-running
commitment authority from the fourteen actual packed-matrix openings.

Does not own: key generation or serialization, Ajtai/MSIS hardness, extraction
of intermediate witnesses from SumCheck acceptance, terminal opening rows,
transcript scheduling, `y_ring`, costs, or row-removal permission.

Emits constraints: none; direct coordinate/refinement theorem only.

| Stable stage path | Obligation | Authority class | Rust owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.witness.commitment.block` | one key block multiplies exactly `Z[lane, block]` | direct dataflow | `AjtaiSModule::commit`, `commit_row_major` |
| `f_prime.pi_ccs_nc.witness.commitment.value` | the complete matrix equation equals the typed Ajtai commitment | derived | same |
| `f_prime.pi_ccs_nc.witness.commitment.data` | Rust `Commitment.data[row*54+lane]` is a bijective flattened view of the typed value | bounded artifact / derived production indexing | `Commitment::col` |
| `f_prime.pi_ccs_nc.witness.commitment.children` | fourteen opened raw matrices bind the successor running children | typed external boundary / derived | combined-NC raw witnesses and terminal CE opening |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessCommitment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
open PackedWitness

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- One complete Rust witness column, read in the coefficient-lane order used
by the typed Phi81 ring block. -/
def matrixBlock
    (witness : Matrix shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) : RingF :=
  fun lane => witness lane (rustBlockOfSemantic block)

/-- Reading a Rust matrix block is exactly reading the corresponding block of
the independently typed unpacked assignment. -/
theorem matrixBlock_eq_assignmentBlock_unpack
    (witness : Matrix shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    matrixBlock witness block =
      CarrierAction.assignmentBlock (unpack witness) block := by
  funext lane
  change witness lane (rustBlockOfSemantic block) =
    unpack witness (Phi81CarrierLayout.carrierColumn block lane)
  calc
    witness lane (rustBlockOfSemantic block) =
        pack (unpack witness) lane (rustBlockOfSemantic block) := by
      rw [pack_unpack]
    _ = unpack witness (Phi81CarrierLayout.carrierColumn block lane) := by
      unfold pack
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane.carrierColumn
      rw [semanticBlockOfRust_rustBlockOfSemantic]

/-- Rust-shaped Ajtai opening equation: key row first, then every packed
witness block, with the 54 coefficient lanes inside the ring value. -/
def matrixCommit
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (witness : Matrix shape) : CommitmentValue verifierRows :=
  fun row =>
    PiRLCAlgebra.Commitment.blockSum fun block =>
      ringFMul (key row block) (matrixBlock witness block)

/-- The Rust-shaped full-matrix equation and the independently typed Ajtai
commitment are identical at every verifier row.  This is coordinate
alignment, not a cryptographic binding theorem. -/
theorem matrixCommit_eq_typedCommit
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (witness : Matrix shape) :
    matrixCommit key witness = commit key (unpack witness) := by
  funext row
  unfold matrixCommit commit PiRLCAlgebra.Commitment.commit
    PiRLCAlgebra.Commitment.ajtaiRow
  apply congrArg PiRLCAlgebra.Commitment.blockSum
  funext block
  rw [matrixBlock_eq_assignmentBlock_unpack]
  rfl

/-- Typed view of Rust's flattened `Commitment.data` buffer. The public
commitment type remains row-then-lane; only its external storage is flat. -/
def flattenedCommitment (width : Nat)
    (value : CommitmentValue width)
    (index : Fin (width * GeneratedLayout.matrixRows)) : F :=
  let address := commitmentDataAddress width index
  value address.1 address.2

@[simp] theorem flattenedCommitment_at (width : Nat)
    (value : CommitmentValue width)
    (address : Fin width × LiveLane) :
    flattenedCommitment width value (commitmentDataIndex width address) =
      value address.1 address.2 := by
  simp [flattenedCommitment]

/-- Bounded κ=4 fixture view exercised by all 108 one-hot constructor probes. -/
def fixtureFlattenedCommitment
    (value : CommitmentValue GeneratedLayout.fixtureCommitmentWidth)
    (index : FixtureCommitmentData) : F :=
  flattenedCommitment GeneratedLayout.fixtureCommitmentWidth value index

/-- Protocol production κ=18 flattened view. The index formula is exact;
native PP coefficient serialization remains an explicit boundary. -/
def productionFlattenedCommitment
    (value : CommitmentValue GeneratedLayout.productionCommitmentWidth)
    (index : ProductionCommitmentData) : F :=
  flattenedCommitment GeneratedLayout.productionCommitmentWidth value index

/-- At the protocol production width, one flattened Rust commitment cell is
exactly one lane of the typed full-matrix Ajtai equation. This binds order and
shape, not Ajtai hardness or native PP coefficient equality. -/
theorem productionFlattened_matrixCommit_at
    (key : VerifierKey shape publicRingColumns publicFits
      GeneratedLayout.productionCommitmentWidth)
    (witness : Matrix shape)
    (address : ProductionCommitmentRow × LiveLane) :
    productionFlattenedCommitment (matrixCommit key witness)
        (commitmentDataIndex GeneratedLayout.productionCommitmentWidth
          address) =
      (PiRLCAlgebra.Commitment.blockSum fun block =>
        ringFMul (key address.1 block) (matrixBlock witness block))
        address.2 := by
  simp [productionFlattenedCommitment, matrixCommit]

/-- Actual packed-matrix openings derive the precise raw-child commitment
premise consumed by the delayed projection theorem.  The child index is the
verifier-owned running-source alignment; no `CeClaim.y_zcol` field or digest
appears in the premise. -/
theorem rawRunningCommitmentsBound_of_openedPackedWitnesses
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (opened : forall child,
      matrixCommit context.key
          (witnesses (context.alignment.semanticRunningIndex child)) =
        (context.input.running child).commitment) :
    DelayedRawChildren.RawRunningCommitmentsBound context
      (decodedData template witnesses) := by
  intro child
  change commit context.key
      (unpack (witnesses (context.alignment.semanticRunningIndex child))) =
    (context.input.running child).commitment
  rw [<- matrixCommit_eq_typedCommit]
  exact opened child

/-- Exact bidirectional Rust handoff contract for successor child authority.
The left side speaks only about the fourteen raw packed matrices and their
public commitments; the right side is the semantic premise used by the
delayed-production proof. -/
theorem openedPackedWitnesses_iff_rawRunningCommitmentsBound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape) :
    (forall child,
        matrixCommit context.key
            (witnesses (context.alignment.semanticRunningIndex child)) =
          (context.input.running child).commitment) <->
      DelayedRawChildren.RawRunningCommitmentsBound context
        (decodedData template witnesses) := by
  constructor
  . exact rawRunningCommitmentsBound_of_openedPackedWitnesses context template
      witnesses
  . intro bound child
    have opened := bound child
    change commit context.key
        (unpack (witnesses (context.alignment.semanticRunningIndex child))) =
      (context.input.running child).commitment at opened
    calc
      matrixCommit context.key
          (witnesses (context.alignment.semanticRunningIndex child)) =
          commit context.key
            (unpack
              (witnesses (context.alignment.semanticRunningIndex child))) :=
        matrixCommit_eq_typedCommit _ _
      _ = (context.input.running child).commitment := opened

/-- Failure of raw-running commitment authority is exactly a mismatch between
one actual packed child matrix and its verifier-owned public commitment.

This theorem makes the negative security branch concrete without assuming
the positive opening equations. It is an exhaustive logical partition, not
an Ajtai binding theorem. -/
theorem rawRunningCommitmentsUnbound_iff_exists_matrixCommit_ne
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape) :
    (¬ DelayedRawChildren.RawRunningCommitmentsBound context
        (decodedData template witnesses)) ↔
      ∃ child,
        matrixCommit context.key
            (witnesses (context.alignment.semanticRunningIndex child)) ≠
          (context.input.running child).commitment := by
  classical
  constructor
  . intro unbound
    exact Classical.byContradiction fun noMismatch => unbound <|
      rawRunningCommitmentsBound_of_openedPackedWitnesses context template
        witnesses fun child =>
          Classical.byContradiction fun mismatch =>
            noMismatch ⟨child, mismatch⟩
  . intro mismatchWitness bound
    rcases mismatchWitness with ⟨child, mismatch⟩
    have opened :=
      (openedPackedWitnesses_iff_rawRunningCommitmentsBound context template
        witnesses).mpr bound
    exact mismatch (opened child)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessCommitment
