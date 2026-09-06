import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.CommitmentSparse

/-!
Owns executable commitment conformance for one sparse full-carrier assignment.
Its three original block addresses match the Rust signed-unit reference test.
The last coefficient is a carried coordinate, not a fresh CCS padding claim.
The output is proved equal to the existing full-domain commitment relation.
-/

namespace NightstreamFPrime.Export.Stage1.AjtaiSparseCommitmentV1Parity

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open NightstreamFPrime.Export.Codec

def shape : Phi81Relation.Shape := Lifecycle.PaperAlgebra.fullShape
  (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
  (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

def firstBlock : Fin Poseidon2HashChainV1Setup.messageColumns :=
  ⟨0, by rw [Poseidon2HashChainV1Setup.messageColumns_eq]; decide⟩

def middleBlock : Fin Poseidon2HashChainV1Setup.messageColumns :=
  ⟨32768, by rw [Poseidon2HashChainV1Setup.messageColumns_eq]; decide⟩

def lastBlock : Fin Poseidon2HashChainV1Setup.messageColumns :=
  ⟨4750595, by rw [Poseidon2HashChainV1Setup.messageColumns_eq]; decide⟩

def assignment : Phi81Relation.Assignment shape :=
  BaseLinear.assignmentAdd
    (CommitmentSparse.singleBlock (shape := shape) firstBlock (ringFMonomial 0 1))
    (BaseLinear.assignmentAdd
      (CommitmentSparse.singleBlock (shape := shape) middleBlock (ringFMonomial 27 (-1)))
      (CommitmentSparse.singleBlock (shape := shape) lastBlock (ringFMonomial 53 1)))

/-- The complete carrier contains exactly the three Rust test coordinates. -/
theorem assignment_coordinate (column : Fin shape.carrierWidth) :
    assignment column =
      (if column.val = 0 then 1 else 0) +
        ((if column.val = 1769499 then -1 else 0) +
          (if column.val = 256532183 then 1 else 0)) := by
  change CommitmentSparse.singleBlock (shape := shape) firstBlock (ringFMonomial 0 1) column +
      (CommitmentSparse.singleBlock (shape := shape) middleBlock (ringFMonomial 27 (-1)) column +
        CommitmentSparse.singleBlock (shape := shape) lastBlock (ringFMonomial 53 1) column) = _
  rw [CommitmentSparse.singleBlock_monomial_coordinate (shape := shape) firstBlock
      (⟨0, by decide⟩ : Fin ringDegree),
    CommitmentSparse.singleBlock_monomial_coordinate (shape := shape) middleBlock
      (⟨27, by decide⟩ : Fin ringDegree),
    CommitmentSparse.singleBlock_monomial_coordinate (shape := shape) lastBlock
      (⟨53, by decide⟩ : Fin ringDegree)]
  simp only [Fin.ext_iff]
  rfl

private def cacheRing (value : RingF) : RingF :=
  let coordinates := Array.ofFn value
  fun lane => coordinates[lane.val]'(by
    simp only [coordinates, Array.size_ofFn]
    exact lane.isLt)

private theorem cacheRing_eq (value : RingF) : cacheRing value = value := by
  funext lane
  simp [cacheRing]

/-- Only three 54-coefficient key blocks are evaluated per row. Cached values
retain the exact selected key and use the existing ring multiplication. -/
def sparseCommitment : Commitment.Value Poseidon2HashChainV1Setup.verifierRows :=
  fun row =>
    let first := cacheRing (Poseidon2HashChainV1Setup.productionAjtaiKey row firstBlock)
    let middle := cacheRing (Poseidon2HashChainV1Setup.productionAjtaiKey row middleBlock)
    let last := cacheRing (Poseidon2HashChainV1Setup.productionAjtaiKey row lastBlock)
    ringFAdd (ringFMul first (ringFMonomial 0 1))
      (ringFAdd (ringFMul middle (ringFMonomial 27 (-1)))
        (ringFMul last (ringFMonomial 53 1)))

/-- Sparse execution is the exact full-carrier Ajtai commitment. The proof
uses structural support elimination and the existing additive theorem. -/
theorem sparseCommitment_eq_commit :
    sparseCommitment =
      Commitment.commit (shape := shape) Poseidon2HashChainV1Setup.productionAjtaiKey assignment := by
  unfold assignment
  rw [Commitment.commit_add, Commitment.commit_add]
  rw [CommitmentSparse.commit_singleBlock, CommitmentSparse.commit_singleBlock,
    CommitmentSparse.commit_singleBlock]
  funext row
  simp only [sparseCommitment, cacheRing_eq, Commitment.commitmentAdd]

private def rowValue (row : Fin Poseidon2HashChainV1Setup.verifierRows) : Value :=
  let firstCoordinates := Array.ofFn
    (Poseidon2HashChainV1Setup.productionAjtaiKey row firstBlock)
  let middleCoordinates := Array.ofFn
    (Poseidon2HashChainV1Setup.productionAjtaiKey row middleBlock)
  let lastCoordinates := Array.ofFn
    (Poseidon2HashChainV1Setup.productionAjtaiKey row lastBlock)
  let first : RingF := fun lane => firstCoordinates[lane.val]'(by
    simp only [firstCoordinates, Array.size_ofFn]; exact lane.isLt)
  let middle : RingF := fun lane => middleCoordinates[lane.val]'(by
    simp only [middleCoordinates, Array.size_ofFn]; exact lane.isLt)
  let last : RingF := fun lane => lastCoordinates[lane.val]'(by
    simp only [lastCoordinates, Array.size_ofFn]; exact lane.isLt)
  let result := ringFAdd (ringFMul first (ringFMonomial 0 1))
    (ringFAdd (ringFMul middle (ringFMonomial 27 (-1)))
      (ringFMul last (ringFMonomial 53 1)))
  .array (List.ofFn fun lane : Fin ringDegree => .atom (result lane).val)

/-- Array buffers are constructed once per emitted row. Their output is the
same complete commitment relation, without repeated key expansion per read. -/
theorem rowValue_eq_commit (row : Fin Poseidon2HashChainV1Setup.verifierRows) :
    rowValue row = .array (List.ofFn fun lane : Fin ringDegree =>
      .atom ((Commitment.commit (shape := shape)
        Poseidon2HashChainV1Setup.productionAjtaiKey assignment row) lane).val) := by
  rw [← sparseCommitment_eq_commit]
  rfl

/-- Schema 1: authority words, original block/lane/scalar support, and all
22 by 54 commitment coefficients. Existing setup fixtures are unchanged. -/
def parityValue (_delay : Unit := ()) : Value :=
  .array [.atom 1,
    .array (Poseidon2HashChainV1Setup.directProductionAuthorityNats.map Value.atom),
    .array [
      .array [.atom firstBlock.val, .atom 0, .atom 1],
      .array [.atom middleBlock.val, .atom 27, .atom ((-1 : F).val)],
      .array [.atom lastBlock.val, .atom 53, .atom 1]],
    .array (List.ofFn rowValue)]

end NightstreamFPrime.Export.Stage1.AjtaiSparseCommitmentV1Parity
