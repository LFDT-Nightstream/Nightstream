import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.Implementation.Rust.NifsProductionGolden.FixedRelation
import Nightstream.Implementation.Rust.NifsProductionGolden.Receipt

/-! Total decoding after the receipt shape check has rejected malformed data. -/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.NifsProductionGolden

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Rust.PiCcsExecution

abbrev GoldenCommitment :=
  PiRLCAlgebra.Commitment.Value 18

abbrev GoldenInstance :=
  CE.Instance
    (Structure FixedRelation.shape)
    (PublicInput FixedRelation.shape)
    (Point FixedRelation.shape)
    Evaluation
    GoldenCommitment

def decodeF (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def decodeCommitment (raw : RawCommitment) : GoldenCommitment :=
  fun row lane => decodeF (raw.data.getD (row.val * 54 + lane.val) 0)

def decodePublicInput (raw : RawClaim) : PublicInput FixedRelation.shape :=
  fun column => decodeF (raw.publicInput.getD column.val 0)

def decodePoint (raw : RawClaim) : Point FixedRelation.shape where
  coordinates := (List.range 6).map fun index =>
    (raw.point.getD index default).decode
  dimension := by simp [FixedRelation.shape]

def decodeEvaluations (raw : RawClaim) : Array Evaluation :=
  Array.ofFn fun matrix : Fin 4 =>
    fun lane =>
      (raw.evaluations.getD (matrix.val * 64 + lane.val) default).decode

def decodeClaim (stage : NormStage) (raw : RawClaim) : GoldenInstance where
  constraintSystem := FixedRelation.system
  commitment := decodeCommitment raw.commitment
  publicInput := decodePublicInput raw
  point := decodePoint raw
  evaluations := decodeEvaluations raw
  stage := stage

def decodeSnapshot (raw : RawTranscriptSnapshot) :
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State where
  lanes := fun lane => raw.lanes.getD lane.val 0
  absorbed := raw.absorbed

end Nightstream.Implementation.Rust.NifsProductionGolden
