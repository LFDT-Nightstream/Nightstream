import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.FirstAccepted
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.RingEncoding

/-!
Recursive-bootstrap `Pi_RLC` coefficient-vector assembly into the concrete
Phi81 ring.

Owns: the exact single-scalar challenge-column formula, the 54-coordinate ring
assembly, and equality between every decoded production challenge coefficient
and the verifier-owned Poseidon2/sampler result.

Does not own: the post-`Pi_CCS` initial transcript-state binding, projection
identities, pairwise strong-set security, native Rust conformance, row removal,
or cost totals.

Emits constraints: no.

Authority boundary: recursive bootstrap has one `Pi_RLC` input. Its ring
challenge is assembled only from the first 54 accepted symbols of the connected
Poseidon2 machine. The output columns are interpreted by the theorem; they are
not independent challenge authority.

| Protocol | Phase | Constraint family | Indexed leaf | Exact obligation |
|---|---|---|---|---|
| `Pi_RLC` | challenge output | column placement | coefficient `Fin 54` | `356244 + 45*coefficient` |
| `Pi_RLC` | coefficient encoding | centered Goldilocks value | one five-symbol coefficient | shared `RingEncoding` theorem |
| `Pi_RLC` | ring assembly | Phi81 coefficient vector | bootstrap scalar | exactly 54 field coefficients in source order |
| `Pi_RLC` | semantic bridge | decoded production vector | all 54 coefficients | equality with connected machine output |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.RingEncoding
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- Exact global column consumed as one recursive-bootstrap quotient-ring
coefficient. -/
def challengeColumn
    (position : Fin ProductionAlphabet.coefficientCount) : Nat :=
  Relabel.column
    (AlphabetSamplingResidualTemplate.tailColumnMap
      OneScalarRows.tailBitStarts OneScalarRows.tailFirstAllocated)
    (SelectionRows.outputCol position.val)

/-- Closed affine address of all 54 recursive-bootstrap challenge outputs. -/
theorem challengeColumn_formula :
    forall position : Fin ProductionAlphabet.coefficientCount,
      challengeColumn position = 356244 + 45 * position.val := by
  decide

/-- Exact recursive-bootstrap challenge columns in coefficient order. -/
def challengeColumns : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    challengeColumn position

@[simp] theorem challengeColumns_length :
    challengeColumns.length = ProductionAlphabet.coefficientCount := by
  simp [challengeColumns]

/-- The readable tail and global assignment select the same output column. -/
theorem localOutput_eq_assignment
    (assignment : Nat -> Nat)
    (position : Fin ProductionAlphabet.coefficientCount) :
    TailRows.localAssignment assignment
        (SelectionRows.outputCol position.val) =
      assignment (challengeColumn position) := by
  rfl

/-- Total 54-coordinate coefficient view of the independent first-accepted
machine output. The default is unreachable under accepted rows. -/
def machineScalar
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    ProductionStrongSet.Scalar :=
  fun position =>
    (Selection.FirstAcceptedRefinement.semanticOutput assignment canonical).getD
      position.val Selection.FirstAcceptedRefinement.defaultCoefficient

/-- Verifier-derived bootstrap scalar embedded coefficientwise in Phi81. -/
def machineChallenge
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) : RingF :=
  fun position => embedCoefficient (machineScalar assignment canonical position)

theorem machineChallenge_eq_embedScalar
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    machineChallenge assignment canonical =
      Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar
        (machineScalar assignment canonical) := by
  funext position
  rfl

/-- Direct field decoding of the production challenge columns. -/
def decodedChallenge (assignment : Nat -> Nat) : RingF :=
  fun position => fieldResidue (assignment (challengeColumn position))

/-- Accepted recursive rows force every decoded coefficient to equal the
connected machine-derived coefficient. -/
theorem decodedChallenge_eq_machineChallenge
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    decodedChallenge assignment = machineChallenge assignment canonical := by
  funext position
  let refinement := OneScalar.accepted_refines prime canonical one accepted
  have output := Selection.FirstAcceptedRefinement.outputAt_refines
    prime canonical one accepted refinement position
  rw [localOutput_eq_assignment assignment position] at output
  unfold decodedChallenge machineChallenge machineScalar
  rw [output]
  exact fieldResidue_centeredField_eq_embedCoefficient _

/-- Every machine challenge has the independently defined centered coefficient
range. This is not the pairwise Definition-17 strong-set theorem. -/
theorem machineChallenge_coefficients_valid
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    ProductionStrongSet.ScalarValid
      (machineScalar assignment canonical) :=
  ProductionStrongSet.everyScalarValid _

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly
