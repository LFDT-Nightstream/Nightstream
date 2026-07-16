import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.RingEncoding

/-!
Terminal `Pi_RLC` coefficient-vector assembly into the concrete Phi81 ring.

Assurance tier: implementation/R1CS correspondence. This module connects the
verifier-derived first-accepted coefficient vectors to the exact production
challenge columns and then embeds each complete 54-coordinate vector into
`Concrete.RingF`.

Owns: the exact terminal challenge-column formula; refinement to the
independently defined centered Goldilocks embedding; all fifteen 54-coordinate
ring challenges; and equality between every decoded production challenge
coefficient and its transcript-machine-derived coefficient.

Does not own: the post-PiCCS initial transcript state, quotient-ring
invertibility of pairwise challenge differences, projection identities,
native Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: the ring challenge is assembled only from the first 54
accepted symbols of the connected verifier-owned Poseidon2 machine. Production
columns are outputs to be interpreted by the theorem, never independent
challenge authority.

| Protocol | Phase | Constraint family | Indexed leaf | Exact obligation |
|---|---|---|---|---|
| `Pi_RLC` | challenge output | column placement | `rho : Fin 15`, coefficient `Fin 54` | `2560849 + 7984*rho + 45*coefficient` |
| `Pi_RLC` | coefficient encoding | centered Goldilocks value | one five-symbol coefficient | canonical field image of `{-2,-1,0,1,2}` |
| `Pi_RLC` | ring assembly | Phi81 coefficient vector | one scalar | exactly 54 field coefficients in source order |
| `Pi_RLC` | full batch | verifier-derived challenges | fifteen scalars | every decoded production coefficient equals the connected machine output |
| `Pi_RLC` | strong-set precursor | coefficient predicate | every scalar position | independently proved centered range `[-2,2]` |
| `Pi_RLC` | semantic bridge | Phi81 scalar embedding | one scalar | machine ring equals the independent semantic embedding |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.RingEncoding
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- The exact global column consumed as one terminal quotient-ring
coefficient. The definition follows the readable tail relabeling; the closed
formula below exposes the production address directly. -/
def challengeColumn
    (rho : Fin ScalarRows.scalarCount)
    (position : Fin ProductionAlphabet.coefficientCount) : Nat :=
  Relabel.column
    (AlphabetSamplingResidualTemplate.tailColumnMap
      (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho))
    (SelectionRows.outputCol position.val)

/-- Closed affine address of every one of the `15 * 54` challenge outputs. -/
theorem challengeColumn_formula :
    forall (rho : Fin ScalarRows.scalarCount)
      (position : Fin ProductionAlphabet.coefficientCount),
      challengeColumn rho position =
        2560849 + 7984 * rho.val + 45 * position.val := by
  decide

/-- Exact global challenge columns for one scalar in coefficient order. -/
def challengeColumns
    (rho : Fin ScalarRows.scalarCount) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    challengeColumn rho position

@[simp] theorem challengeColumns_length
    (rho : Fin ScalarRows.scalarCount) :
    (challengeColumns rho).length = ProductionAlphabet.coefficientCount := by
  simp [challengeColumns]

/-- The local readable tail and the global production assignment select the
same output column. -/
theorem localOutput_eq_assignment
    (assignment : Nat -> Nat)
    (rho : Fin ScalarRows.scalarCount)
    (position : Fin ProductionAlphabet.coefficientCount) :
    TailRows.localAssignment assignment rho
        (SelectionRows.outputCol position.val) =
      assignment (challengeColumn rho position) := by
  rfl

/-! Compatibility exports for the shared encoding owner. Keeping these names
here makes the terminal facade stable while the definitions themselves have a
single implementation. -/

abbrev fieldResidue := RingEncoding.fieldResidue
abbrev embedCoefficient := RingEncoding.embedCoefficient

theorem embedCoefficient_eq_semantic
    (coefficient : ProductionAlphabet.Coefficient) :
    embedCoefficient coefficient =
      Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient coefficient :=
  RingEncoding.embedCoefficient_eq_semantic coefficient

theorem fieldResidue_centeredField_eq_embedCoefficient
    (coefficient : ProductionAlphabet.Coefficient) :
    fieldResidue (CandidateOrder.centeredField coefficient) =
      embedCoefficient coefficient :=
  RingEncoding.fieldResidue_centeredField_eq_embedCoefficient coefficient

theorem embeddedAlphabet_values :
    (List.ofFn fun coefficient : Fin ProductionAlphabet.alphabetSize =>
      (embedCoefficient coefficient).val) =
      [goldilocksP - 2, goldilocksP - 1, 0, 1, 2] :=
  RingEncoding.embeddedAlphabet_values

/-- One complete verifier-derived scalar embedded coefficientwise in the
concrete Phi81 quotient-ring carrier. -/
def machineChallenge
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : RingF :=
  fun position =>
    embedCoefficient (MachineOutput.scalar assignment canonical rho position)

/-- The implementation-facing ring function is definitionally the independent
semantic embedding of the complete 54-symbol machine output. -/
theorem machineChallenge_eq_embedScalar
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) :
    machineChallenge assignment canonical rho =
      Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar
        (MachineOutput.scalar assignment canonical rho) := by
  funext position
  rfl

/-- Direct field decoding of the production challenge columns. -/
def decodedChallenge
    (assignment : Nat -> Nat)
    (rho : Fin ScalarRows.scalarCount) : RingF :=
  fun position => fieldResidue (assignment (challengeColumn rho position))

/-- Accepted terminal rows and the connected transcript schedule force every
decoded production coefficient to equal its machine-derived coefficient. -/
theorem decodedChallenge_eq_machineChallenge
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    decodedChallenge assignment rho =
      machineChallenge assignment canonical rho := by
  funext position
  have output := MachineOutput.outputAt_refines
    prime canonical one accepted rho position
  rw [localOutput_eq_assignment assignment rho position] at output
  unfold decodedChallenge machineChallenge embedCoefficient
  rw [output]
  exact fieldResidue_centeredField_eq_embedCoefficient _

/-- Batch form: all fifteen terminal quotient-ring challenges are determined
by the verifier-owned machine and the independently proved selector semantics. -/
theorem decodedChallenges_eq_machineChallenges
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    (fun rho : Fin ScalarRows.scalarCount => decodedChallenge assignment rho) =
      (fun rho : Fin ScalarRows.scalarCount =>
        machineChallenge assignment canonical rho) := by
  funext rho
  exact decodedChallenge_eq_machineChallenge
    prime canonical one accepted rho

/-- Every assembled machine challenge satisfies the independent
coefficient-level range predicate. This is deliberately weaker than the
paper's quotient-ring strong-set law. -/
theorem machineChallenge_coefficients_valid
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) :
    ProductionStrongSet.ScalarValid
      (MachineOutput.scalar assignment canonical rho) :=
  ProductionStrongSet.everyScalarValid _

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly
