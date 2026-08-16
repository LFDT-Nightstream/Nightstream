import Mathlib.Algebra.BigOperators.Fin
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingClaimSchedule

/-!
Contract: complete 86-step PiCCS coordinate-commitment accumulator.

Owns the phase-local partial commitment value, additive accumulator step, exact
zero initial state, and proof that all 86 verifier-owned claim masks produce
the direct 21,220-field production commitment.

Does not own physical overlay rows, phase-to-overlay selection, equality links
to claim-replay chunk columns, recovery of the authoritative claim fields,
Module-SIS hardness, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance : CommRing F :=
  CommRing.ofMinimalAxioms
    ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.zero_add
    Lean.Grind.Fin.neg_add_cancel
    ConcreteCarrier.baseLaws.mul_assoc
    ConcreteCarrier.baseLaws.mul_comm
    ConcreteCarrier.baseLaws.one_mul
    ConcreteCarrier.baseLaws.left_distrib

abbrev Accumulator := Fin (shape.rows * shape.degree) → F

def zeroAccumulator : Accumulator := fun _ => 0

/-- One output coordinate of the exact commitment selected by one claim
chunk. -/
def partialCoordinate
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount)
    (output : Fin (shape.rows * shape.degree)) : F :=
  let pair := outputPair output
  (commit (seededMatrix production.setup) coefficientMap
    (maskedWitness fields (chunkMask chunk)) pair.1).coefficients pair.2

/-- One output coordinate of the direct complete 21,220-field commitment. -/
def directCoordinate
    (production : ProductionSetup) (fields : Fields)
    (output : Fin (shape.rows * shape.degree)) : F :=
  let pair := outputPair output
  (bindingMap (seededMatrix production.setup) coefficientMap fields
    pair.1).coefficients pair.2

private def ringCoordinate (coordinate : Fin ringDegree) :
    ExecutablePhi81.Ring →+ F where
  toFun value := value.coefficients coordinate
  map_zero' := rfl
  map_add' := by
    intro left right
    rfl

private theorem coefficients_sum
    {count : Nat} (values : Fin count → ExecutablePhi81.Ring)
    (coordinate : Fin ringDegree) :
    ((∑ index, values index).coefficients coordinate) =
      ∑ index, (values index).coefficients coordinate := by
  simpa only [ringCoordinate] using
    (map_sum (ringCoordinate coordinate) values Finset.univ)

/-- The coordinate form of the claim-mask partition theorem. -/
theorem partialCoordinate_sum
    (production : ProductionSetup) (fields : Fields)
    (output : Fin (shape.rows * shape.degree)) :
    (∑ chunk : Fin claimChunkCount,
      partialCoordinate production fields chunk output) =
        directCoordinate production fields output := by
  let pair := outputPair output
  change
    (∑ chunk : Fin claimChunkCount,
      (commit (seededMatrix production.setup) coefficientMap
        (maskedWitness fields (chunkMask chunk)) pair.1).coefficients pair.2) =
      (bindingMap (seededMatrix production.setup) coefficientMap fields
        pair.1).coefficients pair.2
  rw [← coefficients_sum]
  exact congrArg (fun value => value.coefficients pair.2)
    (congrFun (concrete_commitments_sum production fields) pair.1)

/-- Exact local algebra required from one selected coordinate overlay. -/
def StepAt
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount)
    (before after : Accumulator) : Prop :=
  ∀ output,
    after output = before output +
      partialCoordinate production fields chunk output

def partialAtNat
    (production : ProductionSetup) (fields : Fields)
    (index : Nat) (output : Fin (shape.rows * shape.degree)) : F :=
  if bound : index < claimChunkCount then
    partialCoordinate production fields ⟨index, bound⟩ output
  else
    0

/-- Canonical accumulator after the first `count` claim chunks. -/
def accumulated
    (production : ProductionSetup) (fields : Fields)
    (count : Nat) : Accumulator :=
  fun output =>
    ∑ index ∈ Finset.range count,
      partialAtNat production fields index output

theorem accumulated_zero
    (production : ProductionSetup) (fields : Fields) :
    accumulated production fields 0 = zeroAccumulator := by
  funext output
  simp [accumulated, zeroAccumulator]

theorem accumulated_succ
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount) :
    StepAt production fields chunk
      (accumulated production fields chunk.val)
      (accumulated production fields (chunk.val + 1)) := by
  intro output
  simp [accumulated, partialAtNat, Finset.sum_range_succ, chunk.isLt]

/-- A physical run can use separate assignments at every phase. The public
state links must make their accumulator values one ordered chain. -/
structure AcceptedRun
    (production : ProductionSetup) (fields : Fields) where
  state : Nat → Accumulator
  initial : state 0 = zeroAccumulator
  step : ∀ chunk : Fin claimChunkCount,
    StepAt production fields chunk (state chunk.val) (state (chunk.val + 1))

namespace AcceptedRun

theorem state_eq_accumulated
    {production : ProductionSetup} {fields : Fields}
    (run : AcceptedRun production fields)
    (count : Nat) (bound : count ≤ claimChunkCount) :
    run.state count = accumulated production fields count := by
  induction count with
  | zero =>
      rw [run.initial, accumulated_zero]
  | succ count inductionHypothesis =>
      have countBound : count < claimChunkCount := by omega
      let chunk : Fin claimChunkCount := ⟨count, countBound⟩
      funext output
      calc
        run.state (count + 1) output =
            run.state count output +
              partialCoordinate production fields chunk output :=
          run.step chunk output
        _ = accumulated production fields count output +
              partialCoordinate production fields chunk output := by
          rw [inductionHypothesis (by omega)]
        _ = accumulated production fields (count + 1) output :=
          (accumulated_succ production fields chunk output).symm

/-- A complete honest chain cannot stop at a self-consistent digest. Its
carried algebraic value is the direct full-vector commitment. -/
theorem final_eq_direct
    {production : ProductionSetup} {fields : Fields}
    (run : AcceptedRun production fields) :
    run.state claimChunkCount = directCoordinate production fields := by
  rw [run.state_eq_accumulated claimChunkCount (by rfl)]
  funext output
  unfold accumulated
  rw [← Fin.sum_univ_eq_sum_range]
  simpa [partialAtNat] using partialCoordinate_sum production fields output

end AcceptedRun

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator
