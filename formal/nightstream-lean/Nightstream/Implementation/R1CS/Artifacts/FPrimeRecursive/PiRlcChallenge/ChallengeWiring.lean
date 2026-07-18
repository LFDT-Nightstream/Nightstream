import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.Generated.ChallengeWiringData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.RhoEvaluations

/-!
Exact active wiring from PiRLC sampler outputs to projection rho inputs.

Owns: the physical protocol edge from one unique 15-by-54 block of
first-accepted selection outputs to the coefficient columns of the shared rho
evaluators, including its exact affine column formula.

Does not own: satisfaction or semantics of selection rows, Poseidon2 transcript
authority, rho evaluation soundness, projection identities, encoded costs, or
row removal.

Emits constraints: no.

Assurance tier: artifact-checked physical aliasing. The Rust drift gate checks
all 810 trace outputs; this facade checks that their exported formula equals
all 810 projection-consumer columns. Neither check makes a digest authoritative.

| Producer -> consumer | Mathematical obligation | Authority class | Count |
|---|---|---|---:|
| `challenge.sampler.selection.bind.symbol -> projection_shared.rho_evaluations` | selected output column equals the corresponding rho coefficient column | direct dataflow | 15 x 54 |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring

open Nightstream.Implementation.R1CS

namespace Generated

def producerStagePath : String :=
  FPrimeRecursivePiRlcChallengeWiringData.producerStagePath
def consumerStagePath : String :=
  FPrimeRecursivePiRlcChallengeWiringData.consumerStagePath
def selectionTraceStart : Nat :=
  FPrimeRecursivePiRlcChallengeWiringData.selectionTraceStart
def selectionTraceEnd : Nat :=
  FPrimeRecursivePiRlcChallengeWiringData.selectionTraceEnd
def outputBase : Nat :=
  FPrimeRecursivePiRlcChallengeWiringData.outputBase
def rhoStride : Nat :=
  FPrimeRecursivePiRlcChallengeWiringData.rhoStride
def coefficientStride : Nat :=
  FPrimeRecursivePiRlcChallengeWiringData.coefficientStride

end Generated

def rhoCount : Nat := 15
def coefficientCount : Nat := 54

def samplerOutputColumn
    (rho : Fin rhoCount) (coefficient : Fin coefficientCount) : Nat :=
  Generated.outputBase + Generated.rhoStride * rho.val +
    Generated.coefficientStride * coefficient.val

def samplerOutputColumnsFor (rho : Fin rhoCount) : List Nat :=
  List.ofFn fun coefficient : Fin coefficientCount =>
    samplerOutputColumn rho coefficient

def samplerOutputColumns : List (List Nat) :=
  List.ofFn samplerOutputColumnsFor

def projectionConsumerColumns : List (List Nat) :=
  FPrimeRecursivePiRlcProjection.RhoEvaluations.owners.map
    PiRlcRhoEvaluationOwner.coefficientColumns

def StructureValid : Prop :=
  Generated.producerStagePath =
      "nifs.pi_rlc.challenge.sampler.selection.bind.symbol" ∧
    Generated.consumerStagePath =
      "nifs.pi_rlc.verify.projection_shared.rho_evaluations" ∧
    Generated.selectionTraceStart < Generated.selectionTraceEnd ∧
    Generated.selectionTraceEnd - Generated.selectionTraceStart =
      rhoCount * coefficientCount ∧
    samplerOutputColumns.length = rhoCount ∧
    (∀ columns ∈ samplerOutputColumns,
      columns.length = coefficientCount) ∧
    projectionConsumerColumns = samplerOutputColumns

instance : Decidable StructureValid := by
  unfold StructureValid
  infer_instance

theorem structure_check : StructureValid := by
  set_option maxRecDepth 100000 in
    decide

theorem selection_trace_count :
    Generated.selectionTraceEnd - Generated.selectionTraceStart = 810 := by
  decide

theorem sampler_output_formula
    (rho : Fin rhoCount) (coefficient : Fin coefficientCount) :
    samplerOutputColumn rho coefficient =
      5045274 + 7984 * rho.val + 45 * coefficient.val := by
  rfl

theorem projection_consumers_are_sampler_outputs :
    projectionConsumerColumns = samplerOutputColumns :=
  structure_check.2.2.2.2.2.2

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring
