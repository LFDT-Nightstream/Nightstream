import Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurity

/-!
Countermodels for the exact release-security composition theorem.

The first family shows that each deterministic event needs a proof of
impossibility. Such an event is a public bad event but is not a computational
budget item. The second family shows that each computational event can have
probability one unless its game bound is supplied.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels

open Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurity
open Nightstream.Protocol.NebulaV2.Soundness

def Only (selected : BadEvent) : BadEvent → Unit → Prop :=
  fun event _ => event = selected

inductive DeterministicCase where
  | decode
  | circuitRefinement
  | applicationPortCoverage
  | fPrimeLifecycle
  | recursiveSizeClosure
  | bundlePropagation
deriving DecidableEq, Fintype, Repr

def DeterministicCase.event : DeterministicCase → BadEvent
  | .decode => .decode
  | .circuitRefinement => .circuitRefinement
  | .applicationPortCoverage => .applicationPortCoverage
  | .fPrimeLifecycle => .fPrimeLifecycle
  | .recursiveSizeClosure => .recursiveSizeClosure
  | .bundlePropagation => .bundlePropagation

/-- Omitting any one deterministic closure field leaves a bad event that the
computational union does not cover. -/
theorem every_deterministic_event_requires_closure
    (selected : DeterministicCase) :
    AnyBad (fun event => Only selected.event event ()) ∧
      ¬ ComputationalAny (Only selected.event) () := by
  constructor
  · exact ⟨selected.event, rfl⟩
  · cases selected <;>
      simp [Only, DeterministicCase.event, ComputationalAny, SetupFamily]

inductive ComputationalCase where
  | commitmentBinding
  | compactTokenBinding
  | seededSetup
  | poseidonOrTranscript
  | memoryFingerprint
  | piRlcSampler
  | foldExtraction
  | terminalBackend
deriving DecidableEq, Fintype, Repr

def ComputationalCase.event : ComputationalCase → BadEvent
  | .commitmentBinding => .commitmentBinding
  | .compactTokenBinding => .compactTokenBinding
  | .seededSetup => .seededSetup
  | .poseidonOrTranscript => .poseidonOrTranscript
  | .memoryFingerprint => .memoryFingerprint
  | .piRlcSampler => .piRlcSampler
  | .foldExtraction => .foldExtraction
  | .terminalBackend => .terminalBackend

/-- Constructor inspection confirms that all eight computational events are
inside the six-family union. -/
theorem every_computational_event_is_covered
    (selected : ComputationalCase) :
    ComputationalAny (Only selected.event) () := by
  cases selected <;>
    simp [Only, ComputationalCase.event, ComputationalAny, SetupFamily]

/-- Exact probability on the one-point outcome space. -/
noncomputable def unitProbabilityModel : ProbabilityModel Unit := by
  classical
  refine
    { probability := fun event => if event () then 1 else 0
      nonnegative := ?_
      impossible := ?_
      certain := ?_
      monotone := ?_
      unionBound := ?_ }
  · intro event
    by_cases holds : event () <;> simp [holds]
  · simp
  · simp
  · intro left right included
    by_cases leftHolds : left ()
    · have rightHolds := included () leftHolds
      simp [leftHolds, rightHolds]
    · by_cases rightHolds : right () <;>
        simp [leftHolds, rightHolds]
  · intro left right
    by_cases leftHolds : left () <;>
      by_cases rightHolds : right () <;>
        simp [leftHolds, rightHolds]

/-- Without the corresponding game bound, each computational event family
can consume the complete probability mass. -/
theorem every_computational_event_can_have_probability_one
    (selected : ComputationalCase) :
    unitProbabilityModel.probability
        (ComputationalAny (Only selected.event)) = 1 := by
  classical
  simp [unitProbabilityModel, every_computational_event_is_covered selected]

end Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels
