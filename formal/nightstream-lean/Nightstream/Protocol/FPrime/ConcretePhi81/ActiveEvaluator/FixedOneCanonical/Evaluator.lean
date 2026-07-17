import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Physical
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator

/-!
Fail-closed evaluator for the payload-minimal fixed-one F-prime verifier.

Owns: canonical output construction and the exact characterization of every
successful execution.

Does not own: physical equation definitions, semantic source truth,
bad-event bounds, Rust/R1CS refinement, physical rows, costs, or row removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.output.result` | use the exact checked NIFS result | computed | `run` |
| `fprime.fixed_one.output.outer` | derive state, counter, inactive slots, and digest | computed | `run` |
| `fprime.fixed_one.output.exact` | successful execution iff physical checks and canonical output equality hold | exact model theorem | `run_eq_some_iff_physicalChecks` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Complete named meaning of one successful canonical output. -/
structure PhysicalChecks
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (result : Slot shape publicRingColumns publicFits verifierRows)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows 1) :
    Prop where
  outer : OuterChecks machine setup input
  nifsAccepted : ConcretePhi81.Accepted
    (nifsContext setup input).materialize certificate
  resultExact :
    FixedActive.resultOf (nifsContext setup input).materialize certificate =
      result
  outputExact :
    output = ActiveSemantics.outputOf machine (input.toActive setup)
      ActiveSemantics.FixedOneCanonical.selected result

/-- Fail-closed evaluator. It computes the complete selected NIFS result and
the complete outer F-prime output. -/
def run
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) :
    Option
      (Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) :=
  if outerCheck machine setup input then
    (FixedActive.Evaluator.run
      (FixedActive.Canonical.Checker.evaluatorChecker
        (nifsContext setup input)) certificate).map
      (ActiveSemantics.outputOf machine (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected)
  else
    none

/-- Successful evaluation is exactly the named physical checks and canonical
output equality. -/
theorem run_eq_some_iff_physicalChecks
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) :
    run machine setup input certificate = some output <->
      exists result : Slot shape publicRingColumns publicFits verifierRows,
        PhysicalChecks machine setup input certificate result output := by
  cases outerChecked : outerCheck machine setup input with
  | false =>
      constructor
      · intro executed
        simp [run, outerChecked] at executed
      · rintro ⟨result, physical⟩
        have checked : outerCheck machine setup input = true :=
          (outerCheck_eq_true_iff machine setup input).2 physical.outer
        simp [outerChecked] at checked
  | true =>
      have outer : OuterChecks machine setup input :=
        (outerCheck_eq_true_iff machine setup input).1 outerChecked
      cases nifsExecuted :
          FixedActive.Evaluator.run
            (FixedActive.Canonical.Checker.evaluatorChecker
              (nifsContext setup input)) certificate with
      | none =>
          constructor
          · intro executed
            simp [run, outerChecked, nifsExecuted] at executed
          · rintro ⟨result, physical⟩
            have executed :
                FixedActive.Evaluator.run
                    (FixedActive.Canonical.Checker.evaluatorChecker
                      (nifsContext setup input)) certificate =
                  some result :=
              (FixedActive.Evaluator.run_eq_some_iff_accepted
                (FixedActive.Canonical.Checker.evaluatorChecker
                  (nifsContext setup input)) certificate result).2
                ⟨physical.nifsAccepted, physical.resultExact⟩
            rw [nifsExecuted] at executed
            contradiction
      | some result =>
          have nifsMeaning :=
            (FixedActive.Evaluator.run_eq_some_iff_accepted
              (FixedActive.Canonical.Checker.evaluatorChecker
                (nifsContext setup input)) certificate result).1 nifsExecuted
          constructor
          · intro executed
            have outputEq :
                output = ActiveSemantics.outputOf machine
                  (input.toActive setup)
                  ActiveSemantics.FixedOneCanonical.selected result := by
              simpa [run, outerChecked, nifsExecuted] using executed.symm
            exact ⟨result, {
              outer := outer
              nifsAccepted := nifsMeaning.1
              resultExact := nifsMeaning.2
              outputExact := outputEq
            }⟩
          · rintro ⟨selectedResult, physical⟩
            have resultEq : selectedResult = result :=
              physical.resultExact.symm.trans nifsMeaning.2
            subst selectedResult
            simpa [run, outerChecked, nifsExecuted] using
              physical.outputExact.symm

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
