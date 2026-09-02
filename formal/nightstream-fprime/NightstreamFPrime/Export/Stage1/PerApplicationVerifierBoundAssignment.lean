import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO

/-!
Owns the verifier-side binding of the four recursive context words in one
per-application raw assignment. The caller supplies all ordinary raw values;
this module overwrites the context interval with the final verification-key
digest derived from the exact package, application, and Ajtai key.

This module does not select a concrete production application or close final
package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationVerifierBoundAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev RawValues := PerApplicationCanonicalAssignment.RawValues
abbrev FitsTwoPow28 (application : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 application
abbrev CommitmentSetup (application : Program) :=
  PerApplicationCanonicalPackage.CommitmentSetup application

def relation (application : Program) (fits : FitsTwoPow28 application) :=
  PerApplicationFixedPoint.relation application fits

/-- Exact four-word identifier read by the recursive NIFS verifier. -/
def verificationKeyDigest {application : Program}
    (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application) :
    KeyDigest :=
  (PerApplicationCanonicalPackage.verificationKeyBinding fits
    commitmentSetup).digest

/-- First final-package column of the four verifier-context words. -/
def contextTargetStart (application : Program) : Nat :=
  PerApplicationPackage.shiftColumn application Spartan.expectedContextPublicStart

/-- Overwrite only the four verifier-context columns. All other base values
remain caller supplied. -/
def bindContextBase (application : Program) (digest : KeyDigest)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) :
    Fin (PiRLCProductPlan.baseSourceWidth application) → F :=
  fun column =>
    if _inside : contextTargetStart application ≤ column.val ∧
        column.val < contextTargetStart application + 4 then
      digest.getD (column.val - contextTargetStart application) 0
    else
      base column

/-- Canonical raw packet with verifier-owned context words. -/
def bind {application : Program} (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (raw : RawValues application) : RawValues application :=
  { raw with
    base := bindContextBase application
      (verificationKeyDigest fits commitmentSetup) raw.base }

private theorem expectedContextSource
    (lane : Fin 4) : PiCCSOrdinarySourceSupport.Source
      (PiCCSInputs.expectedContextStart + lane.val) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_context
  exact ⟨by omega, by
    have bound := lane.isLt
    norm_num [PiCCSInputs.expectedContextWords] at bound ⊢
    ⟩

private theorem expectedContextTargetBound (lane : Fin 4) :
    Spartan.expectedContextPublicStart + lane.val <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have bound := lane.isLt
  have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
      29336725 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [total]
  norm_num [Spartan.expectedContextPublicStart] at bound ⊢
  omega

private theorem shiftedExpectedContext
    (application : Program) (lane : Fin 4) :
    PerApplicationPackage.shiftColumn application
        (Spartan.expectedContextPublicStart + lane.val) =
      contextTargetStart application + lane.val := by
  unfold contextTargetStart PerApplicationPackage.shiftColumn
  have laneBound := lane.isLt
  have lower : ¬ Spartan.expectedContextPublicStart + lane.val <
      PerApplicationPackage.basePackage.layout.constantColumn := by
    have constant : PerApplicationPackage.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    norm_num [Spartan.expectedContextPublicStart] at laneBound ⊢
    omega
  have startLower : ¬ Spartan.expectedContextPublicStart <
      PerApplicationPackage.basePackage.layout.constantColumn := by
    have constant : PerApplicationPackage.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    norm_num [Spartan.expectedContextPublicStart]
  rw [if_neg lower, if_neg startLower]
  omega

private theorem boundBase_expectedContext
    {application : Program} (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (raw : RawValues application) (lane : Fin 4) :
    (bind fits commitmentSetup raw).base
        (PiRLCProductPlan.shiftedPackageColumn application
          (Spartan.expectedContextPublicStart + lane.val)
          (expectedContextTargetBound lane)) =
      (verificationKeyDigest fits commitmentSetup).getD lane.val 0 := by
  have shifted := shiftedExpectedContext application lane
  change (if _inside : contextTargetStart application ≤
        PerApplicationPackage.shiftColumn application
          (Spartan.expectedContextPublicStart + lane.val) ∧
      PerApplicationPackage.shiftColumn application
          (Spartan.expectedContextPublicStart + lane.val) <
        contextTargetStart application + 4 then
      (verificationKeyDigest fits commitmentSetup).getD
        (PerApplicationPackage.shiftColumn application
          (Spartan.expectedContextPublicStart + lane.val) -
            contextTargetStart application) 0
    else raw.base _) =
      (verificationKeyDigest fits commitmentSetup).getD lane.val 0
  rw [dif_pos]
  · apply congrArg (fun index =>
      (verificationKeyDigest fits commitmentSetup).getD index 0)
    change PerApplicationPackage.shiftColumn application
        (Spartan.expectedContextPublicStart + lane.val) -
          contextTargetStart application = lane.val
    rw [shifted]
    omega
  · change contextTargetStart application ≤
        PerApplicationPackage.shiftColumn application
          (Spartan.expectedContextPublicStart + lane.val) ∧
      PerApplicationPackage.shiftColumn application
          (Spartan.expectedContextPublicStart + lane.val) <
        contextTargetStart application + 4
    rw [shifted]
    exact ⟨by omega, by have bound := lane.isLt; omega⟩

theorem transitionExpectedContext
    {application : Program} (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (raw : RawValues application) (lane : Fin 4) :
    PerApplicationDecodedIO.transitionEnv (bind fits commitmentSetup raw)
        (PiCCSInputs.expectedContextStart + lane.val) =
      (verificationKeyDigest fits commitmentSetup).getD lane.val 0 := by
  unfold PerApplicationDecodedIO.transitionEnv Spartan.pullback
  rw [Spartan.sourceToSpartan_expectedContext lane]
  unfold RunningTransitionDirectPlan.transitionEnv
  rw [dif_pos (expectedContextTargetBound lane)]
  exact boundBase_expectedContext fits commitmentSetup raw lane

/-- Accepted rows bind the decoded state context to the exact final
per-application verification-key digest. -/
theorem semantics_imply_contextKey
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (raw : RawValues application)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (PerApplicationFixedPoint.geometry application)
      (bind fits commitmentSetup raw).assignment
      (bind fits commitmentSetup raw).base
      (bind fits commitmentSetup raw).groupValue
      (bind fits commitmentSetup raw).products) :
    PerApplicationDecodedIO.contextKey (bind fits commitmentSetup raw) =
      verificationKeyDigest fits commitmentSetup := by
  let bound := bind fits commitmentSetup raw
  have piCcs :=
    DirectPiCCSCommonPhaseSemantics.semantics_imply_piCcsSpecHolds
      (relation application fits)
      (PerApplicationDecodedIO.prefixGeometry application) bound.assignment
      bound.base bound.groupValue bound.products semantics.runningPrefix
  have context := piCcs.statementBinding.state.priorContext
  unfold PerApplicationDecodedIO.contextKey StateDecoder.keyDigest
    StateDecoder.slice
  apply List.ext_get
  · simpa [PilotProduction.digestWords, PilotValues.digestWords,
      verificationKeyDigest] using
      (Lifecycle.Stage1.VerificationKey.Binding.digest_length
        (PerApplicationCanonicalPackage.verificationKeyBinding fits
          commitmentSetup)).symm
  · intro index leftBound rightBound
    let lane : Fin 4 := ⟨index, by simpa using leftBound⟩
    have row := context lane
    have custody :=
      PerApplicationDecodedIO.commonEnv_eq_transitionEnv_of_source bound
        (PiCCSInputs.expectedContextStart + lane.val)
        (expectedContextSource lane)
    have expected := transitionExpectedContext fits commitmentSetup raw lane
    rw [← custody] at expected
    have stateValue : PerApplicationDecodedIO.priorState bound
        (Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart + lane.val) =
      (verificationKeyDigest fits commitmentSetup).getD lane.val 0 := by
      calc
        PerApplicationDecodedIO.priorState bound
            (Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart + lane.val) =
          PerApplicationDecodedIO.commonEnv bound
            (PilotProduction.priorPreimageStart +
              (Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart + lane.val)) :=
          rfl
        _ = PerApplicationDecodedIO.commonEnv bound
            (PiCCSInputs.expectedContextStart + lane.val) := by
          simpa [PiCCSInvocations.parentInterface, PiCCSInputs.interface,
            Lifecycle.PiCCS.v1_1.Formal.statementBindingInterface,
            Lifecycle.PiCCS.v1_1.Formal.atOffset,
            PiCCSInputs.priorStateWord,
            Lifecycle.PiCCS.v1_1.StateBinding.contextWordStart,
            PiCCSInputs.expectedContext] using row
        _ = (verificationKeyDigest fits commitmentSetup).getD lane.val 0 :=
          expected
    have rightGet := List.getD_eq_get
      (verificationKeyDigest fits commitmentSetup) 0
      ⟨index, rightBound⟩
    simpa [lane] using stateValue.trans rightGet

end NightstreamFPrime.Export.Stage1.PerApplicationVerifierBoundAssignment
