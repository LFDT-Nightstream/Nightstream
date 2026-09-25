import NightstreamFPrime.Lifecycle.Relation

/-!
Owns the deterministic Stage 1 accumulator update.

SuperNeo's accumulator output is exactly the `k` PiDEC child CE instances.
HyperNova installs that one NIFS verifier output into the selected running
slot. This module names that existing verifier graph; it adds no accumulator
digest, verifier check, or circuit row.

Security extraction, base-branch selection, running-slot selection, and the
final recursive fixed point belong to separate Stage 1 results.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.Accumulator

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The exact computed SuperNeo accumulator update consumed by HyperNova. -/
noncomputable def Holds
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (_vk : KeyDigest)
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (degreeBound relation))
    (output : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  Nifs.PaperNonInteractive.verify
    (key relation ajtai) running fresh proof = some output

/-- Accumulator acceptance is exactly the production PiCCS check, the strict
PiDEC check over the verifier-computed PiRLC parent, and the computed PiDEC
running output. -/
theorem holds_iff_checks
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest)
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (degreeBound relation))
    (output : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Holds relation ajtai vk running fresh proof output ↔
      Nifs.PaperNonInteractive.piCcsCheck
          (key relation ajtai) running fresh proof = true ∧
        Nifs.PaperNonInteractive.piDecCheck
          (key relation ajtai) running fresh proof = true ∧
        (key relation ajtai).output running fresh proof = some output := by
  unfold Holds
  exact Nifs.PaperNonInteractive.verify_eq_some_iff
    (key relation ajtai) running fresh proof output

/-- One fixed NIFS proof has at most one accepted accumulator output. -/
theorem output_unique
    (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest)
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (degreeBound relation))
    {left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (leftHolds : Holds relation ajtai vk running fresh proof left)
    (rightHolds : Holds relation ajtai vk running fresh proof right) :
    left = right := by
  unfold Holds at leftHolds rightHolds
  exact Option.some.inj (leftHolds.symm.trans rightHolds)

end

end NightstreamFPrime.Lifecycle.Stage1.Accumulator
