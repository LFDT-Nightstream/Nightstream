import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Lifecycle.Nifs.WideFiatShamir

/-!
The actual local proof and current child witnesses for one recursive reverse
step. Terminal acceptance supplies the exact wide-key NIFS output and every
child membership. This is a deterministic event link, with no adversary
translation, probability, or work premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaRealInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Nifs
open NightstreamFPrime.Layout.Stage1

private def makeOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (proof : Lifecycle.Proof 9)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    WideFiatShamir.RealOutput relation :=
  ⟨proof, children⟩

private theorem success_of_verified_output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Lifecycle.Proof (ProductionKey.degreeBound relation))
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (verified : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai)
      running fresh proof = some result)
    (valid : ∀ child, CE.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.runningStatement relation result child) (children child)) :
    WideFiatShamir.RealSuccess relation ajtai running fresh
      (some (makeOutput relation proof children)) := by
  have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff
    (PiRLC.Wide.Key.key relation ajtai) running fresh proof result).mp verified
  rcases (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff
      (PiRLC.Wide.Key.key relation ajtai) running fresh proof).mp checks.2.1 with
    ⟨attempt, attemptEq, _attemptAccepted⟩
  exact ⟨result, attempt, verified, attemptEq, valid⟩

variable (target : Wide.Target)

/-- The decoded local proof with the terminal's existing ordered sixteen
child witnesses. No claimed output or source witness is added. -/
def output (payload : target.Payload) : WideFiatShamir.RealOutput target.relation :=
  makeOutput target.relation (target.decodedInput payload).nifsProof
    (payload.runningWitness functionIndex)

/-- A non-base accepted terminal opening supplies the actual wide-key NIFS
real-success event for the same prior input used by the reverse history.
Child correctness and verifier-output equality are derived from acceptance. -/
theorem realSuccess_of_terminal (statement : HyperNovaHistory.Statement) (payload : target.Payload)
    (accepted : target.Holds statement (.recursive payload))
    (safe : ¬ target.Collision statement payload)
    (positive : 0 < (target.decodedInput payload).iteration) :
    WideFiatShamir.RealSuccess target.relation target.ajtai
      (target.security.running (HyperNovaHistory.sourceInput target payload))
      (target.security.fresh (HyperNovaHistory.sourceInput target payload))
      (some (output target payload)) := by
  rcases target.terminal_implies_nifsOrBaseOrCollision statement payload accepted with
    (base | ⟨_positive, verified⟩) | collision
  · exact False.elim ((Nat.ne_of_gt positive) base)
  · rcases (Stage1.Terminal.holdsFor_recursive_iff target.relation target.ajtai target.context
      target.program statement payload).mp accepted with
      ⟨_statementValid, _pcValid, _iteration, _publicLink, runningValid, _freshValid⟩
    have success := success_of_verified_output target.relation target.ajtai
      ((target.decodedInput payload).running functionIndex) (target.decodedInput payload).fresh
      (target.decodedInput payload).nifsProof (payload.running functionIndex)
      (payload.runningWitness functionIndex) verified (runningValid functionIndex)
    simp only [SecurityInstance.running, SecurityInstance.fresh, HyperNovaHistory.sourceInput,
      HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims]
    exact success
  · exact False.elim (safe collision)

end NightstreamFPrime.Export.Stage1.HyperNovaRealInput
