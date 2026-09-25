import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
import NightstreamFPrime.Lifecycle.Nifs.FiatShamirTransfer

/-!
The actual local proof and current child witnesses for one recursive reverse
step. Terminal acceptance supplies the exact NIFS output and every child
membership. This is a deterministic event link, with no adversary translation,
probability, or work premise.
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
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private def makeOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (proof : Lifecycle.Proof 9)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    FiatShamirTransfer.RealOutput relation :=
  ⟨proof, children⟩

private theorem success_of_relation_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : ProductionKey.LogicalRelation logicalWidth publicFits)
    (same : left = right)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Lifecycle.Proof 9)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (success : FiatShamirTransfer.RealSuccess right ajtai running fresh
      (some (makeOutput right proof children))) :
    FiatShamirTransfer.RealSuccess left ajtai running fresh
      (some (makeOutput left proof children)) := by
  cases same
  exact success

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
    (verified : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      running fresh proof = some result)
    (valid : ∀ child, CE.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.runningStatement relation result child) (children child)) :
    FiatShamirTransfer.RealSuccess relation ajtai running fresh
      (some (makeOutput relation proof children)) := by
  have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff
    (ProductionKey.key relation ajtai) running fresh proof result).mp verified
  rcases (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff
      (ProductionKey.key relation ajtai) running fresh proof).mp checks.2.1 with
    ⟨attempt, attemptEq, _attemptAccepted⟩
  refine ⟨result, attempt, verified, attemptEq, ?_⟩
  intro child
  rw [Lifecycle.PiDEC.v1_1.OutputWitnessConsumer.runningStatement_eq
    relation ajtai result child]
  exact valid child

/-- The decoded local proof with the terminal's existing ordered sixteen
child witnesses. No claimed output or source witness is added. -/
def output (payload : HyperNovaHistory.Payload) :
    FiatShamirTransfer.RealOutput PiDECInputCheck.relation :=
  makeOutput PiDECInputCheck.relation
    (HyperNovaHistory.decodedInput payload).nifsProof (payload.runningWitness functionIndex)

/-- A non-base accepted terminal opening supplies the actual NIFS real-success
event for the same prior input used by the reverse history. Child correctness
and verifier-output equality are derived from acceptance, not assumed. -/
theorem realSuccess_of_terminal
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive payload))
    (safe : ¬ HyperNovaHistory.Collision statement payload)
    (positive : 0 < (HyperNovaHistory.decodedInput payload).iteration) :
    FiatShamirTransfer.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PiCCSInputCheck.running (HyperNovaHistory.sourceInput payload))
      (PiCCSInputCheck.fresh (HyperNovaHistory.sourceInput payload))
      (some (output payload)) := by
  dsimp only [HyperNovaHistory.decodedInput] at positive
  dsimp only [HyperNovaHistory.Collision] at safe
  rcases ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision
      application fits productionSetup statement payload accepted with
    ⟨_context, base | recursive⟩ | collision
  · exact False.elim ((Nat.ne_of_gt positive) base)
  · rcases recursive with ⟨_positive, _priorPublic, _priorDigest, verified⟩
    rcases (PerApplicationTerminal.holds_recursive_iff application fits
      productionSetup statement payload).mp accepted with
      ⟨_statementValid, _pcValid, _iteration, _publicLink, runningValid, _freshValid⟩
    have success := success_of_verified_output
      (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey productionSetup)
      ((HyperNovaHistory.decodedInput payload).running functionIndex)
      (HyperNovaHistory.decodedInput payload).fresh
      (HyperNovaHistory.decodedInput payload).nifsProof
      (payload.running functionIndex) (payload.runningWitness functionIndex)
      verified (runningValid functionIndex)
    have selected := success_of_relation_eq PiDECInputCheck.relation
      (PerApplicationFixedPoint.relation application fits) PiDECInputCheck.relation_eq_selected
      productionAjtaiKey ((HyperNovaHistory.decodedInput payload).running functionIndex)
      (HyperNovaHistory.decodedInput payload).fresh
      (HyperNovaHistory.decodedInput payload).nifsProof
      (payload.runningWitness functionIndex) success
    simpa only [output, HyperNovaHistory.sourceInput,
      HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims] using selected
  · exact False.elim (safe collision)

end NightstreamFPrime.Export.Stage1.HyperNovaRealInput
