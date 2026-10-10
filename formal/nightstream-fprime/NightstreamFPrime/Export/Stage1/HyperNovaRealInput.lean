import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
import NightstreamFPrime.Export.Stage1.NifsRealSuccess

/-!
The actual local proof and current child witnesses for one recursive reverse
step. Terminal acceptance supplies the prior-state link, the exact NIFS output
and every child membership. This is a deterministic event link, with no
adversary translation, probability, or work premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaRealInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private def makeOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Lifecycle.Proof 8)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    NifsRealSuccess.RealOutput relation :=
  ⟨prior, proof, children⟩

private theorem success_of_relation_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : ProductionKey.LogicalRelation logicalWidth publicFits)
    (same : left = right)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Lifecycle.Proof 8)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (success : NifsRealSuccess.RealSuccess right ajtai context running fresh
      (some (makeOutput right prior proof children))) :
    NifsRealSuccess.RealSuccess left ajtai context running fresh
      (some (makeOutput left prior proof children)) := by
  cases same
  exact success

private theorem success_of_verified_output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : KeyDigest)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Lifecycle.Proof (ProductionKey.degreeBound relation))
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (link : PiCCSSecurity.PriorLink prior running fresh context)
    (verified : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      running fresh proof = some result)
    (valid : ∀ child, CE.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.runningStatement relation result child) (children child)) :
    NifsRealSuccess.RealSuccess relation ajtai context running fresh
      (some (makeOutput relation prior proof children)) := by
  have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff
    (ProductionKey.key relation ajtai) running fresh proof result).mp verified
  rcases (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff
      (ProductionKey.key relation ajtai) running fresh proof).mp checks.2.1 with
    ⟨attempt, attemptEq, _attemptAccepted⟩
  refine ⟨link, result, attempt, verified, attemptEq, ?_⟩
  intro child
  rw [Lifecycle.PiDEC.v1_2.OutputWitnessConsumer.runningStatement_eq
    relation ajtai result child]
  exact valid child

/-- The prior preimage that the decoded local step names under the package's
verifier context: the preimage whose hash the terminal checks. -/
noncomputable def prior (payload : HyperNovaHistory.Payload) :=
  HyperNova.Construction2.Paper.priorHashPreimage
    (Lifecycle.setup (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey productionSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup))
    (HyperNovaHistory.decodedInput payload)

/-- The decoded prior preimage and local proof with the terminal's existing
ordered sixteen child witnesses. No claimed output or source witness is added. -/
noncomputable def output (payload : HyperNovaHistory.Payload) :
    NifsRealSuccess.RealOutput PiDECInputCheck.relation :=
  makeOutput PiDECInputCheck.relation (prior payload)
    (HyperNovaHistory.decodedInput payload).nifsProof (payload.runningWitness functionIndex)

/-- A non-base accepted terminal opening supplies the actual NIFS real-success
event for the same prior input used by the reverse history. The prior-state
link, child correctness and verifier-output equality are derived from
acceptance, not assumed. -/
theorem realSuccess_of_terminal
    (statement : HyperNovaHistory.Statement) (payload : HyperNovaHistory.Payload)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive payload))
    (safe : ¬ HyperNovaHistory.Collision statement payload)
    (positive : 0 < (HyperNovaHistory.decodedInput payload).iteration) :
    NifsRealSuccess.RealSuccess PiDECInputCheck.relation productionAjtaiKey
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
      (PiCCSInputCheck.running (HyperNovaHistory.sourceInput payload))
      (PiCCSInputCheck.fresh (HyperNovaHistory.sourceInput payload))
      (some (output payload)) := by
  dsimp only [HyperNovaHistory.decodedInput] at positive
  dsimp only [HyperNovaHistory.Collision] at safe
  rcases ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision
      application fits productionSetup statement payload accepted with
    ⟨_context, base | recursive⟩ | collision
  · exact False.elim ((Nat.ne_of_gt positive) base)
  · rcases recursive with ⟨_positive, _priorPublic, link, verified⟩
    rcases (PerApplicationTerminal.holds_recursive_iff application fits
      productionSetup statement payload).mp accepted with
      ⟨_statementValid, _canonical, _pcValid, _iteration, _publicLink, runningValid, _freshValid⟩
    have success := success_of_verified_output
      (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey productionSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
      ((HyperNovaHistory.decodedInput payload).running functionIndex)
      (HyperNovaHistory.decodedInput payload).fresh (prior payload)
      (HyperNovaHistory.decodedInput payload).nifsProof
      (payload.running functionIndex) (payload.runningWitness functionIndex)
      link verified (runningValid functionIndex)
    have selected := success_of_relation_eq PiDECInputCheck.relation
      (PerApplicationFixedPoint.relation application fits) PiDECInputCheck.relation_eq_selected
      productionAjtaiKey (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
      ((HyperNovaHistory.decodedInput payload).running functionIndex)
      (HyperNovaHistory.decodedInput payload).fresh (prior payload)
      (HyperNovaHistory.decodedInput payload).nifsProof
      (payload.runningWitness functionIndex) success
    simpa only [output, HyperNovaHistory.sourceInput,
      HyperNovaInput.running_ofClaims, HyperNovaInput.fresh_ofClaims] using selected
  · exact False.elim (safe collision)

end NightstreamFPrime.Export.Stage1.HyperNovaRealInput
