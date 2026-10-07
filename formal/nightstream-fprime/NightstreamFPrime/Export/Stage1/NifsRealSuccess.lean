import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer

/-!
Owns the real success event of one production NIFS fold: the adversary
outputs a prior preimage whose prior-state link (`PiCCSSecurity.PriorLink`)
holds for the running statement and the verifier context digest, the actual
`ProductionKey` verifier accepts, and the adversary's witnesses open the exact
sixteen returned children. Bare public acceptance is not this event.

The prior link is required (owner decision 2026-10-06). No challenge depends
on the running statement, so without the link a prover could choose it after
γ and keep the claimed sum. The preimage must be an output, not merely exist:
its initial state, current state and iteration are free, so some well-formed
preimage with any running vector hashes to almost every digest.

Outputs: `RealSuccess`, `realSuccessProbability` under an adversary output
law, and `contextLaw`, the law's context marginal. The visited history laws
read the event through `HyperNovaVisitedAcceptance`.

Does not own: any extractor, Fiat–Shamir assumption or probability bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsRealSuccess

open scoped BigOperators
attribute [local instance] Classical.propDecidable

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The adversary's prior preimage, the existing proof data, and witnesses
for the existing ordered output. The preimage is the HyperNova Construction 2
state that the IVC prover holds; this adds no NIFS message, claimed verifier
output, or representation. -/
structure RealOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits) where
  prior : Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)
  proof : PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
    (ProductionKey.degreeBound relation)
  children : Fin productionShape.runningCount →
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (contextDigest : KeyDigest)

/-- The output prior preimage links the running statement and the verifier
context to the absorbed digest, the actual NIFS verifier accepts, and the
supplied witnesses open its exact sixteen returned children. The same proof
supplies the PiDEC attempt. -/
def RealSuccess
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Option (RealOutput relation) → Prop
  | none => False
  | some output =>
      let key := ProductionKey.key relation ajtai
      Layout.Stage1.PiCCSSecurity.PriorLink output.prior running fresh contextDigest ∧
      ∃ result attempt,
        PaperNonInteractive.verify key running fresh output.proof = some result ∧
        key.piDecAttempt running fresh output.proof = some attempt ∧
        ∀ child, CE.Holds key.piRlcSemantics key.params
          (PiDEC.OutputWitnessConsumer.runningStatement key result child) (output.children child)

/-- The real event's witnesses are for the exact verifier-computed PiDEC
children, in their original order, with no assumed output correspondence. -/
theorem realSuccess_implies_exact_children
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : RealOutput relation) (success : RealSuccess relation ajtai contextDigest running fresh (some output)) :
    let key := ProductionKey.key relation ajtai
    ∃ result attempt,
      PaperNonInteractive.verify key running fresh output.proof = some result ∧
      key.piDecAttempt running fresh output.proof = some attempt ∧
      ∀ child, CE.Holds key.piRlcSemantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
        (output.children (Fin.cast key.outputCount_eq child)) := by
  dsimp only
  rcases success with ⟨_, result, attempt, accepted, attemptEq, valid⟩
  refine ⟨result, attempt, accepted, attemptEq, ?_⟩
  intro child
  rw [← PiDEC.OutputWitnessConsumer.runningStatement_eq_child
    (ProductionKey.key relation ajtai) running fresh output.proof result attempt attemptEq accepted child]
  exact valid (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child)

variable {Context : Type*}
  (running : Context → Lifecycle.Running
    (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh
    (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Success mass under the supplied classical adversary output law.
There is no caller-supplied scalar standing in for verifier success. -/
noncomputable def realSuccessProbability
    (law : PMF (Context × Option (RealOutput relation))) : ℝ :=
  ∑' outcome, if RealSuccess relation ajtai contextDigest (running outcome.1) (fresh outcome.1) outcome.2
    then (law outcome).toReal else 0

/-- The context marginal of an adversary output law: the law of the running
and fresh public inputs. -/
noncomputable def contextLaw (law : PMF (Context × Option (RealOutput relation))) : PMF Context :=
  law.map Prod.fst

end NightstreamFPrime.Export.Stage1.NifsRealSuccess
