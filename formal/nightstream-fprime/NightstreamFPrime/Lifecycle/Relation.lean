import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.ProductionKey

/-!
Owns the concrete Stage 1 F′ relation: HyperNova Construction 2 instantiated
with the SuperNeo production NIFS verifier, Poseidon2 state hashing, the
`encHash` instance encoding, and the paper default running vector. Its inputs
are protocol data, never predicates: the F′ logical relation and its Ajtai key
(verifier-owned), the verifier-key digest (verifier-owned), and the
application step function `F` (the program being proved).
-/

namespace NightstreamFPrime.Lifecycle

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.ProductionKey

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The Stage 1 NIFS verifier as HyperNova's `NIFS.V`: the key digest selects
nothing (`slotCount = 1`); the concrete key is verifier-owned setup data. -/
noncomputable def nifsVerifier (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Verifier KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) where
  verify := fun _ running fresh proof =>
    Nifs.PaperNonInteractive.verify (key relation ajtai) running fresh proof

/-- Construction-2 `Setup`: verifier keys, `NIFS.V`, and `u_⊥`. -/
noncomputable def setup (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) :
    Setup KeyDigest (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount where
  verifierKeys := fun _ => vk
  nifs := nifsVerifier relation ajtai
  defaultRunning := defaultRunning

/-- Construction-2 `Machine` for uniform IVC: the control function selects the
one augmented function, the step is the application `F`, the fresh public
input is the fresh instance's public block, and the hash is `stateHash`. -/
def machine (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (F : AppState → AppWitness → AppState) :
    Machine KeyDigest Digest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount where
  control := fun _ _ => functionIndex
  step := fun _ z w => F z w
  freshPublic := fun fresh => fresh.publicInputs ⟨0, by decide⟩
  encodeInstance := encHash (publicFits := publicFits)
  hash := stateHash (publicFits := publicFits)

/-- The complete Stage 1 augmented-function relation `F′`: one step from
`input` to `output` is valid exactly when HyperNova's fixed transition holds
for the production setup and machine. -/
noncomputable def StepHolds (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount) : Prop :=
  FixedAugmentedTransition (setup relation ajtai vk) (machine publicFits F)
    functionIndex input output

/-- The CE statement of running slot `i` at the fresh norm stage `CE(b)`. -/
def runningStatement (relation : LogicalRelation logicalWidth publicFits)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (i : Fin productionShape.runningCount) :
    CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment where
  constraintSystem := PaperAlgebra.relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  commitment := running.commitments i
  publicInput := running.publicInputs i
  point := running.point
  evaluations := #[running.evaluations i]
  stage := .fresh

/-- The CCS statement of the fresh instance at `CCS(b)`. -/
def freshStatement (relation : LogicalRelation logicalWidth publicFits)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    CCS.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Commitment where
  constraintSystem := PaperAlgebra.relationSource
    (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) relation.cubeFits)
    relation.system
  commitment := fresh.commitments ⟨0, by decide⟩
  publicInput := fresh.publicInputs ⟨0, by decide⟩
  stage := .fresh

/-- Terminal decider (HyperNova `V`, step 5, and SuperNeo relation membership):
every running CE claim and the fresh CCS claim open with a norm-bounded
witness of the F′ structure. -/
def TerminalHolds (relation : LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (runningWitness : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  (∀ i, CE.Holds (semantics ajtai) productionGlobalParams
    (runningStatement relation running i) (runningWitness i)) ∧
  CCS.Holds (semantics ajtai) productionGlobalParams
    (freshStatement relation fresh) freshWitness

end

end NightstreamFPrime.Lifecycle
