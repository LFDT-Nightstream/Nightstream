import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakCompleteness

/-!
Honest completeness of the selected interactive C/R/D sequence. The same
source assignments feed the causal PiCCS prover and the honest weak suffix.
Every final child opening is derived from those source assignments.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.InteractiveCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.CoordinateForkLaw

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth))

/-- One causal honest C prover succeeds for every C coin stream. Its exact
public output then has an accepted honest R/D continuation for every allowed
R vector, with all child openings derived and the same source order retained. -/
theorem exists_honest_execution
    (valid : SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement running fresh) witness) :
    ∃ prover : CausalExecution.Prover productionShape (Phi81CarrierLayout.carrierWidth logicalWidth) 9,
      ∀ (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
        (point : CubePoint K productionShape.cubeVariables),
        ∃ probe : Probe K productionShape,
          CausalExecution.run prover alpha gamma point = some (probe, witness) ∧
          probe.FixedWidthAccepted extensionOps K.embed
            ((ProductionKey.key relation ajtai).statement running fresh) 9 ∧
          ∀ vector : Fin (ProductionKey.key relation ajtai).arity.total →
              Challenge (ProductionKey.key relation ajtai).piRlcAlgebra,
            let key := ProductionKey.key relation ajtai
            let batch := PaperStrongInterface.piRlcBatchForProbe key running fresh probe
            let assignments := fun index => witness.assignments (Fin.cast key.total_eq_sourceCount index)
            let reply := PaperWeakCompleteness.honestReply key.piRlcAlgebra batch key.piDecAlgebra
              vector assignments
            let attempt := PaperWeakCompleteness.publicAttempt key.piRlcAlgebra batch vector reply.messages
            PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit
              key.piDecEvaluationArity attempt ∧
            ∀ child, CE.Holds key.piRlcSemantics key.params
              (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
              (reply.assignments child) := by
  obtain ⟨prover, honest⟩ := PaperStrongCompleteness.exists_honest_piCcs_prover
    (ProductionKey.key relation ajtai) running fresh witness valid
  refine ⟨prover, ?_⟩
  intro alpha gamma point
  obtain ⟨probe, returned, accepted, strict⟩ := honest alpha gamma point
  refine ⟨probe, returned, accepted, ?_⟩
  intro vector
  let key := ProductionKey.key relation ajtai
  let batch := PaperStrongInterface.piRlcBatchForProbe key running fresh probe
  let assignments := fun index => witness.assignments (Fin.cast key.total_eq_sourceCount index)
  have inputFresh : ∀ index, (batch.inputs index).stage = .fresh := fun _ => rfl
  have inputHolds : ∀ index, CE.Holds key.piRlcSemantics key.params
      (batch.inputs index) (assignments index) := by
    intro index
    exact strict (Fin.cast key.total_eq_sourceCount index)
  have complete := PaperWeakCompleteness.honest_complete key.piRlcAlgebra batch key.piDecAlgebra
    key.piDecPublicInputSplit key.piDecEvaluationArity vector assignments inputFresh inputHolds
  exact complete

end NightstreamFPrime.Lifecycle.Nifs.InteractiveCompleteness
