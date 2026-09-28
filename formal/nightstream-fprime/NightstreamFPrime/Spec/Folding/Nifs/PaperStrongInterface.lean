import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition

/-!
The interactive PiCCS output consumed by the NIFS weak suffix. Public values
come from the same statement and probe; source assignments enter only through
the suffix's returned witness. The noninteractive replay is an exact separate
specialization and does not supply the interactive challenge distribution.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open PaperNonInteractive PiRLC.PaperForkExtraction

universe uExtension uCommitment uPublicInput uScalar uState

variable {Extension : Type uExtension} {Commitment : Type uCommitment}
  {PublicInput : Type uPublicInput} {Scalar : Type uScalar} {State : Type uState}
  {shape : Shape} {columns blockCount width : Nat}
  (key : Key Extension Commitment PublicInput Scalar State shape columns blockCount width)
  (running : Running Extension Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)

/-- The exact interactive output product in the weak suffix's K+k order. -/
def piRlcBatchForProbe (probe : Probe Extension shape) :
    InputBatch (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables) (EvaluationFamily Extension shape)
      Commitment key.params key.arity where
  system := key.relationSource
  point := probe.coins.roundPoint
  inputs := fun coordinate => (key.statement running fresh).publicOutput probe
    (Fin.cast key.total_eq_sourceCount coordinate)
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := 1
  evaluationsSize := fun _ => rfl

/-- Replaying an actual NIFS proof selects exactly the existing public batch.
This equality makes no assertion about the distribution of replayed coins. -/
theorem piRlcBatchForProbe_eq_piRlcBatch (proof : Proof Extension Commitment shape width) :
    piRlcBatchForProbe key running fresh (key.piCcsProbe running fresh proof) =
      PaperSecurityComposition.piRlcBatch key running fresh proof := rfl

/-- Two different transcripts from one source statement have the same Phi.
This pointwise equality also covers aborted-run conditional applications. -/
theorem piRlcBatchForProbe_same_phi (left right : Probe Extension shape) :
    PiRLC.phi (piRlcBatchForProbe key running fresh left).inputs =
      PiRLC.phi (piRlcBatchForProbe key running fresh right).inputs := rfl

/-- Reindex the weak extractor's returned vector into PiCCS source order. -/
def outputWitnessOfAssignments
    (values : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns) :
    OutputWitness shape columns where
  assignments := fun source => values (Fin.cast key.total_eq_sourceCount.symm source)

theorem outputWitnessOfAssignments_at
    (values : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns)
    (coordinate : Fin key.arity.total) :
    (outputWitnessOfAssignments key values).assignments
      (Fin.cast key.total_eq_sourceCount coordinate) = values coordinate := by
  apply congrArg values
  exact Fin.ext rfl

/-- The suffix's actual corrected ambient openings give the literal PiCCS
relaxed output relation. No intermediate opening is invented by this adapter. -/
theorem outputWitnessOfAssignments_ambient
    (probe : Probe Extension shape)
    (values : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns)
    (valid : ∀ coordinate, PiRLC.PaperCorrections.CorrectedAmbientHolds
      key.piRlcSemantics key.params
      ((piRlcBatchForProbe key running fresh probe).inputs coordinate)
      (values coordinate)) :
    AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) probe (outputWitnessOfAssignments key values) := by
  intro source
  rw [key.ambientAgreement
    ((key.statement running fresh).publicOutput probe source)
    ((outputWitnessOfAssignments key values).assignments source) rfl]
  simpa [piRlcBatchForProbe, outputWitnessOfAssignments] using
    valid (Fin.cast key.total_eq_sourceCount.symm source)

end NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
