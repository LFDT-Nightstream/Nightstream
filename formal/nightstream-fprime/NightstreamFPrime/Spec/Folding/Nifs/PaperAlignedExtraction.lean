import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction

/-!
Deterministic specialization of an actual weak-extractor return to the
noninteractive NIFS alignment predicate. The observed probe and sampled base
vector must equal the key's replay. This is an equality boundary, not a claim
that Poseidon2 sampling has the interactive uniform distribution.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperAlignedExtraction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open PaperNonInteractive PaperStrongInterface PaperSecurityComposition
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateOracle PiRLC.CoordinateForkLaw
open PiRLC.CoordinateCheckedCalls PiRLC.CoordinateTerminalLaw
open PiRLC.CoordinateTerminalProgram

universe uExtension uCommitment uPublicInput uScalar uState

variable {Extension : Type uExtension} {Commitment : Type uCommitment}
  {PublicInput : Type uPublicInput} {Scalar : Type uScalar} {State : Type uState}
  {shape : Shape} {columns blockCount width : Nat}
  (key : Key Extension Commitment PublicInput Scalar State shape columns blockCount width)
  (running : Running Extension Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)
  (proof : Proof Extension Commitment shape width)
  (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)

variable [DecidableEq Scalar] [Fintype (Challenge key.piRlcAlgebra)]
  [Nonempty (Challenge key.piRlcAlgebra)]
  [Fintype (PaperLinearAlgebra.Assignment F columns)]

/-- The actual terminal return supplies the aligned fork and preserves every
observed base/fork response. No caller-chosen fork or intermediate opening is
used. The public parent is the key's result for that same observed sampler. -/
theorem positive_return_implies_alignedFork
    (probe : Probe Extension shape)
    (probeEqual : probe = key.piCcsProbe running fresh proof)
    (oracle : Oracle (Fin key.arity.total) (Challenge key.piRlcAlgebra)
      (PaperLinearAlgebra.Assignment F columns))
    (checker : Response (PaperLinearAlgebra.Assignment F columns) Scalar
      key.params key.arity → CheckResult)
    (checkSpec : ∀ response, (checker response).accepted = true ↔
      response.Success key.piRlcSemantics key.params key.piRlcAlgebra
        (piRlcBatchForProbe key running fresh probe))
    (program : Primitives Scalar (PaperLinearAlgebra.Assignment F columns))
    (correct : Correct laws.ring laws.assignmentModule program)
    (vector : Fin key.arity.total → Challenge key.piRlcAlgebra)
    (initial : Option (PaperLinearAlgebra.Assignment F columns))
    (outputs : Fin key.arity.total → Outcome
      (Challenge := Challenge key.piRlcAlgebra)
      (Assignment := PaperLinearAlgebra.Assignment F columns))
    (values : List (PaperLinearAlgebra.Assignment F columns))
    (positive : 0 < endpointMass oracle
      (oracleCheck key.piRlcAlgebra (fun response => (checker response).accepted))
      vector initial outputs)
    (returned : (finish program vector initial outputs).value = some values)
    (sampled : key.piRlcChallenges running fresh proof =
      some (scalarVector key.piRlcAlgebra vector)) :
    ∃ aligned : AlignedFork key running fresh proof,
      aligned.challenges = scalarVector key.piRlcAlgebra vector ∧
      some aligned.fork.base.assignment = initial ∧
      (∀ coordinate,
        (aligned.fork.forks coordinate).challenges =
          Function.update (scalarVector key.piRlcAlgebra vector) coordinate (outputs coordinate).1.val ∧
        some (aligned.fork.forks coordinate).assignment = (outputs coordinate).2) ∧
      values = List.ofFn (extractedAssignment laws strongSet aligned.fork) ∧
      key.parent running fresh proof =
        some (aligned.fork.base.output key.piRlcAlgebra (piRlcBatch key running fresh proof)) := by
  subst probe
  have checkSpecNifs : ∀ response, (checker response).accepted = true ↔
      response.Success key.piRlcSemantics key.params key.piRlcAlgebra
        (piRlcBatch key running fresh proof) := by
    simpa only [piRlcBatchForProbe_eq_piRlcBatch] using checkSpec
  obtain ⟨fork, baseVector, baseAssignment, responses, valuesEq, _openings⟩ :=
    PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
      (piRlcBatch key running fresh proof) laws strongSet oracle checker checkSpecNifs
      program correct vector initial outputs values positive returned
  let aligned : AlignedFork key running fresh proof := {
    fork := fork
    challenges := scalarVector key.piRlcAlgebra vector
    sampled := sampled
    baseChallenges := baseVector }
  refine ⟨aligned, rfl, baseAssignment, responses, valuesEq, ?_⟩
  have outputEqual := baseOutput_eq_parentForChallenges key running fresh proof fork
    (scalarVector key.piRlcAlgebra vector) baseVector
  change Option.map (key.parentForChallenges running fresh proof)
    (key.piRlcChallenges running fresh proof) =
      some (fork.base.output key.piRlcAlgebra (piRlcBatch key running fresh proof))
  rw [sampled, Option.map_some]
  exact congrArg some outputEqual.symm

end NightstreamFPrime.Spec.Folding.Nifs.PaperAlignedExtraction
