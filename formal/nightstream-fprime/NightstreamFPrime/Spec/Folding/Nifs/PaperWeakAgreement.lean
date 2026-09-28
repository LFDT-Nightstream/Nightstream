import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateExtraction

/-!
SuperNeo B.1 compares two weak extractions after different PiCCS transcripts
from one original input. Each witness below comes from a positive-mass return
of the actual coordinate terminal program. A complete fork is derived proof
evidence, never an input used to choose the returned assignments.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAgreement

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open PaperNonInteractive PaperStrongInterface
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
  (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)

variable [DecidableEq Scalar] [Fintype (Challenge key.piRlcAlgebra)]
  [Nonempty (Challenge key.piRlcAlgebra)]
  [Fintype (PaperLinearAlgebra.Assignment F columns)]

include strongSet in
/-- The observed returned list opens the exact public batch of this probe. -/
theorem positive_return_opens_probe
    (probe : Probe Extension shape)
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
    (returned : (finish program vector initial outputs).value = some values) :
    ∃ assignments : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns,
      values = List.ofFn assignments ∧
      PiRLC.AmbientOpenings key.piRlcSemantics key.params
        (piRlcBatchForProbe key running fresh probe).inputs assignments ∧
      AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
        (key.statement running fresh) probe (outputWitnessOfAssignments key assignments) := by
  obtain ⟨fork, _baseVector, _baseValue, _responses, valuesEq, valid⟩ :=
    PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
      (piRlcBatchForProbe key running fresh probe) laws strongSet oracle checker
      checkSpec program correct vector initial outputs values positive returned
  refine ⟨extractedAssignment laws strongSet fork, valuesEq, ?_, ?_⟩
  · intro coordinate
    apply (PiRLC.PaperCorrections.correctedAmbientHolds_iff_ceHolds_of_ambient
      key.piRlcSemantics key.params
      (PiRLC.ambientInput
        ((piRlcBatchForProbe key running fresh probe).inputs coordinate))
      (extractedAssignment laws strongSet fork coordinate) rfl).mp
    simpa [PiRLC.PaperCorrections.CorrectedAmbientHolds, PiRLC.ambientInput]
      using valid coordinate
  · exact outputWitnessOfAssignments_ambient key running fresh probe
      (extractedAssignment laws strongSet fork) valid

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAgreement
