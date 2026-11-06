import NightstreamFPrime.Spec.Folding.Refresh.Verifier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership

/-! The running sources of the refresh fold's Π_CCS are the children of its input Π_DEC
attempt, and CE claims on the key's relation hold in the paper semantics exactly when they
hold in the Π_RLC semantics. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Key Fresh Running)

universe uCommitment uPublicInput uScalar uState uValue

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

theorem addCases_runningSource {Value : Type uValue}
    (left : Fin shape.freshCount -> Value) (right : Fin shape.runningCount -> Value)
    (running : Fin shape.runningCount) :
    (Fin.addCases left right (runningSourceIndex running) : Value) = right running := by
  have natAdd : runningSourceIndex running = Fin.natAdd shape.freshCount running := Fin.ext rfl
  rw [natAdd, Fin.addCases_right]

theorem addCases_freshSource {Value : Type uValue}
    (left : Fin shape.freshCount -> Value) (right : Fin shape.runningCount -> Value)
    (fresh : Fin shape.freshCount) :
    (Fin.addCases left right (freshSourceIndex fresh) : Value) = left fresh := by
  have castAdd : freshSourceIndex fresh = Fin.castAdd shape.runningCount fresh := Fin.ext rfl
  rw [castAdd, Fin.addCases_left]

theorem runningInstance_eq_child
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (split : Split Commitment shape)
    (system : accumulator.constraintSystem = key.relationSource)
    (running : Fin shape.runningCount) :
    SourceMembership.runningInstance (key.statement (children key accumulator split) fresh)
        running =
      PiDEC.PaperVerifier.children key.piDecPublicInputSplit
        (inputAttempt key accumulator split) (Fin.cast key.runningCount_eq_outputCount running) := by
  have castBack : Fin.cast key.outputCount_eq (Fin.cast key.runningCount_eq_outputCount running) =
      running := Fin.ext rfl
  unfold SourceMembership.runningInstance PiDEC.PaperVerifier.children
  simp only [inputAttempt, castBack, system]
  congr 1
  · exact addCases_runningSource _ _ running
  · exact addCases_runningSource _ _ running

theorem ceHolds_iff
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (statement : Accumulator Commitment PublicInput shape columns blockCount)
    (system : statement.constraintSystem = key.relationSource)
    (assignment : PaperLinearAlgebra.Assignment F columns) :
    CE.Holds (paperRelationSemantics key.baseOps key.extensionOps key.lift key.openingMaps)
        key.params statement assignment ↔
      CE.Holds key.piRlcSemantics key.params statement assignment := by
  unfold CE.Holds
  rw [system]
  constructor
  · rintro ⟨opening, _, evaluations⟩
    exact ⟨(key.openingAgreement _ _ _ _).mp opening,
      (key.evaluationAgreement assignment statement.point).1,
      (key.evaluationAgreement assignment statement.point).2.trans evaluations⟩
  · rintro ⟨opening, _, evaluations⟩
    exact ⟨(key.openingAgreement _ _ _ _).mpr opening, trivial,
      (key.evaluationAgreement assignment statement.point).2.symm.trans evaluations⟩

end NightstreamFPrime.Spec.Folding.Refresh
