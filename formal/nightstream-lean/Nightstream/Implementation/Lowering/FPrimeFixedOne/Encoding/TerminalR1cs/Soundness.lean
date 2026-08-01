import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Program

/-!
Contract: semantic soundness of the selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: the refinement from satisfaction of the decoded proof-free manifest,
under verifier-owned public statement columns, to all fourteen running CE
relations and the independent fresh CCS relation.

Does not own: witness construction, benchmark selection, Spartan or WHIR
soundness, Ajtai binding security, or Rust manifest equality.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Soundness

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

/-- Verifier interpretation of every public column in the terminal manifest.
These equalities are not proof claims: the verifier constructs the public
input vector from the authoritative CE and CCS statements. -/
structure AuthoritativeColumns
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F) : Prop where
  one : assignment oneColumn = 1
  runningFreshStage :
    ∀ child,
      (statements child).stage.bound productionGlobalParams = 2
  runningCommitment :
    ∀ child,
      (fun verifierRow output =>
        assignment
          ((Layout.runningFrame key child).commitment verifierRow output)) =
        (statements child).commitment
  runningPublic :
    ∀ child,
      (fun coordinate =>
        assignment
          ((Layout.runningFrame key child).publicColumn coordinate)) =
        (statements child).publicInput
  runningEvaluation :
    ∀ child,
      Running.EvaluationColumnsMatch
        (Layout.runningFrame key child) (statements child) assignment
  freshCommitment :
    (fun verifierRow output =>
      assignment ((Layout.freshFrame key).commitment verifierRow output)) =
      freshPayload.commitment
  freshPublic :
    (fun coordinate =>
      assignment ((Layout.freshFrame key).publicColumn coordinate)) =
      freshPayload.publicInput

/-- Any satisfying assignment of the exact decoded terminal manifest
establishes the paper terminal relations selected by the verifier. -/
theorem decodedProgram_sound
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (authority :
      AuthoritativeColumns key statements freshPayload assignment)
    (satisfied :
      Satisfies (Layout.program valid key statements).decode.rows assignment) :
    (∀ child,
      CE.Holds
        (Phi81Relation.relationSemantics
          (Commitment.commit (Layout.runningFrame key child).key))
        productionGlobalParams (statements child)
        (fun coordinate =>
          assignment
            ((Layout.runningFrame key child).witness coordinate))) ∧
      CCS.Holds
        (Phi81Relation.relationSemantics
          (Commitment.commit (Layout.freshFrame key).key))
        productionGlobalParams
        (freshPayload.materialize
          (NativeCcsPhi81.relation program valid domain
            publicRingColumns publicFits))
        (fun coordinate =>
          assignment ((Layout.freshFrame key).witness coordinate)) := by
  apply Terminal.rows_sound noZeroDivisors valid (Layout.frame key)
      statements freshPayload assignment
  · exact fun _ => authority.one
  · exact authority.runningFreshStage
  · exact authority.runningCommitment
  · exact authority.runningPublic
  · exact authority.runningEvaluation
  · exact authority.one
  · exact authority.freshCommitment
  · exact authority.freshPublic
  · exact
      (Program.decoded_satisfies_iff valid key statements assignment).mp
        satisfied

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Soundness
