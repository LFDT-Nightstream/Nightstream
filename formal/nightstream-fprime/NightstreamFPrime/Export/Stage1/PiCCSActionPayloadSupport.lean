import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadBlock
import NightstreamFPrime.Layout.Stage1.PiCCSInputSupport
import NightstreamFPrime.Layout.Stage1.PiCCSTranscriptSupport

/-!
Owns affine recognition and exact source support for the PiCCS action payload
expressions, including their verifier-derived squeeze expectations.
PiCCSPayloadWiring owns compilation through the parent's source map.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSActionPayloadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSActionPayloadBlock

private abbrev Supported (expression : Expr) : Prop :=
  expression.VarsSatisfy PiCCSOrdinarySourceSupport.Source

private def ListSupported (expressions : List Expr) : Prop :=
  ∀ expression ∈ expressions, Supported expression

private def ActionSupported : Formal.Action → Prop
  | .absorb input => ListSupported input
  | .squeezeK expected => Supported expected.c0 ∧ Supported expected.c1

private theorem constants_supported (words : List F) :
    ListSupported (words.map Expr.const) := by
  intro expression member
  rcases List.mem_map.mp member with ⟨word, _, rfl⟩
  trivial

private theorem block_supported (words : List Expr) (supported : ListSupported words) :
    ListSupported (StatementAbsorption.blockExpr words) := by
  intro expression member
  rcases List.mem_cons.mp member with rfl | member
  · trivial
  · exact supported expression member

private theorem serializeK_supported (value : KExpr)
    (supported : Supported value.c0 ∧ Supported value.c1) :
    ListSupported (StatementAbsorption.serializeKExpr value) := by
  intro expression member
  simp only [StatementAbsorption.serializeKExpr, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact supported.1
  · exact supported.2

private theorem commitment_supported
    (commitment : Fin productionProfile.commitmentWidth → Fin ringDegree → Expr)
    (supported : ∀ row coefficient, Supported (commitment row coefficient)) :
    ListSupported (StatementAbsorption.serializeCommitmentExpr commitment) := by
  intro expression member
  rw [StatementAbsorption.serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _, member⟩
  rcases List.mem_map.mp member with ⟨coefficient, _, rfl⟩
  exact supported row coefficient

private theorem publicInput_supported
    (input : Fin (FullShape Data.logicalWidth Data.publicFits).publicWidth → Expr)
    (supported : ∀ column, Supported (input column)) :
    ListSupported (StatementAbsorption.serializePublicInputExpr input) := by
  intro expression member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  exact supported column

private theorem hashWord_supported
    (input : Fin (FullShape Data.logicalWidth Data.publicFits).publicWidth → Expr)
    (supported : ∀ column, Supported (input column)) (word : Fin 4) :
    Supported (StatementAbsorption.decodeHashWordExpr input word) := by
  have loop (indices : List Nat) (initial : Expr) (initialSupported : Supported initial) :
      Supported (indices.foldl (fun value bit =>
        value + Expr.const (Poseidon2.ofNat (2 ^ bit)) *
          input (digestBitIndexNat (logicalWidth := Data.logicalWidth) word bit)) initial) := by
    induction indices generalizing initial with
    | nil => exact initialSupported
    | cons bit indices inductionHypothesis =>
        rw [List.foldl_cons]
        apply inductionHypothesis
        exact ⟨initialSupported, trivial, supported _⟩
  exact loop (List.range 64) 0 trivial

private theorem statementActions_supported : ∀ action ∈ statementActions,
    ActionSupported action := by
  have caller := (PiCCSOrdinarySourceSupport.externalInputsSupported
    Data.logicalWidth Data.publicFits).mono PiCCSOrdinarySourceSupport.external_source
  let interface := PiCCSInvocations.statementInterface Data.logicalWidth Data.publicFits
  have freshCommitment : ∀ source row coefficient,
      Supported ((interface.fresh PiCCSInvocations.statementWitnessStart).commitment
        source row coefficient) := by
    intro source row coefficient
    exact caller.freshCommitment source row coefficient
  have freshPublic : ∀ source column,
      Supported ((interface.fresh PiCCSInvocations.statementWitnessStart).publicInput
        source column) := by
    intro source column
    exact caller.freshPublicInput source column
  have blocks : ∀ block ∈ StatementAbsorption.publicInputBlocks interface
      PiCCSInvocations.statementWitnessStart, ListSupported block := by
    intro block member
    simp only [StatementAbsorption.publicInputBlocks, List.mem_append,
      List.mem_singleton, List.mem_flatMap] at member
    rcases member with rfl | ⟨source, _, member⟩
    · intro expression member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      exact hashWord_supported _ (freshPublic ⟨0, by decide⟩) lane
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl
      · exact commitment_supported _ (freshCommitment source)
      · exact publicInput_supported _ (freshPublic source)
  intro action member
  change action ∈ StatementAbsorption.publicInputActions interface
    PiCCSInvocations.statementWitnessStart at member
  simp only [StatementAbsorption.publicInputActions, List.mem_append,
    List.mem_singleton, List.mem_map] at member
  rcases member with rfl | ⟨block, member, rfl⟩
  · exact constants_supported _
  · exact block_supported block (blocks block member)

private theorem labelledActions_supported
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr)
    (supported : ∀ sample ∈ samples, Supported sample.c0 ∧ Supported sample.c1) :
    ∀ action ∈ ChallengeDerivation.labelledActions labels samples,
      ActionSupported action := by
  induction labels generalizing samples with
  | nil => intro action member; cases member
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => intro action member; cases member
      | cons sample samples =>
          intro action member
          simp only [ChallengeDerivation.labelledActions, List.mem_append,
            ChallengeDerivation.labelActions, List.mem_cons, List.not_mem_nil,
            or_false] at member
          rcases member with (rfl | rfl) | member
          · exact constants_supported _
          · exact supported sample (by simp)
          · exact inductionHypothesis samples
              (fun value member => supported value (List.mem_cons_of_mem _ member))
              action member

private theorem challengeActions_supported : ∀ action ∈ challengeActions,
    ActionSupported action := by
  unfold challengeActions
  rw [ChallengeDerivation.actions_eq_labelled]
  apply labelledActions_supported
  intro sample member
  rw [← ChallengeDerivation.layoutWiring_samples_eq] at member
  have supported := PiCCSOrdinarySourceSupport.challengeWiring_supported
    Data.logicalWidth Data.publicFits
  exact supported sample member

private theorem roundActions_supported : ∀ action ∈ roundActions,
    ActionSupported action := by
  have caller := (PiCCSOrdinarySourceSupport.externalInputsSupported
    Data.logicalWidth Data.publicFits).mono PiCCSOrdinarySourceSupport.external_source
  have transcript := PiCCSOrdinarySourceSupport.transcriptValuesSupported
    Data.logicalWidth Data.publicFits
  let interface := PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits
  have coefficients : ∀ roundIndex,
      ListSupported (RoundTranscript.serializeRoundExpr
        (interface.round PiCCSInvocations.roundWitnessStart roundIndex)) := by
    intro roundIndex expression member
    change expression ∈ (List.ofFn
      (interface.round PiCCSInvocations.roundWitnessStart roundIndex).coefficient).flatMap
        StatementAbsorption.serializeKExpr at member
    rcases List.mem_flatMap.mp member with ⟨value, valueMember, member⟩
    rcases List.mem_ofFn.mp valueMember with ⟨coefficient, rfl⟩
    exact serializeK_supported _ (caller.roundCoefficient roundIndex coefficient)
      expression member
  intro action member
  change action ∈ RoundTranscript.actions interface PiCCSInvocations.roundWitnessStart at member
  rw [RoundTranscript.actions, List.mem_flatMap] at member
  rcases member with ⟨roundIndex, _, member⟩
  simp only [RoundTranscript.roundActions, RoundTranscript.roundActionsWithExpected,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · apply block_supported
    intro expression member
    rcases List.mem_cons.mp member with rfl | member
    · trivial
    · exact coefficients roundIndex expression member
  · exact constants_supported _
  · exact transcript.roundPoint roundIndex

private theorem outputActions_supported : ∀ action ∈ outputActions,
    ActionSupported action := by
  have caller := (PiCCSOrdinarySourceSupport.externalInputsSupported
    Data.logicalWidth Data.publicFits).mono PiCCSOrdinarySourceSupport.external_source
  intro action member
  simp only [outputActions, PiCCSInvocations.outputActions, OutputBinding.actions,
    List.mem_singleton] at member
  subst action
  apply block_supported
  intro expression member
  rw [OutputBinding.outputWords, List.mem_flatMap] at member
  rcases member with ⟨source, _, member⟩
  rw [OutputBinding.sourceWords, List.mem_append] at member
  rcases member with pad | matrix
  · rw [OutputBinding.padWords, List.mem_flatMap] at pad
    rcases pad with ⟨coefficient, _, member⟩
    exact serializeK_supported _ (caller.outputEval_K source coefficient) expression member
  · rw [OutputBinding.matrixWords, List.mem_flatMap] at matrix
    rcases matrix with ⟨matrix, _, member⟩
    rcases List.mem_flatMap.mp member with ⟨coefficient, _, member⟩
    exact serializeK_supported _ (caller.outputEval_A source matrix coefficient)
      expression member

private theorem selectedBlock_supported (actions : List Formal.Action)
    (supported : ∀ action ∈ actions, ActionSupported action)
    (kind : PoseidonActionSchedule.Kind)
    (member : kind ∈ PoseidonActionSchedule.kinds actions) :
    ListSupported (selectedBlockForKind kind) := by
  rw [PoseidonActionSchedule.kinds, List.mem_flatMap] at member
  rcases member with ⟨action, actionMember, kindMember⟩
  have actionProperty := supported action actionMember
  cases action with
  | absorb input =>
      simp only [PoseidonActionSchedule.actionKinds, List.mem_map] at kindMember
      rcases kindMember with ⟨block, blockMember, rfl⟩
      unfold Hash.inputChunks at blockMember
      rcases List.mem_map.mp blockMember with ⟨index, _, rfl⟩
      intro expression expressionMember
      exact actionProperty expression
        (List.mem_of_mem_drop (List.mem_of_mem_take expressionMember))
  | squeezeK expected =>
      simp only [PoseidonActionSchedule.actionKinds, List.mem_cons,
        List.not_mem_nil, or_false] at kindMember
      rcases kindMember with rfl | rfl
      · exact serializeK_supported expected actionProperty
      · intro expression member; cases member

/-- All actual absorb words and squeeze expectations use the exact declared
PiCCS source families. No intermediate unowned permutation recipe is read. -/
theorem payloadExpression_supported (index : Fin payloadCount) :
    (payloadExpression index).VarsSatisfy PiCCSOrdinarySourceSupport.Source := by
  let decoded : Fin invocationCount × Fin Spec.Poseidon2.rate := Fin.decodeProd index
  have member : kindAt decoded.1 ∈ List.ofFn kindAt :=
    List.mem_ofFn.mpr ⟨decoded.1, rfl⟩
  rw [kindAt_materializes] at member
  simp only [List.mem_append] at member
  have selected : ListSupported (selectedBlock decoded.1) := by
    rcases member with statement | challenge | round | output
    · exact selectedBlock_supported statementActions statementActions_supported _ statement
    · exact selectedBlock_supported challengeActions challengeActions_supported _ challenge
    · exact selectedBlock_supported roundActions roundActions_supported _ round
    · exact selectedBlock_supported outputActions outputActions_supported _ output
  change Supported ((selectedBlock decoded.1).getD decoded.2.val 0)
  by_cases bound : decoded.2.val < (selectedBlock decoded.1).length
  · rw [List.getD_eq_get _ _ ⟨decoded.2.val, bound⟩]
    exact selected _ (List.get_mem _ _)
  · rw [List.getD_eq_default _ _ (Nat.le_of_not_gt bound)]
    trivial

private theorem selectedBlock_affine (actions : List Formal.Action)
    (affine : ActionsAffine actions) (kind : PoseidonActionSchedule.Kind)
    (member : kind ∈ PoseidonActionSchedule.kinds actions) :
    ListAffine (selectedBlockForKind kind) := by
  rw [PoseidonActionSchedule.kinds, List.mem_flatMap] at member
  rcases member with ⟨action, actionMember, kindMember⟩
  have actionProperty := affine action actionMember
  cases action with
  | absorb input =>
      simp only [PoseidonActionSchedule.actionKinds, List.mem_map] at kindMember
      rcases kindMember with ⟨block, blockMember, rfl⟩
      unfold Hash.inputChunks at blockMember
      simp only [List.mem_map] at blockMember
      rcases blockMember with ⟨index, _, rfl⟩
      intro expression expressionMember
      exact actionProperty expression
        (List.mem_of_mem_drop (List.mem_of_mem_take expressionMember))
  | squeezeK expected =>
      simp only [PoseidonActionSchedule.actionKinds, List.mem_cons,
        List.not_mem_nil, or_false] at kindMember
      rcases kindMember with rfl | rfl
      · intro expression expressionMember
        simp only [selectedBlockForKind, List.mem_cons,
          List.not_mem_nil, or_false] at expressionMember
        rcases expressionMember with rfl | rfl
        · exact actionProperty.1
        · exact actionProperty.2
      · intro expression expressionMember
        cases expressionMember

private theorem kindAt_affine
    (current : Fin invocationCount) :
    ListAffine (selectedBlock current) := by
  -- These four syntactic input-shape fields do not inspect matrix values.
  have shapes := PiCCSInvocations.inputShapes Data.logicalWidth Data.publicFits
    ⟨fun _ _ _ => 0, by decide⟩
  have member : kindAt current ∈ List.ofFn kindAt :=
    (List.mem_ofFn).2 ⟨current, rfl⟩
  rw [kindAt_materializes] at member
  simp only [List.mem_append] at member
  unfold selectedBlock
  rcases member with statement | challenge | round | output
  · exact selectedBlock_affine statementActions
      (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.actions_affine
        (PiCCSInvocations.statementInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.statementWitnessStart
        (shapes.statementAbsorption PiCCSInvocations.statementWitnessStart)) _
      statement
  · exact selectedBlock_affine challengeActions
      (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.actions_affine
        (PiCCSInvocations.challengeInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.challengeWitnessStart
        (shapes.challengeDerivation PiCCSInvocations.challengeWitnessStart)) _
      challenge
  · exact selectedBlock_affine roundActions
      (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.actions_affine
        (PiCCSInvocations.roundInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.roundWitnessStart
        (shapes.roundTranscript PiCCSInvocations.roundWitnessStart)) _ round
  · exact selectedBlock_affine outputActions
      (NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.actions_affine
        (PiCCSInvocations.outputInterface Data.logicalWidth Data.publicFits)
        PiCCSInvocations.outputWitnessStart
        (shapes.outputBinding PiCCSInvocations.outputWitnessStart)) _ output

/-- Every emitted payload expression is recognized by the existing affine
lowerer, including the actual squeeze expectations and zero padding. -/
theorem payloadExpression_affine
    (index : Fin payloadCount) : R1CS.IsAffine (payloadExpression index) := by
  let decoded : Fin invocationCount × Fin Spec.Poseidon2.rate := Fin.decodeProd index
  change R1CS.IsAffine ((selectedBlock decoded.1).getD decoded.2.val 0)
  by_cases bound : decoded.2.val < (selectedBlock decoded.1).length
  · rw [List.getD_eq_get _ _ ⟨decoded.2.val, bound⟩]
    exact kindAt_affine decoded.1 _ (List.get_mem _ _)
  · rw [List.getD_eq_default]
    · exact R1CS.isAffine_const 0
    · exact Nat.le_of_not_gt bound

end NightstreamFPrime.Export.Stage1.PiCCSActionPayloadSupport
