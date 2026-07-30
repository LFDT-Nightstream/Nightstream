import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4TranscriptMessages

/-!
Contract: prove that the complete physical ΠCCS operational input for the
42-times-6 WASM benchmark depends on the selected constraint polynomial, not
on the recursive relation matrix payload.

Assurance tier: model-level.

Owns: stability of the transcript input, authoritative endpoint columns, and
the joined `KSplitNcOperationalRows.Input`.

Does not own: operational rows, the ΠRLC sampler, other NIFS row families,
activation, the recursive fixed point, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4TranscriptMessages
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Selected transcript input before the endpoint authority is joined. -/
noncomputable def transcript
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalOccurrence.transcriptInput
    (application setup) (operational setup) (invokePlan setup).frame

/-- Selected verifier-owned endpoint columns. -/
noncomputable def authority
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalFrame.authorityColumns
    ((application setup).family _) (invokePlan setup).frame
    (operational setup).endpointViews

/-- Complete selected physical ΠCCS input before row emission. -/
noncomputable def input
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    KSplitNcOperationalRows.Input
      (KSplitNcStaticInput.layoutInput
        (operational setup).constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcript := transcript setup
  authority := authority setup

/-- Compact physical identity of one serialized transcript source. Semantic
value functions and dependent codec proofs are erased. -/
inductive SourceKey where
  | constant (value : Field)
  | running (index : Nat)
  | fresh (index : Nat)
  | proof (index : Nat)
  deriving DecidableEq

/-- Erase one proof-carrying transcript source to its physical kind and codec
index. -/
def sourceKey
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {runningCodec :
      Codec
        (ConcreteNifsParameters.SelectedRunning shape publicRingColumns
          publicFits verifierRows)}
    {freshCodec :
      Codec
        (ConcreteNifsParameters.SelectedFresh shape publicRingColumns
          publicFits verifierRows)}
    {proofCodec :
      Codec
        (ConcreteNifsParameters.SelectedProof shape Poseidon2Duplex.State
          publicRingColumns publicFits verifierRows)}
    (source :
      ConcreteNifsOperationalProfile.FieldSource
        runningCodec freshCodec proofCodec) : SourceKey :=
  match source with
  | .constant value => .constant value
  | .running _ view => .running view.index.val
  | .fresh _ view => .fresh view.index.val
  | .proof _ view => .proof view.index.val

private abbrev SourceFor
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalProfile.FieldSource
    (((application setup).family _).codecFor (.data .running))
    (((application setup).family _).codecFor (.data .fresh))
    (((application setup).family _).codecFor (.data .nifsProof))

/-- Equal source keys in equal physical frames produce equal symbolic
transcript expressions. -/
theorem sourceExpression_eq_of_key_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (leftSource : SourceFor (template.withSystem left))
    (rightSource : SourceFor (template.withSystem right))
    (keyEqual :
      sourceKey
          (publicFits :=
            ConcreteNifsPlain270Profile.publicFits dimensions)
          leftSource =
        sourceKey
          (publicFits :=
            ConcreteNifsPlain270Profile.publicFits dimensions)
          rightSource) :
    ConcreteNifsOperationalOccurrence.sourceExpression
        (application (template.withSystem left)).family
        (invokePlan (template.withSystem left)).frame leftSource =
      ConcreteNifsOperationalOccurrence.sourceExpression
        (application (template.withSystem right)).family
        (invokePlan (template.withSystem right)).frame rightSource := by
  cases leftSource with
  | constant leftValue =>
      cases rightSource with
      | constant rightValue =>
          simp only [sourceKey] at keyEqual
          injection keyEqual with valueEqual
          unfold ConcreteNifsOperationalOccurrence.sourceExpression
          rw [valueEqual]
      | running _ _ => cases keyEqual
      | fresh _ _ => cases keyEqual
      | proof _ _ => cases keyEqual
  | running _ leftView =>
      cases rightSource with
      | constant _ => cases keyEqual
      | running _ rightView =>
          simp only [sourceKey] at keyEqual
          injection keyEqual with indexEqual
          unfold ConcreteNifsOperationalOccurrence.sourceExpression
          apply fCarried_eq_of_numeric_eq
          unfold PaperNifsGlobalColumnMap.fLocation
          apply PaperNifsGlobalColumnMap.locate_source_congr
          · exact orderedIds_eq_of_constraintPolynomial_eq
              template left right same
          · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
              (PaperNifsCallFrame.runningOperand
                (invokePlan (template.withSystem left)).frame.operands)
              (PaperNifsCallFrame.runningOperand
                (invokePlan (template.withSystem right)).frame.operands)
              (PaperNifsCallFrame.running_widthsAgree
                (invokePlan (template.withSystem left)).frame)
              (PaperNifsCallFrame.running_widthsAgree
                (invokePlan (template.withSystem right)).frame)
              leftView.index rightView.index
              (runningOperandIds_eq_of_constraintPolynomial_eq
                template left right same)
              indexEqual
      | fresh _ _ => cases keyEqual
      | proof _ _ => cases keyEqual
  | fresh _ leftView =>
      cases rightSource with
      | constant _ => cases keyEqual
      | running _ _ => cases keyEqual
      | fresh _ rightView =>
          simp only [sourceKey] at keyEqual
          injection keyEqual with indexEqual
          unfold ConcreteNifsOperationalOccurrence.sourceExpression
          apply fCarried_eq_of_numeric_eq
          unfold PaperNifsGlobalColumnMap.fLocation
          apply PaperNifsGlobalColumnMap.locate_source_congr
          · exact orderedIds_eq_of_constraintPolynomial_eq
              template left right same
          · exact PaperNifsCodecProjection.coordinateId_eq_of_ids
              (PaperNifsCallFrame.freshOperand
                (invokePlan (template.withSystem left)).frame.operands)
              (PaperNifsCallFrame.freshOperand
                (invokePlan (template.withSystem right)).frame.operands)
              (PaperNifsCallFrame.fresh_widthsAgree
                (invokePlan (template.withSystem left)).frame)
              (PaperNifsCallFrame.fresh_widthsAgree
                (invokePlan (template.withSystem right)).frame)
              leftView.index rightView.index
              (freshOperandIds_eq_of_constraintPolynomial_eq
                template left right same)
              indexEqual
      | proof _ _ => cases keyEqual
  | proof _ leftView =>
      cases rightSource with
      | constant _ => cases keyEqual
      | running _ _ => cases keyEqual
      | fresh _ _ => cases keyEqual
      | proof _ rightView =>
          simp only [sourceKey] at keyEqual
          injection keyEqual with indexEqual
          unfold ConcreteNifsOperationalOccurrence.sourceExpression
          apply fCarried_eq_of_numeric_eq
          unfold ConcreteNifsOperationalOccurrence.proofFieldLocation
          exact proofFNumeric_eq_of_ids_and_index
            (invokePlan (template.withSystem left)).frame
            (invokePlan (template.withSystem right)).frame
            leftView rightView
            (orderedIds_eq_of_constraintPolynomial_eq
              template left right same)
            (proofOperandIds_eq_of_constraintPolynomial_eq
              template left right same)
            indexEqual

/-- Equal constraint polynomials give the same compact source keys for the
complete statement serialization. -/
theorem statementSourceKeys_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ((operational (template.withSystem left)).statementSources.map
        (sourceKey (publicFits :=
          ConcreteNifsPlain270Profile.publicFits dimensions))) =
      ((operational (template.withSystem right)).statementSources.map
        (sourceKey (publicFits :=
          ConcreteNifsPlain270Profile.publicFits dimensions))) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

/-- Equal constraint polynomials give the same compact source keys for the
complete output serialization. -/
theorem outputSourceKeys_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ((operational (template.withSystem left)).outputSources.map
        (sourceKey (publicFits :=
          ConcreteNifsPlain270Profile.publicFits dimensions))) =
      ((operational (template.withSystem right)).outputSources.map
        (sourceKey (publicFits :=
          ConcreteNifsPlain270Profile.publicFits dimensions))) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

/-- Equal key lists and key-determined values give equal mapped lists, even
when the source element types differ. -/
theorem map_eq_of_key_map_eq
    {α β Key Value : Type}
    (left : List α)
    (right : List β)
    (leftKey : α → Key)
    (rightKey : β → Key)
    (leftValue : α → Value)
    (rightValue : β → Value)
    (keysEqual : left.map leftKey = right.map rightKey)
    (valueEqual :
      ∀ leftElement rightElement,
        leftKey leftElement = rightKey rightElement →
          leftValue leftElement = rightValue rightElement) :
    left.map leftValue = right.map rightValue := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail =>
          simp only [List.map_nil, List.map_cons] at keysEqual
          cases keysEqual
  | cons head tail inductionHypothesis =>
      cases right with
      | nil =>
          simp only [List.map_cons, List.map_nil] at keysEqual
          cases keysEqual
      | cons rightHead rightTail =>
          simp only [List.map_cons] at keysEqual
          injection keysEqual with headEqual tailEqual
          simp only [List.map_cons]
          rw [valueEqual head rightHead headEqual]
          rw [inductionHypothesis rightTail tailEqual]

/-- Equal constraint polynomials give identical physical statement
expressions. -/
theorem statementFields_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).statementFields =
      (transcript (template.withSystem right)).statementFields := by
  change
    (operational (template.withSystem left)).statementSources.map
        (ConcreteNifsOperationalOccurrence.sourceExpression
          (application (template.withSystem left)).family
          (invokePlan (template.withSystem left)).frame) =
      (operational (template.withSystem right)).statementSources.map
        (ConcreteNifsOperationalOccurrence.sourceExpression
          (application (template.withSystem right)).family
          (invokePlan (template.withSystem right)).frame)
  apply map_eq_of_key_map_eq
    (leftKey :=
      sourceKey (publicFits :=
        ConcreteNifsPlain270Profile.publicFits dimensions))
    (rightKey :=
      sourceKey (publicFits :=
        ConcreteNifsPlain270Profile.publicFits dimensions))
  · exact statementSourceKeys_eq_of_constraintPolynomial_eq
      template left right same
  · intro leftSource rightSource keyEqual
    exact sourceExpression_eq_of_key_eq
      template left right same leftSource rightSource keyEqual

/-- Equal constraint polynomials give identical physical output
expressions. -/
theorem outputFields_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).outputFields =
      (transcript (template.withSystem right)).outputFields := by
  change
    (operational (template.withSystem left)).outputSources.map
        (ConcreteNifsOperationalOccurrence.sourceExpression
          (application (template.withSystem left)).family
          (invokePlan (template.withSystem left)).frame) =
      (operational (template.withSystem right)).outputSources.map
        (ConcreteNifsOperationalOccurrence.sourceExpression
          (application (template.withSystem right)).family
          (invokePlan (template.withSystem right)).frame)
  apply map_eq_of_key_map_eq
    (leftKey :=
      sourceKey (publicFits :=
        ConcreteNifsPlain270Profile.publicFits dimensions))
    (rightKey :=
      sourceKey (publicFits :=
        ConcreteNifsPlain270Profile.publicFits dimensions))
  · exact outputSourceKeys_eq_of_constraintPolynomial_eq
      template left right same
  · intro leftSource rightSource keyEqual
    exact sourceExpression_eq_of_key_eq
      template left right same leftSource rightSource keyEqual

/-- Equal constraint polynomials give identical physical prior-duplex lane
expressions. -/
theorem priorLanes_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).priorLanes =
      (transcript (template.withSystem right)).priorLanes := by
  apply funext
  intro lane
  change
    (ConcreteNifsOperationalOccurrence.proofFieldLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      ((operational (template.withSystem left)).priorLane lane)).carried =
    (ConcreteNifsOperationalOccurrence.proofFieldLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      ((operational (template.withSystem right)).priorLane lane)).carried
  apply fCarried_eq_of_numeric_eq
  unfold ConcreteNifsOperationalOccurrence.proofFieldLocation
  apply proofFNumeric_eq_of_ids_and_index
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · cases left with
    | mk leftMatrices leftPolynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

/-- Equal constraint polynomials give the same transcript allocation base. -/
theorem transcriptBase_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).transcriptBase =
      (transcript (template.withSystem right)).transcriptBase := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold transcript
          rfl

/-- Equal constraint polynomials give the same prior transcript cursor. -/
theorem priorAbsorbed_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).priorAbsorbed =
      (transcript (template.withSystem right)).priorAbsorbed := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold transcript
          rfl

private theorem authority_ext
    {shape : SemanticShape}
    (left right : KSplitNcEndpoints.AuthorityColumns shape)
    (priorPoint : left.priorPoint = right.priorPoint)
    (claimedYRing : left.claimedYRing = right.claimedYRing)
    (outputYRing : left.outputYRing = right.outputYRing)
    (outputYZcol : left.outputYZcol = right.outputYZcol) :
    left = right := by
  cases left
  cases right
  simp only at *
  cases priorPoint
  cases claimedYRing
  cases outputYRing
  cases outputYZcol
  rfl

private theorem authority_priorPoint_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (authority (template.withSystem left)).priorPoint =
      (authority (template.withSystem right)).priorPoint := by
  apply funext
  intro coordinate
  change
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      ((operational
        (template.withSystem left)).endpointViews.priorPoint
          coordinate)).carried =
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      ((operational
        (template.withSystem right)).endpointViews.priorPoint
          coordinate)).carried
  apply carried_eq_of_numeric_eq
  exact priorPoint_numeric_eq_of_constraintPolynomial_eq
    template left right same coordinate

private theorem authority_claimedYRing_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (authority (template.withSystem left)).claimedYRing =
      (authority (template.withSystem right)).claimedYRing := by
  apply funext
  intro running
  apply funext
  intro matrix
  apply funext
  intro lane
  change
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      ((operational
        (template.withSystem left)).endpointViews.claimedYRing
          running matrix lane)).carried =
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      ((operational
        (template.withSystem right)).endpointViews.claimedYRing
          running matrix lane)).carried
  apply carried_eq_of_numeric_eq
  exact claimedYRing_numeric_eq_of_constraintPolynomial_eq
    template left right same running matrix lane

private theorem authority_outputYRing_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (authority (template.withSystem left)).outputYRing =
      (authority (template.withSystem right)).outputYRing := by
  apply funext
  intro source
  apply funext
  intro matrix
  apply funext
  intro lane
  change
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      ((operational
        (template.withSystem left)).endpointViews.outputYRing
          source matrix lane)).carried =
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      ((operational
        (template.withSystem right)).endpointViews.outputYRing
          source matrix lane)).carried
  apply carried_eq_of_numeric_eq
  exact outputYRing_numeric_eq_of_constraintPolynomial_eq
    template left right same source matrix lane

private theorem authority_outputYZcol_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (authority (template.withSystem left)).outputYZcol =
      (authority (template.withSystem right)).outputYZcol := by
  apply funext
  intro source
  apply funext
  intro lane
  change
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      ((operational
        (template.withSystem left)).endpointViews.outputYZcol
          source lane)).carried =
    (ConcreteNifsOperationalFrame.proofLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      ((operational
        (template.withSystem right)).endpointViews.outputYZcol
          source lane)).carried
  apply carried_eq_of_numeric_eq
  exact outputYZcol_numeric_eq_of_constraintPolynomial_eq
    template left right same source lane

/-- Equal constraint polynomials select the same verifier-owned endpoint
columns. Matrix coefficients remain semantic setup data. -/
theorem authority_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    authority (template.withSystem left) =
      authority (template.withSystem right) :=
  authority_ext _ _
    (authority_priorPoint_eq template left right same)
    (authority_claimedYRing_eq template left right same)
    (authority_outputYRing_eq template left right same)
    (authority_outputYZcol_eq template left right same)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput
