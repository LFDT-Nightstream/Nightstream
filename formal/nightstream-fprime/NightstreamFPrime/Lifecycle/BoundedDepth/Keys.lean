import NightstreamFPrime.Lifecycle.BoundedDepth.State
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKeys
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.LightNorm

/-! The bounded-depth scheme's production verifier keys. The refresh key is the production key with its own
transcript: the Π_DEC children are prover messages, so the refresh transcript absorbs their
commitments and evaluation families after the fresh claim (`refreshWords_children`). Their
point and public inputs come from the input accumulator, which the prior state hash binds. The
light key is `BoundedDepthKeys.lightKey` on the refresh key: one relation, one commitment key and
one Π_RLC algebra for both folds. Its transcript absorbs the fresh claim, binds the running
claim through the prior state hash as the production key does, and samples one challenge with
the production sampler. Each fold transcript starts from its own domain. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra)
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds (lightShape Keys)
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

def lightFoldDomainChunk : List F := domainChunk "Nightstream/SuperNeo/fold/light/v1"

def refreshFoldDomainChunk : List F := domainChunk "Nightstream/SuperNeo/fold/refresh/v1"

def lightFoldInitialState : Transcript.State :=
  Transcript.absorb Transcript.initialState lightFoldDomainChunk

def refreshFoldInitialState : Transcript.State :=
  Transcript.absorb Transcript.initialState refreshFoldDomainChunk

theorem foldDomainChunks_distinct :
    lightFoldDomainChunk ≠ refreshFoldDomainChunk ∧
      lightFoldDomainChunk ≠ Transcript.foldDomainChunk ∧
      refreshFoldDomainChunk ≠ Transcript.foldDomainChunk ∧
      lightFoldDomainChunk ≠ slotDomainChunk ∧ lightFoldDomainChunk ≠ stateDomainChunk ∧
      refreshFoldDomainChunk ≠ slotDomainChunk ∧
      refreshFoldDomainChunk ≠ stateDomainChunk := by
  decide

section Refresh

abbrev Children (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth) :=
  Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape

/-- Each Π_DEC child's commitment and evaluation family. -/
def childWords (children : Children logicalWidth publicFits) : List F :=
  (List.finRange productionShape.runningCount).flatMap fun child =>
    serializeCommitment (children.commitments child) ++
      serializeEvaluations (children.evaluations child)

def refreshWords (children : Children logicalWidth publicFits)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape) : List F :=
  (ProductionKey.publicInputBlocks fresh).flatten ++ childWords children

def refreshAbsorbPublicInput (state : Transcript.State)
    (children : Children logicalWidth publicFits)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape) : Transcript.State :=
  Transcript.absorb state (refreshWords children fresh)

def withRefreshTranscript {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (key : ProductionKey.KeyType relation) : ProductionKey.KeyType relation :=
  { key with
    initialTranscriptState := refreshFoldInitialState
    absorbPublicInput := refreshAbsorbPublicInput }

/-- The refresh key: the production key with the refresh transcript. -/
noncomputable def refreshKey (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ProductionKey.KeyType relation :=
  withRefreshTranscript (ProductionKey.key relation ajtai)

/-- The refresh transcript keeps the key's relation, algebras and Π_CCS oracle. -/
theorem withRefreshTranscript_relation
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    (key : ProductionKey.KeyType relation) :
    (withRefreshTranscript key).relationSource = key.relationSource ∧
      (withRefreshTranscript key).piRlcSemantics = key.piRlcSemantics ∧
      (withRefreshTranscript key).piRlcAlgebra = key.piRlcAlgebra ∧
      (withRefreshTranscript key).piDecAlgebra = key.piDecAlgebra ∧
      (withRefreshTranscript key).oracle = key.oracle :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem childBlock_length (children : Children logicalWidth publicFits)
    (child : Fin productionShape.runningCount) :
    (serializeCommitment (children.commitments child) ++
        serializeEvaluations (children.evaluations child)).length =
      productionProfile.commitmentWidth * ringDegree +
        (productionShape.matrixCount + 1) * productionShape.coefficientCount * 2 := by
  simp

theorem childWords_length (children : Children logicalWidth publicFits) :
    (childWords children).length =
      productionShape.runningCount * (productionProfile.commitmentWidth * ringDegree +
        (productionShape.matrixCount + 1) * productionShape.coefficientCount * 2) := by
  simp only [childWords, List.length_flatMap, childBlock_length, List.map_const',
    List.sum_replicate, List.length_finRange, smul_eq_mul]

theorem finRange_flatMap_eq {count width : Nat} (left right : Fin count → List F)
    (leftLength : ∀ index, (left index).length = width)
    (rightLength : ∀ index, (right index).length = width)
    (same : (List.finRange count).flatMap left = (List.finRange count).flatMap right) :
    ∀ index, left index = right index := by
  induction count with
  | zero => exact fun index => Fin.elim0 index
  | succ count inductionHypothesis =>
      simp only [List.finRange_succ, List.flatMap_cons, List.flatMap_map] at same
      obtain ⟨head, tail⟩ := List.append_inj same (by rw [leftLength, rightLength])
      intro index
      refine Fin.cases head (fun index => ?_) index
      exact inductionHypothesis (fun index => left index.succ) (fun index => right index.succ)
        (fun _ => leftLength _) (fun _ => rightLength _) tail index

theorem childWords_injective {left right : Children logicalWidth publicFits}
    (same : childWords left = childWords right) :
    left.commitments = right.commitments ∧ left.evaluations = right.evaluations := by
  have blocks := finRange_flatMap_eq _ _ (childBlock_length left) (childBlock_length right) same
  have parts := fun child => List.append_inj (blocks child) (by simp)
  exact ⟨funext fun child => serializeCommitment_injective (parts child).1,
    funext fun child => serializeEvaluations_injective (parts child).2⟩

/-- The words that the refresh transcript absorbs fix the children's commitments and
evaluation families. -/
theorem refreshWords_children {left right : Children logicalWidth publicFits}
    {leftFresh rightFresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape}
    (same : refreshWords left leftFresh = refreshWords right rightFresh) :
    left.commitments = right.commitments ∧ left.evaluations = right.evaluations :=
  childWords_injective
    (List.append_inj' same (by rw [childWords_length, childWords_length])).2

end Refresh

section Light

abbrev LightShape : Shape := lightShape productionShape

def lightCoinPosition : FiatShamir.ChallengeLabel LightShape → Nat
  | .alpha coordinate => coordinate.val
  | .gamma => LightShape.cubeVariables
  | .sumcheck _ => 0

/-- `Transcript.piCcsOracle` on the light shape. -/
def lightPiCcsOracle : PiCCS.TranscriptReplay.Oracle K Transcript.State LightShape where
  transcript :=
    { initialState := fun statement => statement.priorState
      absorbRound := fun state _round message =>
        Transcript.absorb state (Transcript.serializeMessage message)
      squeeze := fun state label => Transcript.squeezeAt state (lightCoinPosition label) }
  initialState_is_prior := fun _ => rfl

def lightAbsorbPublicInput (state : Transcript.State)
    (_running : Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape) : Transcript.State :=
  Transcript.absorb state
    (ProductionKey.publicInputBlocks (publicFits := publicFits)
      ⟨fresh.commitments, fresh.publicInputs⟩).flatten

def lightOutputWords (output : FullOutputCoordinates.FullOutput K LightShape) : List F :=
  (List.finRange LightShape.sourceCount).flatMap fun source =>
    serializeEvaluations ⟨output.padCoordinate source, output.matrixCoordinate source⟩

def lightAbsorbOutput (state : Transcript.State)
    (output : FullOutputCoordinates.FullOutput K LightShape) : Transcript.State :=
  Transcript.absorb state (lightOutputWords output)

/-- The light Π_RLC challenge: the first challenge of the production sampler. -/
def lightChallenge (state : Transcript.State) : RingF :=
  Transcript.PiRlcSampler.piRlcChallenges state 1 0

theorem lightChallenge_valid (state : Transcript.State) :
    RingRelation.PiRLCAlgebra.Challenge.challengeValid (lightChallenge state) :=
  Transcript.PiRlcSampler.piRlcChallenges_member state 1 0

/-- Cyclo (ePrint 2026/359) Theorem 3, item 1, on the production Π_RLC algebra. -/
theorem lightNormLaw (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Light.Rlc.NormLaw (PaperAlgebra.piRlcAlgebra ajtai) ringFOne :=
  ⟨(RingRelation.PiRLCAlgebra.LightNorm.norm_law ajtai).add_bound⟩

noncomputable def lightParts (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    HyperNova.Construction2.BoundedDepthKeys.LightParts (refreshKey relation ajtai) where
  one := ringFOne
  normLaw := lightNormLaw ajtai
  oracle := lightPiCcsOracle
  initialTranscriptState := lightFoldInitialState
  absorbPublicInput := lightAbsorbPublicInput
  absorbPiCcsOutput := lightAbsorbOutput
  piRlcResponse := fun state => some (lightChallenge state)
  piRlcResponseValid := by
    intro state challenge returned
    rw [← Option.some.inj returned]
    exact lightChallenge_valid state

/-- The bounded-depth scheme's production verifier keys. -/
noncomputable def keys (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :=
  HyperNova.Construction2.BoundedDepthKeys.keys (refreshKey relation ajtai) rfl
    (lightParts relation ajtai)

/-- SuperNeo v1.2 Lemma 8's extraction algebra for the production light key. -/
def lightExtraction (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    ExtractionAlgebra
      (HyperNova.Construction2.BoundedDepthKeys.semanticsToLight
        (columns := RingCarrierLayout.carrierWidth logicalWidth)
        (blockCount := ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth))
        (semantics ajtai))
      productionGlobalParams
      (HyperNova.Construction2.BoundedDepthKeys.algebraToLight (piRlcAlgebra ajtai)) :=
  HyperNova.Construction2.BoundedDepthKeys.extractionToLight
    (PaperExtractionAlgebra.extractionAlgebra ajtai)

/-- Cyclo Theorem 3, item 2, for the production light key: the extracted accumulator
`w − ρ · u` is within `T = 128` of `w`. -/
theorem lightExtractionNormLaw
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Light.Rlc.ExtractionNormLaw (lightExtraction ajtai) :=
  HyperNova.Construction2.BoundedDepthKeys.extractionNormLawToLight
    ⟨fun bound challenge accumulated fresh valid accumulatedBounded freshBounded =>
      RingRelation.PiRLCAlgebra.LightNorm.sub_act_bound
        (shape := FullShape logicalWidth publicFits) accumulated fresh challenge valid bound
        accumulatedBounded freshBounded⟩

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The two folds share one relation (`BoundedDepthKeys.keys_coherent`). -/
theorem keys_coherent :
    HyperNova.Construction2.BoundedDepthFolds.sourceFromLight
        (keys relation ajtai).light.relationSource =
        (keys relation ajtai).refresh.relationSource ∧
      (keys relation ajtai).light.params = (keys relation ajtai).refresh.params ∧
      (keys relation ajtai).light.openingMaps = (keys relation ajtai).refresh.openingMaps ∧
      (keys relation ajtai).light.piRlcSemantics =
        HyperNova.Construction2.BoundedDepthKeys.semanticsToLight
          (keys relation ajtai).refresh.piRlcSemantics ∧
      (keys relation ajtai).light.piRlcAlgebra =
        HyperNova.Construction2.BoundedDepthKeys.algebraToLight
          (keys relation ajtai).refresh.piRlcAlgebra :=
  HyperNova.Construction2.BoundedDepthKeys.keys_coherent _ _ _

/-- The refresh key keeps the production relation and changes only the transcript. -/
theorem refreshKey_relation :
    (refreshKey relation ajtai).relationSource =
        (ProductionKey.key relation ajtai).relationSource ∧
      (refreshKey relation ajtai).piRlcSemantics =
        (ProductionKey.key relation ajtai).piRlcSemantics ∧
      (refreshKey relation ajtai).piRlcAlgebra = (ProductionKey.key relation ajtai).piRlcAlgebra ∧
      (refreshKey relation ajtai).piDecAlgebra = (ProductionKey.key relation ajtai).piDecAlgebra ∧
      (refreshKey relation ajtai).oracle = (ProductionKey.key relation ajtai).oracle :=
  withRefreshTranscript_relation _

end Light

end NightstreamFPrime.Lifecycle.BoundedDepth
