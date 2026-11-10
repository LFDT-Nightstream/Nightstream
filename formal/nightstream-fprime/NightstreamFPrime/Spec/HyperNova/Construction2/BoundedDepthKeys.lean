import NightstreamFPrime.Spec.Folding.Light.Extraction
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds

/-! The bounded-depth scheme's light key, built from its refresh key. The light key reads the refresh key's
relation through the shape conversion of `BoundedDepthFolds`: the same opening maps, matrices,
norm predicate and Π_RLC algebra, with each evaluation family renamed to the light shape. So a
claim holds in the light relation exactly when it holds in the refresh relation
(`holds_fromLight_iff`, `holds_claimToLight_iff`), and the extraction algebra of SuperNeo v1.2
Lemma 8 carries over (`extractionToLight`). Only the Fiat–Shamir transcript and the sampler of
one challenge belong to the light key alone (`LightParts`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKeys

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra LinearMapLaws)
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Key Fresh Running)
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

section Shape

@[simp] theorem evaluationsFromLight_toLight (family : EvaluationFamily K shape) :
    evaluationsFromLight (evaluationsToLight family) = family :=
  rfl

@[simp] theorem evaluationsToLight_fromLight (family : EvaluationFamily K (lightShape shape)) :
    evaluationsToLight (evaluationsFromLight family) = family :=
  rfl

@[simp] theorem map_fromLight_toLight (families : Array (EvaluationFamily K shape)) :
    (families.map evaluationsToLight).map evaluationsFromLight = families := by
  rw [Array.map_map]
  exact Array.map_id families

@[simp] theorem map_toLight_fromLight
    (families : Array (EvaluationFamily K (lightShape shape))) :
    (families.map evaluationsFromLight).map evaluationsToLight = families := by
  rw [Array.map_map]
  exact Array.map_id families

theorem map_toLight_eq_iff (families : Array (EvaluationFamily K shape))
    (light : Array (EvaluationFamily K (lightShape shape))) :
    families.map evaluationsToLight = light ↔ families = light.map evaluationsFromLight := by
  constructor
  · rintro rfl
    exact (map_fromLight_toLight families).symm
  · rintro rfl
    exact map_toLight_fromLight light

def matrixSourceToLight (source : MatrixSource F shape columns blockCount) :
    MatrixSource F (lightShape shape) columns blockCount :=
  ⟨source.columnLayout, source.matrices, source.constraintPolynomial, source.kernel⟩

def sourceToLight (source : RelationSource shape columns blockCount) :
    RelationSource (lightShape shape) columns blockCount :=
  ⟨source.cubeLayout, matrixSourceToLight source.matrixSource⟩

@[simp] theorem sourceFromLight_toLight (source : RelationSource shape columns blockCount) :
    sourceFromLight (sourceToLight source) = source :=
  rfl

/-- A slot claim as an instance of the light shape: the inverse of `fromLight`. -/
def claimToLight (claim : Claim Commitment PublicInput shape columns blockCount) :
    CE.Instance (RelationSource (lightShape shape) columns blockCount) PublicInput
      (CubePoint K (lightShape shape).cubeVariables) (EvaluationFamily K (lightShape shape))
      Commitment where
  constraintSystem := sourceToLight claim.constraintSystem
  commitment := claim.commitment
  publicInput := claim.publicInput
  point := claim.point
  evaluations := claim.evaluations.map evaluationsToLight
  stage := claim.stage

@[simp] theorem fromLight_claimToLight
    (claim : Claim Commitment PublicInput shape columns blockCount) :
    fromLight (claimToLight claim) = claim := by
  simp only [fromLight, claimToLight, sourceFromLight_toLight, map_fromLight_toLight]

end Shape

section Relation

variable {params : GlobalParams}

/-- The refresh semantics on the light shape. -/
def semanticsToLight
    (semantics : RelationSemantics (RelationSource shape columns blockCount)
      (Assignment F columns) PublicInput (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape) Commitment) :
    RelationSemantics (RelationSource (lightShape shape) columns blockCount)
      (Assignment F columns) PublicInput (CubePoint K (lightShape shape).cubeVariables)
      (EvaluationFamily K (lightShape shape)) Commitment where
  commit := semantics.commit
  projectPublicInput := semantics.projectPublicInput
  normBounded := semantics.normBounded
  ccsSatisfied := fun source => semantics.ccsSatisfied (sourceFromLight source)
  evaluationPointValid := fun source => semantics.evaluationPointValid (sourceFromLight source)
  evaluations := fun source assignment point =>
    (semantics.evaluations (sourceFromLight source) assignment point).map evaluationsToLight

variable
  {semantics : RelationSemantics (RelationSource shape columns blockCount)
    (Assignment F columns) PublicInput (CubePoint K shape.cubeVariables)
    (EvaluationFamily K shape) Commitment}

/-- The refresh Π_RLC algebra on the light shape: evaluation families are combined in the
refresh shape. -/
def algebraToLight
    (algebra : PiRLC.Algebra (RelationSource shape columns blockCount) (Assignment F columns)
      PublicInput (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment Scalar
      semantics params) :
    PiRLC.Algebra (RelationSource (lightShape shape) columns blockCount) (Assignment F columns)
      PublicInput (CubePoint K (lightShape shape).cubeVariables)
      (EvaluationFamily K (lightShape shape)) Commitment Scalar (semanticsToLight semantics)
      params where
  challengeValid := algebra.challengeValid
  combineAssignment := algebra.combineAssignment
  combineCommitment := algebra.combineCommitment
  combinePublicInput := algebra.combinePublicInput
  combineEvaluations := fun challenges families =>
    (algebra.combineEvaluations challenges fun index =>
      (families index).map evaluationsFromLight).map evaluationsToLight
  commit_hom := algebra.commit_hom
  publicInput_hom := algebra.publicInput_hom
  evaluations_hom := by
    intro count system point challenges assignments
    simp only [semanticsToLight, map_fromLight_toLight]
    exact congrArg (Array.map evaluationsToLight)
      (algebra.evaluations_hom (sourceFromLight system) point challenges assignments)
  norm_growth := algebra.norm_growth

/-- SuperNeo v1.2 Definition 21 holds for a light output exactly when it holds for the slot
claim that the output becomes. -/
theorem holds_fromLight_iff
    (output : CE.Instance (RelationSource (lightShape shape) columns blockCount) PublicInput
      (CubePoint K (lightShape shape).cubeVariables) (EvaluationFamily K (lightShape shape))
      Commitment)
    (assignment : Assignment F columns) :
    CE.Holds semantics params (fromLight output) assignment ↔
      CE.Holds (semanticsToLight semantics) params output assignment := by
  simp only [CE.Holds, fromLight, semanticsToLight, Opening.Holds,
    map_toLight_eq_iff]

/-- `holds_fromLight_iff` at an explicit norm bound. -/
theorem boundHolds_fromLight_iff
    (output : CE.Instance (RelationSource (lightShape shape) columns blockCount) PublicInput
      (CubePoint K (lightShape shape).cubeVariables) (EvaluationFamily K (lightShape shape))
      Commitment)
    (bound : Nat) (assignment : Assignment F columns) :
    Light.Rlc.Holds semantics bound (fromLight output) assignment ↔
      Light.Rlc.Holds (semanticsToLight semantics) bound output assignment := by
  simp only [Light.Rlc.Holds, fromLight, semanticsToLight, Opening.Holds,
    map_toLight_eq_iff]

theorem holds_claimToLight_iff (claim : Claim Commitment PublicInput shape columns blockCount)
    (assignment : Assignment F columns) :
    CE.Holds (semanticsToLight semantics) params (claimToLight claim) assignment ↔
      CE.Holds semantics params claim assignment := by
  rw [← holds_fromLight_iff, fromLight_claimToLight]

theorem boundHolds_claimToLight_iff
    (claim : Claim Commitment PublicInput shape columns blockCount) (bound : Nat)
    (assignment : Assignment F columns) :
    Light.Rlc.Holds (semanticsToLight semantics) bound (claimToLight claim) assignment ↔
      Light.Rlc.Holds semantics bound claim assignment := by
  rw [← boundHolds_fromLight_iff, fromLight_claimToLight]

end Relation

section Extraction

variable {params : GlobalParams}
  {semantics : RelationSemantics (RelationSource shape columns blockCount)
    (Assignment F columns) PublicInput (CubePoint K shape.cubeVariables)
    (EvaluationFamily K shape) Commitment}
  {algebra : PiRLC.Algebra (RelationSource shape columns blockCount) (Assignment F columns)
    PublicInput (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment Scalar
    semantics params}

def evaluationModuleToLight (module : ModuleOps Scalar (EvaluationFamily K shape)) :
    ModuleOps Scalar (EvaluationFamily K (lightShape shape)) where
  zero := evaluationsToLight module.zero
  add := fun left right =>
    evaluationsToLight (module.add (evaluationsFromLight left) (evaluationsFromLight right))
  neg := fun value => evaluationsToLight (module.neg (evaluationsFromLight value))
  smul := fun scalar value => evaluationsToLight (module.smul scalar (evaluationsFromLight value))

theorem evaluationLawsToLight {ring : CommutativeRingOps Scalar}
    {module : ModuleOps Scalar (EvaluationFamily K shape)} (laws : ModuleLaws ring module) :
    ModuleLaws ring (evaluationModuleToLight module) where
  add_assoc := fun _ _ _ => congrArg evaluationsToLight (laws.add_assoc _ _ _)
  add_comm := fun _ _ => congrArg evaluationsToLight (laws.add_comm _ _)
  zero_add := fun _ => congrArg evaluationsToLight (laws.zero_add _)
  add_zero := fun _ => congrArg evaluationsToLight (laws.add_zero _)
  add_neg := fun _ => congrArg evaluationsToLight (laws.add_neg _)
  zero_smul := fun _ => congrArg evaluationsToLight (laws.zero_smul _)
  add_smul := fun _ _ _ => congrArg evaluationsToLight (laws.add_smul _ _ _)
  one_smul := fun _ => congrArg evaluationsToLight (laws.one_smul _)
  mul_smul := fun _ _ _ => congrArg evaluationsToLight (laws.mul_smul _ _ _)
  smul_zero := fun _ => congrArg evaluationsToLight (laws.smul_zero _)
  smul_add := fun _ _ _ => congrArg evaluationsToLight (laws.smul_add _ _ _)

theorem linearCombination_toLight (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar (EvaluationFamily K shape)) :
    ∀ {count : Nat} (coefficients : Fin count → Scalar)
      (values : Fin count → EvaluationFamily K (lightShape shape)),
    linearCombination ring (evaluationModuleToLight module) coefficients values =
      evaluationsToLight (linearCombination ring module coefficients
        fun index => evaluationsFromLight (values index))
  | 0, _, _ => rfl
  | count + 1, coefficients, values => by
      simp only [linearCombination]
      rw [linearCombination_toLight ring module (count := count)]
      rfl

theorem getD_map {Value Result : Type} (map : Value → Result) (values : Array Value)
    (index : Nat) (fallback : Value) :
    (values.map map).getD index (map fallback) = map (values.getD index fallback) := by
  by_cases inside : index < values.size
  · simp [Array.getD, inside]
  · simp [Array.getD, inside]

theorem linearMapToLight {Source : Type} {sourceModule : ModuleOps Scalar Source}
    {module : ModuleOps Scalar (EvaluationFamily K shape)}
    {map : Source → EvaluationFamily K shape} (laws : LinearMapLaws sourceModule module map) :
    LinearMapLaws sourceModule (evaluationModuleToLight module)
      fun value => evaluationsToLight (map value) :=
  ⟨fun left right => congrArg evaluationsToLight (laws.map_sub left right),
    fun scalar value => congrArg evaluationsToLight (laws.map_smul scalar value)⟩

theorem combineEvaluations_getD_toLight (laws : ExtractionAlgebra semantics params algebra)
    {count : Nat} (coefficients : Fin count → Scalar)
    (values : Fin count → Array (EvaluationFamily K (lightShape shape)))
    (expectedSize index : Nat) (positive : 0 < count)
    (sizes : ∀ source, (values source).size = expectedSize) :
    ((algebraToLight algebra).combineEvaluations coefficients values).getD index
        (evaluationModuleToLight laws.evaluationModule).zero =
      linearCombination laws.ring (evaluationModuleToLight laws.evaluationModule) coefficients
        fun source =>
          (values source).getD index (evaluationModuleToLight laws.evaluationModule).zero := by
  rw [linearCombination_toLight]
  have combined := laws.combineEvaluations_getD coefficients
    (fun source => (values source).map evaluationsFromLight) expectedSize index positive
    (fun source => by simpa using sizes source)
  refine (getD_map evaluationsToLight _ index _).trans ?_
  rw [combined]
  congr 2
  funext source
  exact getD_map evaluationsFromLight (values source) index
    (evaluationsToLight laws.evaluationModule.zero)

/-- SuperNeo v1.2 Lemma 8's extraction algebra on the light shape. -/
def extractionToLight (laws : ExtractionAlgebra semantics params algebra) :
    ExtractionAlgebra (semanticsToLight semantics) params (algebraToLight algebra) where
  ring := laws.ring
  ringLaws := laws.ringLaws
  assignmentModule := laws.assignmentModule
  assignmentLaws := laws.assignmentLaws
  commitmentModule := laws.commitmentModule
  commitmentLaws := laws.commitmentLaws
  publicInputModule := laws.publicInputModule
  publicInputLaws := laws.publicInputLaws
  evaluationModule := evaluationModuleToLight laws.evaluationModule
  evaluationLaws := evaluationLawsToLight laws.evaluationLaws
  combineCommitment_eq := laws.combineCommitment_eq
  combinePublicInput_eq := laws.combinePublicInput_eq
  semanticEvaluations_size_eq := by
    intro system point left right
    simpa [semanticsToLight] using
      laws.semanticEvaluations_size_eq (sourceFromLight system) point left right
  combineEvaluations_size := by
    intro count coefficients values expectedSize positive sizes
    simpa [algebraToLight] using
      laws.combineEvaluations_size coefficients
        (fun index => (values index).map evaluationsFromLight) expectedSize positive
        (fun index => by simpa using sizes index)
  combineEvaluations_getD := combineEvaluations_getD_toLight laws
  commitMap := laws.commitMap
  publicInputMap := laws.publicInputMap
  evaluationsMap := by
    intro system point index
    have images : (fun assignment =>
        ((semanticsToLight semantics).evaluations system assignment point).getD index
          (evaluationModuleToLight laws.evaluationModule).zero) = fun assignment =>
        evaluationsToLight ((semantics.evaluations (sourceFromLight system) assignment point).getD
          index laws.evaluationModule.zero) :=
      funext fun assignment => getD_map evaluationsToLight _ index _
    exact (congrArg (LinearMapLaws _ _) images).mpr
      (linearMapToLight (laws.evaluationsMap (sourceFromLight system) point index))
  correctedNormCoverage := laws.correctedNormCoverage

/-- Cyclo (ePrint 2026/359) Theorem 3, item 2, carries over: the light extraction subtracts
with the refresh assignment module. -/
theorem extractionNormLawToLight {laws : ExtractionAlgebra semantics params algebra}
    (normLaw : Light.Rlc.ExtractionNormLaw laws) :
    Light.Rlc.ExtractionNormLaw (extractionToLight laws) :=
  ⟨normLaw.sub_bound⟩

end Extraction

section Keys

/-- The parts of a light key that the refresh key does not fix: the unit coefficient of the
running source, the Fiat–Shamir transcript over one fresh and one running source, and the
sampler of one challenge. -/
structure LightParts
    (refresh : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    where
  one : Scalar
  normLaw : Light.Rlc.NormLaw refresh.piRlcAlgebra one
  oracle : PiCCS.TranscriptReplay.Oracle K State (lightShape shape)
  initialTranscriptState : State
  absorbPublicInput : State → Running K Commitment PublicInput (lightShape shape) →
    Fresh Commitment PublicInput (lightShape shape) → State
  absorbPiCcsOutput : State → FullOutputCoordinates.FullOutput K (lightShape shape) → State
  piRlcResponse : State → Option Scalar
  piRlcResponseValid : ∀ state challenge,
    piRlcResponse state = some challenge → refresh.piRlcAlgebra.challengeValid challenge

/-- The light key: the refresh key's relation on the light shape, with the light parts. -/
def lightKey
    (refresh : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (freshCount : shape.freshCount = 1) (parts : LightParts refresh) :
    Light.Key Commitment PublicInput Scalar State (lightShape shape) columns blockCount
      degreeBound where
  baseOps := refresh.baseOps
  baseLaws := refresh.baseLaws
  baseZero := refresh.baseZero
  noZeroDivisors := refresh.noZeroDivisors
  extensionOps := refresh.extensionOps
  extensionLaws := refresh.extensionLaws
  extensionZeroLaws := refresh.extensionZeroLaws
  lift := refresh.lift
  liftLaws := refresh.liftLaws
  openingMaps := refresh.openingMaps
  params := refresh.params
  freshCount_eq := freshCount
  runningCount_eq := rfl
  cubeLayout := refresh.cubeLayout
  matrixSource := matrixSourceToLight refresh.matrixSource
  degreeBoundExact := refresh.degreeBoundExact
  constantLaw := refresh.constantLaw
  piRlcSemantics := semanticsToLight refresh.piRlcSemantics
  openingAgreement := refresh.openingAgreement
  evaluationAgreement := by
    intro assignment point
    have agreement := refresh.evaluationAgreement assignment point
    refine ⟨agreement.1, ?_⟩
    change (refresh.piRlcSemantics.evaluations refresh.relationSource assignment point).map
      evaluationsToLight = _
    rw [Key.relationSource, agreement.2]
    simp only [paperRelationSemantics, List.map_toArray, List.map_cons, List.map_nil]
    rfl
  piRlcAlgebra := algebraToLight refresh.piRlcAlgebra
  one := parts.one
  normLaw := ⟨parts.normLaw.add_bound⟩
  oracle := parts.oracle
  initialTranscriptState := parts.initialTranscriptState
  absorbPublicInput := parts.absorbPublicInput
  absorbPiCcsOutput := parts.absorbPiCcsOutput
  piRlcResponse := parts.piRlcResponse
  piRlcResponseValid := parts.piRlcResponseValid

/-- The bounded-depth scheme's verifier keys for one slot. -/
def keys
    (refresh : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (freshCount : shape.freshCount = 1) (parts : LightParts refresh) :
    Keys Commitment PublicInput Scalar State shape columns blockCount degreeBound degreeBound where
  light := lightKey refresh freshCount parts
  refresh := refresh

variable
  (refresh : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
  (freshCount : shape.freshCount = 1) (parts : LightParts refresh)

/-- The two folds share one relation: the light key's constraint system, parameters, opening
maps, semantics and Π_RLC algebra are the refresh key's on the light shape. -/
theorem keys_coherent :
    sourceFromLight (keys refresh freshCount parts).light.relationSource =
        (keys refresh freshCount parts).refresh.relationSource ∧
      (keys refresh freshCount parts).light.params =
        (keys refresh freshCount parts).refresh.params ∧
      (keys refresh freshCount parts).light.openingMaps =
        (keys refresh freshCount parts).refresh.openingMaps ∧
      (keys refresh freshCount parts).light.piRlcSemantics =
        semanticsToLight (keys refresh freshCount parts).refresh.piRlcSemantics ∧
      (keys refresh freshCount parts).light.piRlcAlgebra =
        algebraToLight (keys refresh freshCount parts).refresh.piRlcAlgebra :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem light_verify_output
    {light : Light.Key Commitment PublicInput Scalar State (lightShape shape) columns blockCount
      degreeBound}
    {running : Running K Commitment PublicInput (lightShape shape)}
    {fresh : Fresh Commitment PublicInput (lightShape shape)}
    {proof : Light.Proof (lightShape shape) degreeBound}
    {output : CE.Instance (RelationSource (lightShape shape) columns blockCount) PublicInput
      (CubePoint K (lightShape shape).cubeVariables) (EvaluationFamily K (lightShape shape))
      Commitment}
    (accepted : Light.verify light running fresh proof = some output) :
    output.constraintSystem = light.relationSource ∧ output.stage = .combined := by
  unfold Light.verify at accepted
  split at accepted
  · obtain ⟨_, _, rfl⟩ := Option.map_eq_some_iff.mp accepted
    exact ⟨rfl, rfl⟩
  · cases accepted

theorem refresh_verify_output
    {accumulator : Claim Commitment PublicInput shape columns blockCount}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Refresh.Proof Commitment shape degreeBound}
    {output : Claim Commitment PublicInput shape columns blockCount}
    (accepted : Refresh.verify refresh accumulator fresh proof = some output) :
    output.constraintSystem = refresh.relationSource ∧ output.stage = .combined := by
  obtain ⟨_, _, parent⟩ :=
    (Refresh.verify_eq_some_iff refresh accumulator fresh proof output).mp accepted
  simp only [Nifs.PaperNonInteractive.Key.parent, Option.map_eq_some_iff] at parent
  obtain ⟨_, _, rfl⟩ := parent
  exact ⟨rfl, rfl⟩

/-- Every bounded-depth fold output is a claim of the refresh key's constraint system at stage
`combined`, so a slot can carry it without its constraint system and stage. -/
theorem folds_output {step : Folding.Schedule.Step}
    {claim : Claim Commitment PublicInput shape columns blockCount}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : FoldProof Commitment shape degreeBound degreeBound}
    {output : Claim Commitment PublicInput shape columns blockCount}
    (accepted : folds step (keys refresh freshCount parts) claim fresh proof = some output) :
    output.constraintSystem = refresh.relationSource ∧ output.stage = .combined := by
  match step, proof, accepted with
  | .light, .light proof, accepted =>
      obtain ⟨_, lightOutput, _, verified, rfl⟩ := (light_eq_some_iff _ _ _ _ _).mp accepted
      have shape := light_verify_output verified
      simp only [fromLight, shape.1, shape.2]
      exact ⟨rfl, trivial⟩
  | .refresh, .refresh proof, accepted =>
      rw [refresh_eq] at accepted
      exact refresh_verify_output refresh accepted
  | .light, .refresh _, accepted =>
      rw [light_rejects_refresh_proof] at accepted
      cases accepted
  | .refresh, .light _, accepted =>
      rw [refresh_rejects_light_proof] at accepted
      cases accepted

end Keys

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKeys
