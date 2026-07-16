import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Parameters
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.ProjectionPrefix

/-!
Owns: the model-level fixed Pi_RLC projection-binding profile, including the
exact relationship between active 54-coefficient projections and their
authoritative 64-entry `y_ring` and `y_zcol` carriers.

Does not own: versioned label framing, serialized field counts, a generated
fixed-F-prime Rust conformance artifact, SIS binding security, or R1CS rows.

Emits constraints: no. The predicates below make the active-prefix and
zero-padding obligations explicit before any concrete serializer is counted.

Authority boundary: these are structural shape facts. They do not make a
digest authoritative and do not permit row removal.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `ProjectionLaneShape` | `nifs.pi_rlc.verify.projection_binding.{combined,quotient}.*` | One active polynomial has 54 coefficients and its quotient has 53 | Model-level lists | No |
| `PaddedCarrierShape` | `nifs.pi_rlc.verify.padding.{y_ring,y_zcol}` | One carrier has 64 entries and exactly ten zero tail entries | Model-level carrier | No |
| `PaddedProjectionLaneShape` | projection binding plus padding | The serialized combined polynomial is exactly `carrier.take 54` | Carrier and binding supplied together | No |
| `PlainFixedProfileShape` | complete plain projection binding | 15 outputs, 18 commitment lanes, no adv, 5 X lanes, 6 `y_ring` limb carriers, and 2 `y_zcol` limb carriers | Fixed model parameters | No - Rust conformance artifact required |
| `plainFixedProfile_yRing_activePrefixes` | `projection_binding.combined.y_ring` | Every `y_ring` combined polynomial is its paired carrier's active prefix | `PlainFixedProfileShape` | No |
| `plainFixedProfile_yZcol_zeroTails` | `padding.y_zcol` | Every paired `y_zcol` carrier has ten zero tail entries | `PlainFixedProfileShape` | No |

The exact 3,616-field plain serializer count is a refinement theorem in
`Refinement/ProjectionBindingSerialization.lean`; it is deliberately not a
semantic-shape theorem.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

open PiRlcChallenge

/-! ## Active, quotient, and padded-carrier shapes -/

/-- Coefficients consumed by one degree-54 projection identity. -/
def ActivePolynomialShape (coefficients : List SuperNeo.F) : Prop :=
  coefficients.length = SuperNeo.d

/-- Quotient of division by `X - beta`, hence one coefficient shorter. -/
def ProjectionQuotientShape (quotient : List SuperNeo.F) : Prop :=
  quotient.length = SuperNeo.d - 1

/-- One active output polynomial paired with its projection quotient. -/
structure ProjectionLaneShape (binding : ProjectionLaneBinding) : Prop where
  combined : ActivePolynomialShape binding.combined
  quotient : ProjectionQuotientShape binding.quotient

/-- A fixed number of identically shaped active projection lanes. -/
structure ProjectionLaneFamilyShape
    (count : Nat) (bindings : List ProjectionLaneBinding) : Prop where
  count_eq : bindings.length = count
  lanes : bindings.Forall ProjectionLaneShape

/-- A fixed number of bare quotient lists. -/
structure ProjectionQuotientFamilyShape
    (count : Nat) (quotients : List (List SuperNeo.F)) : Prop where
  count_eq : quotients.length = count
  quotients : quotients.Forall ProjectionQuotientShape

/--
One authoritative split-NC limb carrier: 64 entries with the ten entries after
the active degree-54 prefix fixed to zero.
-/
structure PaddedCarrierShape (carrier : List SuperNeo.F) : Prop where
  length_eq : carrier.length = paddedDegree
  zero_tail :
    carrier.drop SuperNeo.d =
      List.replicate (paddedDegree - SuperNeo.d) (0 : SuperNeo.F)

/--
One padded carrier paired with the exact active polynomial and quotient that
the projection-binding serializer consumes.
-/
structure PaddedProjectionLaneShape
    (carrier : List SuperNeo.F) (binding : ProjectionLaneBinding) : Prop where
  carrier_shape : PaddedCarrierShape carrier
  active_prefix : binding.combined = carrier.take SuperNeo.d
  quotient : ProjectionQuotientShape binding.quotient

/-- A fixed family of padded carriers paired one-for-one with projection lanes. -/
structure PaddedProjectionLaneFamilyShape
    (count : Nat)
    (carriers : List (List SuperNeo.F))
    (bindings : List ProjectionLaneBinding) : Prop where
  carrier_count_eq : carriers.length = count
  binding_count_eq : bindings.length = count
  lanes : List.Forall₂ PaddedProjectionLaneShape carriers bindings

theorem activePolynomialShape_length
    {coefficients : List SuperNeo.F}
    (shape : ActivePolynomialShape coefficients) :
    coefficients.length = 54 := by
  simpa [ActivePolynomialShape, SuperNeo.d] using shape

theorem projectionQuotientShape_length
    {quotient : List SuperNeo.F}
    (shape : ProjectionQuotientShape quotient) :
    quotient.length = 53 := by
  simpa [ProjectionQuotientShape, SuperNeo.d] using shape

theorem paddedCarrierShape_length
    {carrier : List SuperNeo.F}
    (shape : PaddedCarrierShape carrier) :
    carrier.length = 64 := by
  simpa [paddedDegree] using shape.length_eq

/-- The padded carrier has exactly ten entries after the active prefix. -/
theorem paddedCarrierShape_zeroTail
    {carrier : List SuperNeo.F}
    (shape : PaddedCarrierShape carrier) :
    carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F) := by
  simpa [SuperNeo.d, paddedDegree] using shape.zero_tail

theorem paddedCarrierShape_tailLength
    {carrier : List SuperNeo.F}
    (shape : PaddedCarrierShape carrier) :
    (carrier.drop 54).length = 10 := by
  rw [paddedCarrierShape_zeroTail shape]
  simp

theorem paddedCarrierShape_tailForallZero
    {carrier : List SuperNeo.F}
    (shape : PaddedCarrierShape carrier) :
    (carrier.drop 54).Forall (fun value => value = 0) := by
  rw [paddedCarrierShape_zeroTail shape]
  simp

theorem paddedProjectionLaneShape_activePrefix
    {carrier : List SuperNeo.F} {binding : ProjectionLaneBinding}
    (shape : PaddedProjectionLaneShape carrier binding) :
    binding.combined = carrier.take 54 := by
  simpa [SuperNeo.d] using shape.active_prefix

theorem paddedProjectionLaneShape_projectionLaneShape
    {carrier : List SuperNeo.F} {binding : ProjectionLaneBinding}
    (shape : PaddedProjectionLaneShape carrier binding) :
    ProjectionLaneShape binding := by
  constructor
  · unfold ActivePolynomialShape
    rw [shape.active_prefix, List.length_take, shape.carrier_shape.length_eq]
    simp [SuperNeo.d, paddedDegree]
  · exact shape.quotient

/-- Active projection values are strictly shorter than their padded carriers. -/
theorem paddedProjectionLaneShape_combined_lt_carrier
    {carrier : List SuperNeo.F} {binding : ProjectionLaneBinding}
    (shape : PaddedProjectionLaneShape carrier binding) :
    binding.combined.length < carrier.length := by
  rw [activePolynomialShape_length
      (paddedProjectionLaneShape_projectionLaneShape shape).combined,
    paddedCarrierShape_length shape.carrier_shape]
  decide

theorem paddedProjectionLaneFamily_bindingShapes
    {count : Nat}
    {carriers : List (List SuperNeo.F)}
    {bindings : List ProjectionLaneBinding}
    (shape : PaddedProjectionLaneFamilyShape count carriers bindings) :
    bindings.Forall ProjectionLaneShape := by
  have lanes := shape.lanes
  clear shape
  induction lanes with
  | nil => simp
  | cons head _ inductionHypothesis =>
      simp only [List.forall_cons]
      exact ⟨paddedProjectionLaneShape_projectionLaneShape head,
        inductionHypothesis⟩

theorem paddedProjectionLaneFamily_activePrefixes
    {count : Nat}
    {carriers : List (List SuperNeo.F)}
    {bindings : List ProjectionLaneBinding}
    (shape : PaddedProjectionLaneFamilyShape count carriers bindings) :
    List.Forall₂
      (fun carrier binding => binding.combined = carrier.take 54)
      carriers bindings := by
  have lanes := shape.lanes
  clear shape
  induction lanes with
  | nil => exact .nil
  | cons head _ inductionHypothesis =>
      exact .cons
        (paddedProjectionLaneShape_activePrefix head)
        inductionHypothesis

theorem paddedProjectionLaneFamily_zeroTails
    {count : Nat}
    {carriers : List (List SuperNeo.F)}
    {bindings : List ProjectionLaneBinding}
    (shape : PaddedProjectionLaneFamilyShape count carriers bindings) :
    carriers.Forall
      (fun carrier =>
        carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F)) := by
  have lanes := shape.lanes
  clear shape
  induction lanes with
  | nil => simp
  | cons head _ inductionHypothesis =>
      simp only [List.forall_cons]
      exact ⟨paddedCarrierShape_zeroTail head.carrier_shape,
        inductionHypothesis⟩

/-! ## Fixed profile -/

/--
All model values needed to state the fixed projection profile. The padded
carriers are separate from, but paired explicitly with, the serialized active
lane material.
-/
structure FixedProjectionProfile where
  outputCount : Nat
  material : ProjectionBindingMaterial
  yRingCarriers : List (List SuperNeo.F)
  yZcolCarriers : List (List SuperNeo.F)

/-- The absent-adv branch has neither leaf digests nor quotient advice. -/
structure NoAdvProjectionShape (material : ProjectionBindingMaterial) : Prop where
  leaves : material.combinedAdvLeaves = []
  quotients : material.advQuotients = []

/-
The all-adv branch binds three four-field leaf digests and three complete sets
of commitment-lane quotients (`ops`, `is`, and `fs`).
-/
structure AllAdvProjectionShape (material : ProjectionBindingMaterial) : Prop where
  leaf_count : material.combinedAdvLeaves.length = 3
  leaves : material.combinedAdvLeaves.Forall (fun leaf => leaf.length = 4)
  quotient_family :
    ProjectionQuotientFamilyShape
      (3 * commitmentLanes) material.advQuotients

/-- Families shared by the plain and counterfactual same-X profiles. -/
structure FixedProjectionFamiliesShape
    (profile : FixedProjectionProfile) : Prop where
  combined_commitment :
    profile.material.combinedCommitment.length =
      SuperNeo.d * commitmentLanes
  commitment_quotients :
    ProjectionQuotientFamilyShape
      commitmentLanes profile.material.commitmentQuotients
  x : ProjectionLaneFamilyShape activeXColumns profile.material.x
  y_ring :
    PaddedProjectionLaneFamilyShape
      (yRingRows * extensionLimbs)
      profile.yRingCarriers profile.material.yRing
  y_zcol :
    PaddedProjectionLaneFamilyShape
      extensionLimbs profile.yZcolCarriers profile.material.yZcol

/-- Plain fixed profile: exactly fifteen Pi_CCS outputs and no adv material. -/
structure PlainFixedProfileShape
    (profile : FixedProjectionProfile) : Prop where
  output_count : profile.outputCount = inputCount
  families : FixedProjectionFamiliesShape profile
  adv : NoAdvProjectionShape profile.material

/-!
Counterfactual diagnostic only: add all adv material while retaining the plain
five-lane X profile. Nebula expands the public input and therefore does not
instantiate this shape.
-/
structure CounterfactualAllAdvSameXProfileShape
    (profile : FixedProjectionProfile) : Prop where
  output_count : profile.outputCount = inputCount
  families : FixedProjectionFamiliesShape profile
  adv : AllAdvProjectionShape profile.material

/-! ## Exact fixed-family consequences -/

theorem plainFixedProfile_outputCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.outputCount = 15 := by
  simpa [inputCount] using shape.output_count

theorem plainFixedProfile_commitmentQuotientCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.material.commitmentQuotients.length = 18 := by
  simpa [commitmentLanes] using
    shape.families.commitment_quotients.count_eq

theorem plainFixedProfile_xCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.material.x.length = 5 := by
  simpa [activeXColumns] using shape.families.x.count_eq

theorem plainFixedProfile_yRingCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.material.yRing.length = 6 := by
  simpa [yRingRows, extensionLimbs] using
    shape.families.y_ring.binding_count_eq

theorem plainFixedProfile_yRingCarrierCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.yRingCarriers.length = 6 := by
  simpa [yRingRows, extensionLimbs] using
    shape.families.y_ring.carrier_count_eq

theorem plainFixedProfile_yZcolCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.material.yZcol.length = 2 := by
  simpa [extensionLimbs] using
    shape.families.y_zcol.binding_count_eq

theorem plainFixedProfile_yZcolCarrierCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.yZcolCarriers.length = 2 := by
  simpa [extensionLimbs] using
    shape.families.y_zcol.carrier_count_eq

theorem plainFixedProfile_yRing_activePrefixes
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    List.Forall₂
      (fun carrier binding => binding.combined = carrier.take 54)
      profile.yRingCarriers profile.material.yRing :=
  paddedProjectionLaneFamily_activePrefixes shape.families.y_ring

theorem plainFixedProfile_yRing_zeroTails
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.yRingCarriers.Forall
      (fun carrier =>
        carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F)) :=
  paddedProjectionLaneFamily_zeroTails shape.families.y_ring

theorem plainFixedProfile_yZcol_activePrefixes
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    List.Forall₂
      (fun carrier binding => binding.combined = carrier.take 54)
      profile.yZcolCarriers profile.material.yZcol :=
  paddedProjectionLaneFamily_activePrefixes shape.families.y_zcol

theorem plainFixedProfile_yZcol_zeroTails
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.yZcolCarriers.Forall
      (fun carrier =>
        carrier.drop 54 = List.replicate 10 (0 : SuperNeo.F)) :=
  paddedProjectionLaneFamily_zeroTails shape.families.y_zcol

theorem plainFixedProfile_projectionLaneCount
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    profile.material.commitmentQuotients.length +
        profile.material.x.length + profile.material.yRing.length +
        profile.material.yZcol.length = 31 := by
  rw [plainFixedProfile_commitmentQuotientCount shape,
    plainFixedProfile_xCount shape,
    plainFixedProfile_yRingCount shape,
    plainFixedProfile_yZcolCount shape]

theorem counterfactualAllAdvSameXProfile_advLeafCount
    {profile : FixedProjectionProfile}
    (shape : CounterfactualAllAdvSameXProfileShape profile) :
    profile.material.combinedAdvLeaves.length = 3 :=
  shape.adv.leaf_count

theorem counterfactualAllAdvSameXProfile_advQuotientCount
    {profile : FixedProjectionProfile}
    (shape : CounterfactualAllAdvSameXProfileShape profile) :
    profile.material.advQuotients.length = 54 := by
  simpa [commitmentLanes] using shape.adv.quotient_family.count_eq

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
