import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge

/-!
Paper-owned logical/public carrier refinement for the fixed-active F' profile.

Protocol: SuperNeo Sections 6--7, especially the public-input branches of
`Pi_RLC` and `Pi_DEC`.
Phase: 257-field external F' input to the complete five-ring paper carrier.
Constraint family: typed semantic decoding only; this file emits no rows.

Owns: the external 257-coordinate input type; the paper `L_in` type containing
all 270 ordered coefficients; an explicit 257-plus-13 coordinate equivalence;
the zero-padding fresh encoder; a fail-closed decoder; exact soundness,
completeness, injectivity, and coordinate coverage; closure of the full paper
carrier under the concrete `Pi_RLC` and `Pi_DEC` operations; and a
sampler-valid counterexample showing that the zero-padded fresh image is not
closed under `Pi_RLC`.

Does not own: Rust/R1CS layout, generated rows, serialized commitments,
Fiat--Shamir/Poseidon2 refinement, PiCCS, NIFS composition, or F' transition
classification.

Authority boundary: the paper carrier is the complete 270-coordinate value.
The last thirteen coordinates are verifier-fixed zeros only while constructing
a fresh input. They remain authoritative running coordinates after folding and
must not be projected away.

| Obligation | Lean owner |
|---|---|
| exact `257 + 13 = 270` ordered ownership | `coordinateEquiv` |
| fresh encoding and fail-closed decoding | `encodeFresh`, `decodeFresh` |
| decoding soundness / encoding completeness | `decodeFresh_sound`, `decodeFresh_complete` |
| no omission, duplication, or aliasing | `coordinateEquiv_bijective` |
| full-carrier `Pi_RLC` / `Pi_DEC` closure | `piRlcCombine`, `piDecSplit`, `piDecRecompose` |
| zero-padded image is not `Pi_RLC`-closed | `freshImage_not_piRlcClosed` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-- The external F' logical input: one constant followed by 256 digest bits. -/
abbrev ExternalInput := Fin legacyPublicWidth -> F

/-- Paper-owned `L_in`: five complete Phi81 public ring columns. -/
abbrev LIn (dimensions : Dimensions) := PublicInput dimensions.shape

/-- Every paper coordinate has exactly one owner: an external coordinate or
one of the thirteen fresh-padding coordinates. -/
abbrev Coordinate := Sum (Fin legacyPublicWidth) (Fin fixedPaddingWidth)

/-- Ordered embedding of the disjoint logical coordinate owners into `L_in`. -/
def carrierColumn (dimensions : Dimensions) : Coordinate ->
    Fin dimensions.shape.publicWidth
  | .inl logical =>
      ⟨logical.val, by
        have bound := logical.isLt
        simp only [Dimensions.shape_publicWidth, legacyPublicWidth] at bound ⊢
        omega⟩
  | .inr padding =>
      ⟨legacyPublicWidth + padding.val, by
        have bound := padding.isLt
        simp only [Dimensions.shape_publicWidth, legacyPublicWidth,
          fixedPaddingWidth] at bound ⊢
        omega⟩

@[simp] theorem carrierColumn_logical_val (dimensions : Dimensions)
    (logical : Fin legacyPublicWidth) :
    (carrierColumn dimensions (.inl logical)).val = logical.val := by
  rfl

@[simp] theorem carrierColumn_padding_val (dimensions : Dimensions)
    (padding : Fin fixedPaddingWidth) :
    (carrierColumn dimensions (.inr padding)).val =
      legacyPublicWidth + padding.val := by
  rfl

/-- Total inverse classifier for one complete paper coordinate. -/
def coordinateOfCarrier (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) : Coordinate :=
  if isLogical : column.val < legacyPublicWidth then
    .inl ⟨column.val, isLogical⟩
  else
    .inr ⟨column.val - legacyPublicWidth, by
      have columnBound := column.isLt
      simp only [Dimensions.shape_publicWidth, legacyPublicWidth,
        fixedPaddingWidth] at columnBound ⊢
      omega⟩

@[simp] theorem coordinateOfCarrier_carrierColumn_logical
    (dimensions : Dimensions) (logical : Fin legacyPublicWidth) :
    coordinateOfCarrier dimensions (carrierColumn dimensions (.inl logical)) =
      .inl logical := by
  simp [coordinateOfCarrier]

@[simp] theorem coordinateOfCarrier_carrierColumn_padding
    (dimensions : Dimensions) (padding : Fin fixedPaddingWidth) :
    coordinateOfCarrier dimensions (carrierColumn dimensions (.inr padding)) =
      .inr padding := by
  have notLogical : ¬ legacyPublicWidth + padding.val < legacyPublicWidth := by
    omega
  simp [coordinateOfCarrier, notLogical]

@[simp] theorem carrierColumn_coordinateOfCarrier
    (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) :
    carrierColumn dimensions (coordinateOfCarrier dimensions column) = column := by
  unfold coordinateOfCarrier
  split
  · apply Fin.ext
    rfl
  · apply Fin.ext
    simp only [carrierColumn_padding_val]
    omega

/-- Explicit two-sided coordinate equivalence. Lean's minimal dependency
surface does not supply a general `Equiv` type, so both functions and both
inverse laws are retained directly. -/
structure CoordinateEquivalence (dimensions : Dimensions) where
  toCarrier : Coordinate -> Fin dimensions.shape.publicWidth
  toCoordinate : Fin dimensions.shape.publicWidth -> Coordinate
  leftInverse : Function.LeftInverse toCoordinate toCarrier
  rightInverse : Function.RightInverse toCoordinate toCarrier

/-- Exact ordering equivalence. It is the no-omission/no-duplication theorem
used by all subsequent encoder/decoder facts. -/
def coordinateEquiv (dimensions : Dimensions) :
    CoordinateEquivalence dimensions where
  toCarrier := carrierColumn dimensions
  toCoordinate := coordinateOfCarrier dimensions
  leftInverse := by
    intro coordinate
    cases coordinate with
    | inl logical => exact coordinateOfCarrier_carrierColumn_logical dimensions logical
    | inr padding => exact coordinateOfCarrier_carrierColumn_padding dimensions padding
  rightInverse := carrierColumn_coordinateOfCarrier dimensions

/-- No two logical owners alias the same paper coordinate. -/
theorem coordinateEquiv_injective (dimensions : Dimensions) :
    Function.Injective (carrierColumn dimensions) :=
  (coordinateEquiv dimensions).leftInverse.injective

/-- Every paper coordinate has an owner. -/
theorem coordinateEquiv_surjective (dimensions : Dimensions) :
    Function.Surjective (carrierColumn dimensions) :=
  (coordinateEquiv dimensions).rightInverse.surjective

/-- The coordinate map is injective and surjective, hence no public
coordinate is omitted, duplicated, or aliased. -/
theorem coordinateEquiv_bijective (dimensions : Dimensions) :
    Function.Injective (carrierColumn dimensions) /\
      Function.Surjective (carrierColumn dimensions) :=
  ⟨coordinateEquiv_injective dimensions,
    coordinateEquiv_surjective dimensions⟩

/-- Read the external 257-coordinate view without granting it authority over
any of the thirteen remaining paper coordinates. -/
def projectExternal (dimensions : Dimensions) (input : LIn dimensions) :
    ExternalInput :=
  fun logical => input (carrierColumn dimensions (.inl logical))

/-- Read one authoritative padding/running coordinate. -/
def paddingValue (dimensions : Dimensions) (input : LIn dimensions)
    (padding : Fin fixedPaddingWidth) : F :=
  input (carrierColumn dimensions (.inr padding))

/-- Executable finite zero check for the thirteen fresh coordinates. -/
def freshCanonicalCheck (dimensions : Dimensions) (input : LIn dimensions) : Bool :=
  (List.ofFn fun padding : Fin fixedPaddingWidth =>
    paddingValue dimensions input padding).all fun value => decide (value = 0)

/-- Fresh inputs alone require the thirteen inserted coordinates to be zero. -/
def FreshCanonical (dimensions : Dimensions) (input : LIn dimensions) : Prop :=
  freshCanonicalCheck dimensions input = true

instance freshCanonicalDecidable (dimensions : Dimensions)
    (input : LIn dimensions) : Decidable (FreshCanonical dimensions input) := by
  unfold FreshCanonical
  exact Bool.decEq _ _

/-- The executable check has exactly the intended coordinatewise meaning. -/
theorem freshCanonical_iff (dimensions : Dimensions) (input : LIn dimensions) :
    FreshCanonical dimensions input <->
      forall padding, paddingValue dimensions input padding = 0 := by
  unfold FreshCanonical freshCanonicalCheck
  rw [List.all_eq_true]
  constructor
  · intro checked padding
    have member : paddingValue dimensions input padding ∈
        List.ofFn fun item : Fin fixedPaddingWidth =>
          paddingValue dimensions input item := by
      exact List.mem_ofFn.mpr ⟨padding, rfl⟩
    exact of_decide_eq_true (checked _ member)
  · intro allZero value member
    obtain ⟨padding, rfl⟩ := List.mem_ofFn.mp member
    exact decide_eq_true (allZero padding)

/-- The external public prefix extracted from the existing typed legacy
assignment boundary. Private legacy coordinates are not part of `L_in`. -/
def externalOfLegacy (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : ExternalInput :=
  fun logical => legacy ⟨logical.val,
    Nat.lt_of_lt_of_le logical.isLt dimensions.legacyPublicFits⟩

/-- Canonical fresh encoding. Coordinate ownership is defined through the
explicit inverse classifier, not a row artifact or truncating list read. -/
def encodeFresh (dimensions : Dimensions) (external : ExternalInput) :
    LIn dimensions :=
  fun column =>
    match coordinateOfCarrier dimensions column with
    | .inl logical => external logical
    | .inr _ => 0

@[simp] theorem encodeFresh_logical (dimensions : Dimensions)
    (external : ExternalInput) (logical : Fin legacyPublicWidth) :
    encodeFresh dimensions external (carrierColumn dimensions (.inl logical)) =
      external logical := by
  simp [encodeFresh]

@[simp] theorem encodeFresh_padding (dimensions : Dimensions)
    (external : ExternalInput) (padding : Fin fixedPaddingWidth) :
    encodeFresh dimensions external (carrierColumn dimensions (.inr padding)) =
      0 := by
  simp [encodeFresh]

/-- Alignment with the repository's existing typed fresh-assignment public
projection. This theorem depends only on the semantic assignment constructor,
not generated rows or a captured checker artifact. -/
theorem encodeFresh_externalOfLegacy_eq_expectedPublicInput
    (dimensions : Dimensions) (legacy : LegacyAssignment dimensions) :
    encodeFresh dimensions (externalOfLegacy dimensions legacy) =
      expectedPublicInput dimensions legacy := by
  funext column
  by_cases isLegacy : column.val < legacyPublicWidth
  · simp [encodeFresh, coordinateOfCarrier, expectedPublicInput,
      externalOfLegacy, isLegacy]
  · simp [encodeFresh, coordinateOfCarrier, expectedPublicInput,
      isLegacy]

@[simp] theorem projectExternal_encodeFresh (dimensions : Dimensions)
    (external : ExternalInput) :
    projectExternal dimensions (encodeFresh dimensions external) = external := by
  funext logical
  exact encodeFresh_logical dimensions external logical

theorem encodeFresh_freshCanonical (dimensions : Dimensions)
    (external : ExternalInput) :
    FreshCanonical dimensions (encodeFresh dimensions external) := by
  exact (freshCanonical_iff dimensions
    (encodeFresh dimensions external)).2 fun padding =>
      encodeFresh_padding dimensions external padding

/-- A canonical paper carrier is exactly reconstructed from its external
view. -/
theorem encodeFresh_projectExternal_of_canonical (dimensions : Dimensions)
    (input : LIn dimensions) (canonical : FreshCanonical dimensions input) :
    encodeFresh dimensions (projectExternal dimensions input) = input := by
  funext column
  have cover := carrierColumn_coordinateOfCarrier dimensions column
  cases owner : coordinateOfCarrier dimensions column with
  | inl logical =>
      have columnEq : carrierColumn dimensions (.inl logical) = column := by
        simpa [owner] using cover
      subst column
      simp [projectExternal]
  | inr padding =>
      have columnEq : carrierColumn dimensions (.inr padding) = column := by
        simpa [owner] using cover
      subst column
      rw [encodeFresh_padding]
      exact ((freshCanonical_iff dimensions input).1 canonical padding).symm

/-- Fail-closed production carrier decoder. Nonzero fresh padding returns
`none`; the digest or first 257 coordinates alone never authorize a decode. -/
def decodeFresh (dimensions : Dimensions) (input : LIn dimensions) :
    Option ExternalInput :=
  if FreshCanonical dimensions input then
    some (projectExternal dimensions input)
  else
    none

/-- Decoding soundness and canonicality. -/
theorem decodeFresh_sound (dimensions : Dimensions)
    (input : LIn dimensions) (external : ExternalInput)
    (decoded : decodeFresh dimensions input = some external) :
    input = encodeFresh dimensions external := by
  by_cases canonical : FreshCanonical dimensions input
  · have projected : projectExternal dimensions input = external := by
      simpa [decodeFresh, canonical] using decoded
    rw [← projected]
    exact (encodeFresh_projectExternal_of_canonical dimensions input canonical).symm
  · simp [decodeFresh, canonical] at decoded

/-- Honest encoding completeness. -/
@[simp] theorem decodeFresh_complete (dimensions : Dimensions)
    (external : ExternalInput) :
    decodeFresh dimensions (encodeFresh dimensions external) = some external := by
  simp [decodeFresh, encodeFresh_freshCanonical]

/-- Canonical accepted carriers are unique. -/
theorem encodeFresh_injective (dimensions : Dimensions) :
    Function.Injective (encodeFresh dimensions) := by
  intro left right equal
  have projected := congrArg (projectExternal dimensions) equal
  simpa using projected

/-- Any accepted decoder result proves all thirteen verifier-owned zeros. -/
theorem decodeFresh_implies_canonical (dimensions : Dimensions)
    (input : LIn dimensions) (external : ExternalInput)
    (decoded : decodeFresh dimensions input = some external) :
    FreshCanonical dimensions input := by
  by_cases canonical : FreshCanonical dimensions input
  · exact canonical
  · simp [decodeFresh, canonical] at decoded

/-- A nonzero padding coordinate is rejected, including a self-consistent
257-coordinate prefix. -/
theorem decodeFresh_rejects_nonzero_padding (dimensions : Dimensions)
    (input : LIn dimensions) (padding : Fin fixedPaddingWidth)
    (nonzero : paddingValue dimensions input padding ≠ 0) :
    decodeFresh dimensions input = none := by
  have notCanonical : ¬ FreshCanonical dimensions input := by
    intro canonical
    exact nonzero ((freshCanonical_iff dimensions input).1 canonical padding)
  simp [decodeFresh, notCanonical]

/-- All-zero complete paper carrier. -/
def zeroLIn (dimensions : Dimensions) : LIn dimensions := fun _ => 0

/-- First of the thirteen fresh-padding coordinates. -/
def firstPadding : Fin fixedPaddingWidth := ⟨0, by decide⟩

/-- A carrier differing only at the first of the thirteen authoritative tail
coordinates. -/
def firstPaddingOne (dimensions : Dimensions) : LIn dimensions :=
  fun column =>
    if column = carrierColumn dimensions (.inr firstPadding) then 1 else 0

/-- The first-tail mutation is invisible to the legacy 257-coordinate view. -/
theorem projectExternal_firstPaddingOne_eq_zero (dimensions : Dimensions) :
    projectExternal dimensions (firstPaddingOne dimensions) =
      projectExternal dimensions (zeroLIn dimensions) := by
  funext logical
  have distinct :
      carrierColumn dimensions (.inl logical) ≠
        carrierColumn dimensions (.inr firstPadding) := by
    intro equal
    have impossible : (Sum.inl logical : Coordinate) = Sum.inr firstPadding :=
      coordinateEquiv_injective dimensions equal
    cases impossible
  simp [projectExternal, zeroLIn, firstPaddingOne, distinct]

/-- The invisible first-tail mutation is nevertheless a different complete
paper input. -/
theorem firstPaddingOne_ne_zero (dimensions : Dimensions) :
    firstPaddingOne dimensions ≠ zeroLIn dimensions := by
  intro equal
  have atPadding := congrFun equal
    (carrierColumn dimensions (.inr firstPadding))
  have oneEqZero : (1 : F) = 0 := by
    simpa [firstPaddingOne, zeroLIn] using atPadding
  exact (by decide : (1 : F) ≠ 0) oneEqZero

/-- The 257-coordinate projection is not injective on the paper carrier. -/
theorem projectExternal_not_injective (dimensions : Dimensions) :
    ¬ Function.Injective (projectExternal dimensions) := by
  intro injective
  exact firstPaddingOne_ne_zero dimensions
    (injective (projectExternal_firstPaddingOne_eq_zero dimensions))

/-- Full paper `L_in` is closed by construction under the exact concrete
`Pi_RLC` finite combination. -/
def piRlcCombine (dimensions : Dimensions) {count : Nat}
    (challenges : Fin count -> RingF)
    (inputs : Fin count -> LIn dimensions) : LIn dimensions :=
  PiRLCAlgebra.PublicInput.combinePublicInputs challenges inputs

/-- Full paper `L_in` is closed by construction under verifier-owned
coordinatewise `Pi_DEC` splitting. -/
def piDecSplit (dimensions : Dimensions) (input : LIn dimensions)
    (child : PiDECAlgebra.Radix.ChildIndex) : LIn dimensions :=
  PiDECAlgebra.PublicInput.splitPublicInput input child

/-- Full paper `L_in` is closed by construction under exact `Pi_DEC`
recomposition. -/
def piDecRecompose (dimensions : Dimensions)
    (children : PiDECAlgebra.Radix.ChildIndex -> LIn dimensions) :
    LIn dimensions :=
  PiDECAlgebra.PublicInput.recomposePublicInput children

@[simp] theorem piDecSplit_recompose (dimensions : Dimensions)
    (input : LIn dimensions) :
    piDecRecompose dimensions (piDecSplit dimensions input) = input :=
  PiDECAlgebra.PublicInput.splitPublicInput_recompose input

/-- Sampler scalar encoding the valid challenge `X`: centered `+1` at
coefficient one and zero at every other coefficient. -/
def shiftScalar : Scalar :=
  fun position =>
    if position.val = 1 then ⟨3, by decide⟩ else ⟨2, by decide⟩

/-- Concrete verifier-valid `Pi_RLC` shift challenge. -/
def shiftChallenge : RingF := embedScalar shiftScalar

theorem shiftChallenge_valid :
    PiRLCAlgebra.Challenge.challengeValid shiftChallenge :=
  PiRLCAlgebra.Challenge.embedScalar_valid shiftScalar

theorem embedCoefficient_zero_symbol :
    embedCoefficient ⟨2, by decide⟩ = (0 : F) := by
  decide

theorem embedCoefficient_one_symbol :
    embedCoefficient ⟨3, by decide⟩ = (1 : F) := by
  decide

theorem shiftChallenge_eq_monomial :
    shiftChallenge = ringFMonomial 1 1 := by
  funext position
  by_cases atOne : position.val = 1
  · simp [shiftChallenge, embedScalar, shiftScalar, scalarPosition,
      ringFMonomial, atOne, embedCoefficient_one_symbol]
  · simp [shiftChallenge, embedScalar, shiftScalar, scalarPosition,
      ringFMonomial, atOne, embedCoefficient_zero_symbol]

/-- External input whose final coordinate is one and all preceding
coordinates are zero. -/
def finalExternalOne : ExternalInput :=
  fun logical => if logical.val = legacyPublicWidth - 1 then 1 else 0

/-- First inserted tail coordinate in the complete paper public input. -/
def firstPaddingColumn (dimensions : Dimensions) :
    Fin dimensions.shape.publicWidth :=
  carrierColumn dimensions (.inr firstPadding)

@[simp] theorem firstPaddingColumn_val (dimensions : Dimensions) :
    (firstPaddingColumn dimensions).val = 257 := by
  change legacyPublicWidth + firstPadding.val = 257
  rfl

/-- Fifth complete public ring column. -/
def fifthPublicBlock (dimensions : Dimensions) :
    Fin dimensions.shape.publicRingColumns :=
  ⟨4, by
    change 4 < 5
    decide⟩

/-- First lane beyond the 257-coordinate external prefix. -/
def firstPaddingLane : Fin ringDegree := ⟨41, by decide⟩

/-- The fifth public ring block of the boundary input is exactly `X^40`. -/
theorem finalExternalOne_fifthBlock (dimensions : Dimensions) :
    PiRLCAlgebra.PublicInput.publicBlock
        (encodeFresh dimensions finalExternalOne)
        (fifthPublicBlock dimensions) =
      ringFMonomial 40 1 := by
  funext lane
  unfold PiRLCAlgebra.PublicInput.publicBlock
  by_cases atForty : lane.val = 40
  · let logical : Fin legacyPublicWidth := ⟨256, by decide⟩
    have columnEq :
        (⟨(fifthPublicBlock dimensions).val * ringDegree + lane.val, by
          have laneLt := lane.isLt
          change 4 * 54 + lane.val < 270
          omega⟩ : Fin dimensions.shape.publicWidth) =
          carrierColumn dimensions (.inl logical) := by
      apply Fin.ext
      simp [fifthPublicBlock, logical, ringDegree, atForty]
    rw [columnEq, encodeFresh_logical]
    simp [logical, finalExternalOne, legacyPublicWidth, ringFMonomial, atForty]
  · by_cases inExternal : lane.val < 41
    · have sourceLogical : 4 * ringDegree + lane.val < legacyPublicWidth := by
        simp only [ringDegree, legacyPublicWidth]
        omega
      let logical : Fin legacyPublicWidth :=
        ⟨4 * ringDegree + lane.val, sourceLogical⟩
      have columnEq :
          (⟨(fifthPublicBlock dimensions).val * ringDegree + lane.val, by
            have laneLt := lane.isLt
            change 4 * 54 + lane.val < 270
            omega⟩ : Fin dimensions.shape.publicWidth) =
            carrierColumn dimensions (.inl logical) := by
        apply Fin.ext
        simp [fifthPublicBlock, logical]
      have notFinal : logical.val ≠ legacyPublicWidth - 1 := by
        simp only [logical, ringDegree, legacyPublicWidth]
        omega
      rw [columnEq, encodeFresh_logical]
      simp [finalExternalOne, logical, notFinal, ringFMonomial, atForty]
    · have laneAtLeast : 41 <= lane.val := Nat.le_of_not_gt inExternal
      have laneLt54 : lane.val < 54 := by
        simpa only [ringDegree] using lane.isLt
      have offsetLt13 : lane.val - 41 < 13 := by
        omega
      let padding : Fin fixedPaddingWidth :=
        ⟨lane.val - 41, by
          simpa only [fixedPaddingWidth] using offsetLt13⟩
      have columnEq :
          (⟨(fifthPublicBlock dimensions).val * ringDegree + lane.val, by
            rw [Dimensions.shape_publicWidth]
            change 4 * 54 + lane.val < 270
            omega⟩ : Fin dimensions.shape.publicWidth) =
            carrierColumn dimensions (.inr padding) := by
        apply Fin.ext
        change 4 * 54 + lane.val = 257 + (lane.val - 41)
        omega
      rw [columnEq, encodeFresh_padding]
      simp [ringFMonomial, atForty]

theorem ringFMul_shift40_at41 :
    ringFMul (ringFMonomial 1 1) (ringFMonomial 40 1)
      ⟨41, by decide⟩ = 1 := by
  have product :=
    EvaluationHomomorphism.RingFLaws.ringFMul_basis_basis
      (⟨1, by decide⟩ : Fin ringDegree)
      (⟨40, by decide⟩ : Fin ringDegree)
  have coefficient := congrFun product (⟨41, by decide⟩ : Fin ringDegree)
  simpa [EvaluationHomomorphism.RingFLaws.basis,
    EvaluationHomomorphism.RingFLaws.monomialReduce, ringDegree,
    ringFMonomial] using coefficient

@[simp] theorem firstPadding_blockIndex (dimensions : Dimensions) :
    PiRLCAlgebra.PublicInput.publicBlockIndex dimensions.shape
        (firstPaddingColumn dimensions) = fifthPublicBlock dimensions := by
  apply Fin.ext
  change (firstPaddingColumn dimensions).val / ringDegree = 4
  rw [firstPaddingColumn_val]
  decide

@[simp] theorem firstPadding_laneIndex (dimensions : Dimensions) :
    PiRLCAlgebra.PublicInput.publicLaneIndex
        (firstPaddingColumn dimensions) = firstPaddingLane := by
  apply Fin.ext
  change (firstPaddingColumn dimensions).val % ringDegree = 41
  rw [firstPaddingColumn_val]
  decide

/-- A verifier-valid `Pi_RLC` action moves external coordinate 256 into paper
coordinate 257. Thus fresh zero padding is initialization data, not a
permanently inert subcarrier. -/
theorem shift_enters_first_padding (dimensions : Dimensions) :
    PiRLCAlgebra.PublicInput.publicAct shiftChallenge
        (encodeFresh dimensions finalExternalOne)
        (firstPaddingColumn dimensions) = 1 := by
  unfold PiRLCAlgebra.PublicInput.publicAct
  rw [shiftChallenge_eq_monomial, firstPadding_blockIndex,
    firstPadding_laneIndex, finalExternalOne_fifthBlock]
  exact ringFMul_shift40_at41

/-- The concrete zero-padded fresh image is not closed under the exact
sampler-valid `Pi_RLC` operation used by the selected profile. -/
theorem freshImage_not_piRlcClosed (dimensions : Dimensions) :
    ¬ FreshCanonical dimensions
      (PiRLCAlgebra.PublicInput.publicAct shiftChallenge
        (encodeFresh dimensions finalExternalOne)) := by
  intro canonical
  have zero := (freshCanonical_iff dimensions _).1 canonical firstPadding
  have one := shift_enters_first_padding dimensions
  exact (by decide : (1 : F) ≠ 0) (one.symm.trans zero)

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
