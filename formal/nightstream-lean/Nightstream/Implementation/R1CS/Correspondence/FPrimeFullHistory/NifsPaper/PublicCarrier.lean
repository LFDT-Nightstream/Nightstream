import Nightstream.SuperNeo.Concrete.Relation
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Folding.PiDEC
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPiDecArtifact

/-!
Canonical public carrier for refining the fixed F' NIFS implementation into
the independent SuperNeo reduction model.

Owns: fixed-profile decoding of the verifier-visible commitment, packed public
`X`, paper CE point `r`, and the three active ring evaluations; mathematical
radix recomposition on those carriers; and kernel-checked shape facts.

Does not own: a private CE witness, `Concrete.relationSemantics`, `CE.Holds`,
knowledge extraction, Ajtai/MSIS binding, transcript/NC sidecars, or row
removal. In particular, no fake private `RelationSemantics` is defined here.

Emits constraints: no.

Authority boundary: every decoded value comes from an explicit assignment;
the module neither validates that assignment nor promotes a digest to
authority.

| Carrier | Production columns | Mathematical content | Excluded sidecars/padding |
|---|---|---|---|
| `PackedCommitment` | `commitment.dataCols` | verifier-visible Ajtai payload, flat production order | `d`, `kappa` shape headers |
| `PackedPublicInput` | all 270 `xActiveCols` | 54×5 ring-packed public carrier used by Π_RLC/Π_DEC | inactive `X` columns |
| `Point` | `rCols` | paper CE evaluation point in `K` | `sColCols`, `foldDigestCols` |
| `Evaluation` | first 108 limbs of each `yRingCols` row | one active `RingK = Fin 54 → K` | 20 zero-padding limbs per row |
| `Array Evaluation` | three `yRingCols` rows | the three paper CE evaluations | none |

`PackedPublicInput` is intentionally not called `Concrete.PublicInput`.
Production folds all 270 coefficients, while the current Concrete model uses
`List F`/`take 257`. `PublicInputBoundary.lean` records the resulting failed
paper dimension precondition and non-injective projection; this module does
not silently truncate the 13 non-image packed coefficients.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec

abbrev Scalar := Concrete.F
abbrev Point := Concrete.Point
abbrev Evaluation := Concrete.Evaluation

structure PackedCommitment where
  data : List Scalar
deriving DecidableEq

theorem PackedCommitment.eq_of_data_eq (left right : PackedCommitment)
    (data : left.data = right.data) : left = right := by
  cases left
  cases right
  cases data
  rfl

structure PackedPublicInput where
  data : List Scalar
deriving DecidableEq

theorem PackedPublicInput.eq_of_data_eq (left right : PackedPublicInput)
    (data : left.data = right.data) : left = right := by
  cases left
  cases right
  cases data
  rfl

def residue (value : Nat) : Scalar :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

def values (assignment : Nat → Nat) (columns : List Nat) : List Scalar :=
  columns.map (fun column => residue (assignment column))

/-- Base-field pairs used by the implementation sidecar refinements. -/
def pairValues (assignment : Nat → Nat)
    (columns : List (Nat × Nat)) : List (Scalar × Scalar) :=
  columns.map fun pair =>
    (residue (assignment pair.1), residue (assignment pair.2))

def extensionValue (assignment : Nat → Nat) (columns : Nat × Nat) : Concrete.K :=
  ⟨residue (assignment columns.1), residue (assignment columns.2)⟩

def extensionValues (assignment : Nat → Nat)
    (columns : List (Nat × Nat)) : List Concrete.K :=
  columns.map (extensionValue assignment)

theorem k_eq_of_coeffs (left right : Concrete.K)
    (c0 : left.c0 = right.c0) (c1 : left.c1 = right.c1) : left = right := by
  cases left
  cases right
  cases c0
  cases c1
  rfl

def decodedPackedCommitment (assignment : Nat → Nat)
    (claim : ClaimLayout) : PackedCommitment :=
  ⟨values assignment claim.commitment.dataCols⟩

def decodedPackedInput (assignment : Nat → Nat)
    (claim : ClaimLayout) : PackedPublicInput :=
  ⟨values assignment claim.xActiveCols⟩

def decodedPoint (assignment : Nat → Nat)
    (claim : ClaimLayout) : Point :=
  extensionValues assignment claim.rCols

/-- Decode only the active 54 `K` coefficients. The remaining 20 base-field
limbs in the fixed 128-limb row are padding, not a paper evaluation. -/
def decodedEvaluation (assignment : Nat → Nat)
    (row : List Nat) : Evaluation :=
  fun coefficient => ⟨
    residue (assignment (row.getD (2 * coefficient.val) 0)),
    residue (assignment (row.getD (2 * coefficient.val + 1) 0))⟩

/-- The fixed profile has three independent paper evaluations. -/
def decodedEvaluations (assignment : Nat → Nat)
    (claim : ClaimLayout) : Array Evaluation :=
  (claim.yRingCols.map (decodedEvaluation assignment)).toArray

universe uStructure

def decodedInstance {Structure : Type uStructure} (system : Structure)
    (assignment : Nat → Nat) (claim : ClaimLayout) (stage : NormStage) :
    CE.Instance Structure PackedPublicInput Point Evaluation PackedCommitment where
  constraintSystem := system
  commitment := decodedPackedCommitment assignment claim
  publicInput := decodedPackedInput assignment claim
  point := decodedPoint assignment claim
  evaluations := decodedEvaluations assignment claim
  stage := stage

theorem production_child_count :
    Concrete.productionGlobalParams.k = layout.children.length := by
  decide

/-- Kernel-checked fixed-profile facts used by every phase refinement. -/
theorem production_profile :
    layout.radix = 2 ∧ layout.parent.adv = none ∧
      layout.ringDimension = Concrete.ringDegree ∧
      layout.extensionLimbs = 2 ∧
      ∀ child ∈ layout.children, child.adv = none := by
  decide

/-- Kernel-reduced fixed-layout facts needed by the public paper bridge. This
is separate from the strict compiler's `native_decide` shape certificate. -/
structure PublicShapeFacts : Prop where
  commitmentLengths : ∀ child ∈ layout.children,
    child.commitment.dataCols.length = layout.parent.commitment.dataCols.length
  xLengths : ∀ child ∈ layout.children,
    child.xActiveCols.length = layout.parent.xActiveCols.length
  yShapes : ∀ child ∈ layout.children,
    child.yRingCols.length = layout.parent.yRingCols.length ∧
      ∀ row, row < layout.parent.yRingCols.length →
        (child.yRingCols.getD row []).length =
          (layout.parent.yRingCols.getD row []).length
  activeEvaluationRows : ∀ claim ∈ layout.parent :: layout.children,
    ∀ row ∈ claim.yRingCols,
      layout.ringDimension * layout.extensionLimbs ≤ row.length
  rShapes : ∀ child ∈ layout.children,
    child.rCols.length = layout.parent.rCols.length

set_option maxRecDepth 524288 in
theorem production_public_shape : PublicShapeFacts := by
  constructor <;> decide

def childLayout (index : Fin Concrete.productionGlobalParams.k) : ClaimLayout :=
  layout.children.get (Fin.cast production_child_count index)

def firstIndex : Fin Concrete.productionGlobalParams.k :=
  ⟨0, by decide⟩

/-- Field interpretation of verifier-owned radix powers in child order. -/
def radixWeights : List Scalar :=
  (List.range Concrete.productionGlobalParams.k).map fun exponent =>
    residue (layout.radix ^ exponent % goldilocksP)

def combineScalar
    (items : Fin Concrete.productionGlobalParams.k → Scalar) : Scalar :=
  ((List.ofFn items).zip radixWeights).foldr
    (fun pair suffix => pair.2 * pair.1 + suffix) 0

def combineList
    (items : Fin Concrete.productionGlobalParams.k → List Scalar) : List Scalar :=
  (List.range (items firstIndex).length).map fun lane =>
    combineScalar fun index => (items index).getD lane 0

def combinePackedCommitment
    (items : Fin Concrete.productionGlobalParams.k → PackedCommitment) :
    PackedCommitment :=
  ⟨combineList fun index => (items index).data⟩

def combinePackedPublicInput
    (items : Fin Concrete.productionGlobalParams.k → PackedPublicInput) :
    PackedPublicInput :=
  ⟨combineList fun index => (items index).data⟩

def combineK
    (items : Fin Concrete.productionGlobalParams.k → Concrete.K) : Concrete.K :=
  ⟨combineScalar fun index => (items index).c0,
   combineScalar fun index => (items index).c1⟩

def combineEvaluation
    (items : Fin Concrete.productionGlobalParams.k → Evaluation) : Evaluation :=
  fun coefficient => combineK fun index => items index coefficient

def combineEvaluations
    (items : Fin Concrete.productionGlobalParams.k → Array Evaluation) :
    Array Evaluation :=
  ((List.range (items firstIndex).size).map fun row =>
    combineEvaluation fun index =>
      (items index).getD row Concrete.ringKZero).toArray

/-- Canonical logical-order readout `c ↦ X[c mod 54, c / 54]`. This is only
the current 270-to-257 projection used to state the public-input mismatch; it
is not a refinement theorem and is not used by Π_DEC acceptance. -/
def unpackPublicInput (packed : PackedPublicInput) : Concrete.PublicInput :=
  (List.range layout.parent.mIn).map fun column =>
    packed.data.getD
      ((column % layout.ringDimension) * activeColumns layout +
        column / layout.ringDimension) 0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
