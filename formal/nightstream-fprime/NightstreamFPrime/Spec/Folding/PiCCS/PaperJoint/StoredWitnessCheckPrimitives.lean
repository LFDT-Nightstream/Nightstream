import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork

/-!
Concrete scalar operations for the stored-witness checker. Counts use the
existing abstract operation model: one fixed-size Goldilocks operation,
comparison, record read, branch, or constructor is one operation. They are
not wall-clock bounds. No function-valued key, matrix, or probe access is
assigned a cost here. The norm check uses the selected Goldilocks ambient
bound. `withScalarChecks` replaces six fields of the existing program.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckPrimitives

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier StoredWitnessProjection
open StoredWitnessCheckWork (Program PrimitiveBounds Correct Bounded)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- One field multiplication, one addition, and one result. -/
def dotStep (accumulated matrix value : F) : Result F :=
  ⟨accumulated + matrix * value, 1 + 1 + 1⟩

theorem dotStep_value (accumulated matrix value : F) :
    (dotStep accumulated matrix value).value = baseOps.add accumulated (baseOps.mul matrix value) := rfl

theorem dotStep_work (accumulated matrix value : F) :
    (dotStep accumulated matrix value).work = 3 := rfl

/-- Construct the two-field extension value and the result. -/
def embed (value : F) : Result K := ⟨⟨value, 0⟩, 1 + 1⟩

theorem embed_value (value : F) : (embed value).value = K.embed value := rfl
theorem embed_work (value : F) : (embed value).work = 2 := rfl

/-- Read six coefficients once, then execute two subtractions, five
multiplications, four additions, and the two output constructors. -/
def interpolate (low high point : K) : Result K :=
  let low0 := low.c0
  let low1 := low.c1
  let high0 := high.c0
  let high1 := high.c1
  let point0 := point.c0
  let point1 := point.c1
  let difference0 := high0 - low0
  let difference1 := high1 - low1
  let output0 := low0 + (point0 * difference0 + 7 * point1 * difference1)
  let output1 := low1 + (point0 * difference1 + point1 * difference0)
  ⟨⟨output0, output1⟩, 6 + 2 + 5 + 4 + 1 + 1⟩

theorem interpolate_value (low high point : K) :
    (interpolate low high point).value =
      extensionOps.add low (extensionOps.mul point (extensionOps.sub high low)) := by
  rw [derived_sub_eq_concrete_sub]
  rfl

theorem interpolate_work (low high point : K) : (interpolate low high point).work = 19 := rfl

/-- Compare canonical field words and return the result. -/
def equalF (left right : F) : Result Bool := ⟨decide (left = right), 1 + 1⟩

theorem equalF_value (left right : F) : (equalF left right).value = decide (left = right) := rfl
theorem equalF_work (left right : F) : (equalF left right).work = 2 := rfl

/-- The second coefficient is read and compared only if the first matches.
Each path counts its actual reads, scalar comparison calls, branch and return. -/
def equalK (left right : K) : Result Bool :=
  let first := equalF left.c0 right.c0
  if first.value then
    let second := equalF left.c1 right.c1
    ⟨second.value, 2 + first.work + 2 + second.work + 1 + 1⟩
  else ⟨false, 2 + first.work + 1 + 1⟩

theorem equalK_value (left right : K) : (equalK left right).value = decide (left = right) := by
  cases left with
  | mk left0 left1 =>
      cases right with
      | mk right0 right1 =>
          by_cases same : left0 = right0 <;> simp [equalK, equalF, same, K.mk.injEq]

theorem equalK_work (left right : K) :
    (equalK left right).work = if left.c0 = right.c0 then 10 else 6 := by
  by_cases same : left.c0 = right.c0 <;> simp [equalK, equalF, same]

theorem equalK_work_le (left right : K) : (equalK left right).work ≤ 10 := by
  rw [equalK_work]
  split <;> omega

/-- Read the residue, subtract it from q, compare and select the centered
value, then compare with the selected ambient bound and return. -/
def norm (value : F) : Result Bool :=
  let residue := value.val
  let complement := goldilocksModulus - residue
  if residue ≤ complement then
    ⟨decide (residue < productionGlobalParams.ambientBound), 1 + 1 + 1 + 1 + 1 + 1⟩
  else ⟨decide (complement < productionGlobalParams.ambientBound), 1 + 1 + 1 + 1 + 1 + 1⟩

theorem norm_value (value : F) :
    (norm value).value = decide (centeredMagnitude value < productionGlobalParams.ambientBound) := by
  by_cases low : value.val ≤ goldilocksModulus - value.val
  · simp [norm, centeredMagnitude, low, min_eq_left low]
  · have high : goldilocksModulus - value.val ≤ value.val := Nat.le_of_lt (Nat.lt_of_not_ge low)
    simp [norm, centeredMagnitude, low, min_eq_right high]

theorem norm_work (value : F) : (norm value).work = 6 := by
  dsimp only [norm]
  split <;> rfl

/-- Keep the existing access interface and replace all scalar operations. -/
def withScalarChecks {shape : Shape} {carrier : Phi81Relation.Shape}
    (program : Program shape carrier) : Program shape carrier :=
  { program with dotStep, embed, interpolate, equalF, equalK, norm }

/-- Exact fixed scalar counts; the extension equality bound covers both paths. -/
def withScalarBounds (bounds : PrimitiveBounds) : PrimitiveBounds :=
  { bounds with dotStep := 3, embed := 2, interpolate := 19, equalF := 2, equalK := 10, norm := 6 }

variable {shape : Shape} {carrier : Phi81Relation.Shape}

/-- Only the seven access/gate refinements remain. No prior arithmetic
correctness of `program` is required for the updated program. -/
theorem withScalarChecks_correct {Commitment : Type*} [DecidableEq Commitment]
    {blockCount width : Nat} (program : Program shape carrier)
    (commit : Phi81Relation.Assignment carrier → Commitment)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (publicCheck : ∀ probe, (program.publicCheck probe).value =
      ProtocolPolynomial.FixedWidth.check extensionOps width (statement.verifierInput K.embed)
        probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
        (statement.projectOutput probe.view.response.fullOutput) probe.certificate)
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).value =
      decide (commit (stored.get source).get = statement.commitments source))
    (publicInput : ∀ source column, (program.publicInput source column).value =
      statement.publicInputs source column)
    (padEntry : ∀ coefficient vertex column, (program.padEntry coefficient vertex column).value =
      statement.matrixSource.coefficientMatrixOf baseOps
        (fun row column => statement.cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient vertex column)
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).value =
        statement.matrixSource.coefficientMatrix baseOps matrix coefficient vertex column)
    (padClaim : ∀ probe source coefficient, (program.padClaim probe source coefficient).value =
      probe.view.response.fullOutput.padCoordinate source coefficient)
    (matrixClaim : ∀ probe source matrix coefficient,
      (program.matrixClaim probe source matrix coefficient).value =
        probe.view.response.fullOutput.matrixCoordinate source matrix coefficient) :
    Correct (withScalarChecks program) commit productionGlobalParams statement width where
  publicCheck := publicCheck
  commitmentCheck := commitmentCheck
  publicInput := publicInput
  padEntry := padEntry
  matrixEntry := matrixEntry
  padClaim := padClaim
  matrixClaim := matrixClaim
  dotStep := dotStep_value
  embed := embed_value
  interpolate := interpolate_value
  equalF := equalF_value
  equalK := equalK_value
  norm := norm_value

/-- Only the untouched calls need work bounds. The six scalar bounds follow
from their concrete returned counts. -/
theorem withScalarChecks_bounded (program : Program shape carrier) (bounds : PrimitiveBounds)
    (publicCheck : ∀ probe, (program.publicCheck probe).work ≤ bounds.publicCheck)
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).work ≤ bounds.commitmentCheck)
    (publicInput : ∀ source column, (program.publicInput source column).work ≤ bounds.publicInput)
    (padEntry : ∀ coefficient vertex column,
      (program.padEntry coefficient vertex column).work ≤ bounds.padEntry)
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).work ≤ bounds.matrixEntry)
    (padClaim : ∀ probe source coefficient,
      (program.padClaim probe source coefficient).work ≤ bounds.padClaim)
    (matrixClaim : ∀ probe source matrix coefficient,
      (program.matrixClaim probe source matrix coefficient).work ≤ bounds.matrixClaim) :
    Bounded (withScalarChecks program) (withScalarBounds bounds) where
  publicCheck := publicCheck
  commitmentCheck := commitmentCheck
  publicInput := publicInput
  padEntry := padEntry
  matrixEntry := matrixEntry
  padClaim := padClaim
  matrixClaim := matrixClaim
  dotStep := fun accumulated matrix value => le_of_eq (dotStep_work accumulated matrix value)
  embed := fun value => le_of_eq (embed_work value)
  interpolate := fun low high point => le_of_eq (interpolate_work low high point)
  equalF := fun left right => le_of_eq (equalF_work left right)
  equalK := equalK_work_le
  norm := fun value => le_of_eq (norm_work value)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckPrimitives
