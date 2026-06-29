import Mathlib.FieldTheory.Finite.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Nat.Bitwise

/-!
# Opening Convergence: Shared Definitions

Core types and parameters for the v1 opening convergence formalization.
All definitions are parameterized over an abstract field K satisfying
the minimum cardinality bound. Instantiation to concrete Goldilocks
extension happens only in implementation files when needed for
executable checks.
-/

namespace OpeningConvergence

/-! ## Field Parameters -/

/-- Ring dimension for the Ajtai commitment scheme. -/
def AJTAI_D : Nat := 54

/-- Minimum field cardinality required for soundness (2^128). -/
def MIN_FIELD_CARD : Nat := 2 ^ 128

/-! ## Core Types -/

/-- A packed column evaluation: the D K-valued coefficients of an R_K element
    in canonical increasing-power order. This is what the Ajtai PCS opens. -/
structure PackedColumnEval (K : Type*) where
  coeffs : Fin AJTAI_D → K

@[ext] theorem PackedColumnEval.ext
    {K : Type*} {x y : PackedColumnEval K}
    (h : x.coeffs = y.coeffs) : x = y := by
  cases x
  cases y
  cases h
  rfl

/-- Schema identifier for the six in-scope v1 families. -/
inductive FamilySchema where
  | Stage1Rows
  | Stage2RegisterReads
  | Stage2RegisterWrites
  | Stage2RamEvents
  | Stage2TwistLinks
  | Stage3Continuity
deriving DecidableEq, Repr

/-- The number of packed columns per family schema. -/
def packedColumnCount : FamilySchema → Nat
  | .Stage1Rows => 2
  | _ => 1

/-- A family evaluation payload: m packed column evaluations.
    m = packedColumnCount(schema). -/
structure FamilyEvalPayload (K : Type*) where
  schema : FamilySchema
  columnEvals : Fin (packedColumnCount schema) → PackedColumnEval K

theorem FamilyEvalPayload.ext
    {K : Type*} {x y : FamilyEvalPayload K}
    (hSchema : x.schema = y.schema)
    (hCols : HEq x.columnEvals y.columnEvals) : x = y := by
  cases x
  cases y
  cases hSchema
  cases hCols
  rfl

/-! ## Opened Object Identity -/

/-- Abstract opened-object identity. In the Rust implementation this carries
    family, commitment_root_digest, layout_version, row_domain_log_size, and
    a canonical digest. Here we abstract it as a type with decidable equality
    and an associated evaluation function. -/
structure OpenedObjectId where
  digest : Nat
deriving DecidableEq, Repr

/-! ## Evaluation Claims -/

/-- A family evaluation claim: the verifier-facing object that Phase 0 emits.
    `ell` is the point arity (= row_domain_log_size). -/
structure FamilyEvalClaim (K : Type*) (ell : Nat) where
  openedObject : OpenedObjectId
  point : Fin ell → K
  payload : FamilyEvalPayload K

/-- A reduced evaluation claim: the Phase 2a output. -/
structure ReducedEvalClaim (K : Type*) (ell : Nat) where
  openedObject : OpenedObjectId
  point : Fin ell → K
  payload : FamilyEvalPayload K
  sourceClaims : List Nat  -- source claim indices for provenance

/-! ## Polynomial Evaluation Abstractions -/

/-- Bit-vector embedding for a Boolean-cube point index (little-endian). -/
def boolCubeBits {K : Type*} [Field K]
    {ell : Nat} (x : Fin (2 ^ ell)) : Fin ell → K :=
  fun i => if Nat.testBit x.1 i.1 then 1 else 0

/-- Equality polynomial: eq(x, y) = Π_i (x_i · y_i + (1 - x_i)(1 - y_i)). -/
noncomputable def eqPoly {K : Type*} [Field K]
    {ell : Nat} (x y : Fin ell → K) : K :=
  Finset.prod Finset.univ fun i : Fin ell =>
    x i * y i + (1 - x i) * (1 - y i)

/-- Abstract multilinear extension evaluation over K.
    `table` has `2^ell` entries, `point` has `ell` coordinates.
    Returns the MLE evaluation: Σ_{x ∈ {0,1}^ell} eq(point, x) · table(x). -/
noncomputable def mleEval {K : Type*} [Field K]
    {ell : Nat} (table : Fin (2 ^ ell) → K) (point : Fin ell → K) : K :=
  Finset.sum Finset.univ fun x : Fin (2 ^ ell) =>
    table x * eqPoly (boolCubeBits x) point

/-! ## PACK / UNPACK -/

/-- The full_width (number of pre-pack field elements per row) for a schema. -/
def fullWidth : FamilySchema → Nat
  | .Stage1Rows => 93
  | .Stage2RegisterReads => 21
  | .Stage2RegisterWrites => 21
  | .Stage2RamEvents => 25
  | .Stage2TwistLinks => 25
  | .Stage3Continuity => 25

/-- Total coefficient capacity in the packed-column view for a schema. -/
def packedCoeffCapacity (schema : FamilySchema) : Nat :=
  packedColumnCount schema * AJTAI_D

/-- The frozen full width fits into the packed-column capacity. -/
theorem fullWidth_le_packedCoeffCapacity (schema : FamilySchema) :
    fullWidth schema ≤ packedCoeffCapacity schema := by
  cases schema <;> native_decide

/-- PACK: maps full_width field elements into m ring-element columns.
    Applied row-by-row during commitment. -/
noncomputable def pack {K : Type*} [Field K] (schema : FamilySchema)
    (row : Fin (fullWidth schema) → K) :
    Fin (packedColumnCount schema) → PackedColumnEval K :=
  fun j =>
    { coeffs := fun t =>
        let idx := j.1 * AJTAI_D + t.1
        if hIdx : idx < fullWidth schema then
          row ⟨idx, hIdx⟩
        else
          0 }

/-- UNPACK: extracts the first full_width field values from the
    coefficient-expanded packed column evaluations. Left-inverse of PACK
    on the full_width-dimensional subspace. -/
noncomputable def unpack {K : Type*} [Field K] (schema : FamilySchema)
    (evals : Fin (packedColumnCount schema) → PackedColumnEval K) :
    Fin (fullWidth schema) → K :=
  fun i =>
    let idx := i.1
    let jNat := idx / AJTAI_D
    let tNat := idx % AJTAI_D
    let hDPos : 0 < AJTAI_D := by native_decide
    let hIdxCap : idx < packedCoeffCapacity schema := by
      exact lt_of_lt_of_le i.2 (fullWidth_le_packedCoeffCapacity schema)
    let hJ : jNat < packedColumnCount schema := by
      exact (Nat.div_lt_iff_lt_mul hDPos).2 (by simpa [packedCoeffCapacity, Nat.mul_comm] using hIdxCap)
    let hT : tNat < AJTAI_D := Nat.mod_lt _ hDPos
    (evals ⟨jNat, hJ⟩).coeffs ⟨tNat, hT⟩

/-! ## Padding Invariant -/

/-- The padding invariant: positions full_width..D*m in each committed row
    are zero. This is enforced by encode_vector_for_full_width. -/
def PaddingInvariant {K : Type*} [Field K] (schema : FamilySchema)
    {ell : Nat} (table : Fin (2 ^ ell) → Fin (fullWidth schema) → K) : Prop :=
  ∀ row : Fin (2 ^ ell),
    ∀ j : Fin (packedColumnCount schema),
      ∀ t : Fin AJTAI_D,
        fullWidth schema ≤ j.1 * AJTAI_D + t.1 →
          (pack schema (table row) j).coeffs t = 0

/-! ## Scalarization -/

/-- Coefficient linearization: collapse D=54 K-valued coefficients into
    one K scalar using challenge eta. -/
noncomputable def coeffLinearize {K : Type*} [Field K]
    (eta : K) (eval : PackedColumnEval K) : K :=
  Finset.sum Finset.univ fun (t : Fin AJTAI_D) => eta ^ (t : Nat) * eval.coeffs t

/-- Gamma linearization: collapse m coefficient-linearized values into
    one K scalar using challenge gamma. Only used when m > 1. -/
noncomputable def gammaLinearize {K : Type*} [Field K]
    {m : Nat} (gamma : K) (ws : Fin m → K) : K :=
  Finset.sum Finset.univ fun (j : Fin m) => gamma ^ (j : Nat) * ws j

/-- Full scalarization of a packed payload: eta first, then gamma. -/
noncomputable def scalarize {K : Type*} [Field K]
    (eta gamma : K) (payload : FamilyEvalPayload K) : K :=
  let m := packedColumnCount payload.schema
  let ws : Fin m → K := fun j => coeffLinearize eta (payload.columnEvals j)
  if h : m = 1 then
    ws ⟨0, by omega⟩
  else
    gammaLinearize gamma ws

/-! ## Sumcheck Types -/

/-- Quadratic round polynomial: a₀ + a₁·x + a₂·x². -/
structure QuadraticRoundPoly (K : Type*) where
  a0 : K
  a1 : K
  a2 : K

/-- Evaluate a quadratic round polynomial at a point. -/
noncomputable def QuadraticRoundPoly.eval {K : Type*} [Field K]
    (p : QuadraticRoundPoly K) (x : K) : K :=
  p.a0 + p.a1 * x + p.a2 * x ^ 2

/-! ## Ajtai PCS Boundary -/

/-- Explicit external PCS boundary for this package.
    The closure plan is to instantiate this from the proved SuperNeo
    lattice/CE opening relation rather than keep any local oracle. -/
structure AjtaiPCSBoundary (K : Type*) where
  verify :
    {ell : Nat} →
      OpenedObjectId →
      (Fin ell → K) →
      FamilyEvalPayload K →
      Prop

end OpeningConvergence
