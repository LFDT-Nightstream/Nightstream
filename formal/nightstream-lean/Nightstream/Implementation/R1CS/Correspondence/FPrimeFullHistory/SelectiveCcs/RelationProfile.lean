import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.RowPadding

/-!
Semantic shape and finite-matrix owner for the active selective F' relation.

Assurance tier: model-level. Matrix arity and the gate polynomial come from
the independent selective semantics, not from a generated circuit or profiler.

Owns: the role-indexed thirteen-matrix finite relation; its exhaustive
role-to-physical-port map; an overflow-free meaning of the minimal Boolean row
domain; construction of its typed Phi81 shape and padded matrix structure; and
the incompatibility of that shape with a three-row CE carrier.

Does not own: the concrete fixed-point row/column counts, proof that Rust
selects those dimensions, generated matrix payloads, CE messages, transcript
authority, R1CS rows, costs, or row removal. A production inhabitant must come
from a future compiler-refinement theorem.

Emits constraints: no.

Authority boundary: callers supply finite matrices plus shape-dependent row
and width evidence. They cannot choose the matrix count or polynomial. The
row policy is stated over unbounded `Nat`, so machine overflow cannot silently
change its meaning.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `f_prime.relation.profile.rows` | `rowVariables` is the least exponent covering `max rows 2` | checked | `ExactRowDomain` |
| `f_prime.relation.profile.matrices` | one named matrix image for every selective role and exactly one physical port per role | computed | `FiniteRelation.matrices`, `FiniteRelation.matrixAt`, `Profile.shape_matrixCount_eq_13` |
| `f_prime.relation.profile.polynomial` | the relation uses only the independent selective polynomial | computed | `FiniteRelation.toStructure` |
| `f_prime.relation.profile.padding` | undeclared Boolean-domain rows are definitionally zero | computed | `FiniteRelation.toStructure` |
| `f_prime.relation.profile.three_row_mismatch` | three evaluations cannot carry all thirteen matrix images | derived | `Profile.shape_matrixCount_ne_three` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Matrix arity inherited from the independent selective port vocabulary. -/
def matrixCount : Nat :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.portCount

/-- Overflow-free specification of Rust's
`max(rows, 2).next_power_of_two().trailing_zeros()` policy. -/
def ExactRowDomain (rows rowVariables : Nat) : Prop :=
  max rows 2 <= 2 ^ rowVariables /\
    forall smaller, smaller < rowVariables ->
      2 ^ smaller < max rows 2

/-- Active selective matrices before embedding finite rows into a Boolean
cube. Every matrix is owned by a semantic role; numeric ports are derived
through the sole role/index equivalence. -/
structure FiniteRelation (rows columns : Nat) where
  matrices :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports.Role ->
    RowPadding.NumericMatrix F rows columns

/-- Shape-dependent facts that the fixed-point compiler must eventually
derive. Matrix arity and polynomial are intentionally absent. -/
structure Profile (rows columns : Nat) where
  rowVariables : Nat
  rowDomain : ExactRowDomain rows rowVariables
  publicFits :
    ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth columns

namespace Profile

/-- The semantic Phi81 shape of the active selective relation. -/
def shape {rows columns : Nat} (profile : Profile rows columns) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.Shape where
  rowVariables := profile.rowVariables
  logicalWidth := columns
  matrixCount := matrixCount
  publicRingColumns := publicRingColumns
  publicFits := profile.publicFits

theorem rows_covered {rows columns : Nat} (profile : Profile rows columns) :
    rows <= 2 ^ profile.rowVariables := by
  exact Nat.le_trans (Nat.le_max_left rows 2) profile.rowDomain.1

@[simp] theorem shape_rowVariables
    {rows columns : Nat} (profile : Profile rows columns) :
    (shape profile).rowVariables = profile.rowVariables := by
  rfl

@[simp] theorem shape_logicalWidth
    {rows columns : Nat} (profile : Profile rows columns) :
    (shape profile).logicalWidth = columns := by
  rfl

@[simp] theorem shape_matrixCount
    {rows columns : Nat} (profile : Profile rows columns) :
    (shape profile).matrixCount = matrixCount := by
  rfl

/-- The independently specified selective relation has exactly thirteen
matrix images. -/
theorem shape_matrixCount_eq_13
    {rows columns : Nat} (profile : Profile rows columns) :
    (shape profile).matrixCount = 13 := by
  rfl

/-- A three-evaluation carrier cannot be the complete CE carrier of this
relation. Any future compression must establish a separate protocol theorem. -/
theorem shape_matrixCount_ne_three
    {rows columns : Nat} (profile : Profile rows columns) :
    (shape profile).matrixCount ≠ 3 := by
  rw [shape_matrixCount_eq_13]
  decide

end Profile

namespace FiniteRelation

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports

/-- Matrix selected by the physical polynomial port. The inverse role map
makes it impossible for a physical port to remain unnamed. -/
def matrixAt
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (port : Fin matrixCount) :
    RowPadding.NumericMatrix F rows columns :=
  relation.matrices (Role.ofIndex port)

@[simp] theorem matrixAt_role
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (role : Role) :
    relation.matrixAt role.index = relation.matrices role := by
  change relation.matrices (Role.ofIndex role.index) = relation.matrices role
  rw [Role.ofIndex_index]

/-- Embed finite rows into the exact Boolean row domain and attach the sole
independent selective constraint polynomial. -/
def toStructure
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (profile : Profile rows columns) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.Structure
      (Profile.shape profile) where
  matrices := fun port => RowPadding.padRows (relation.matrixAt port)
  constraintPolynomial :=
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial

@[simp] theorem toStructure_matrix
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (profile : Profile rows columns)
    (port : Fin matrixCount) :
    (relation.toStructure profile).matrices port =
      RowPadding.padRows (relation.matrixAt port) := by
  rfl

/-- Role-facing form of the structure matrix theorem. The same role names the
independent polynomial variable and the physical matrix port. -/
@[simp] theorem toStructure_roleMatrix
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (profile : Profile rows columns)
    (role : Role) :
    (relation.toStructure profile).matrices role.index =
      RowPadding.padRows (relation.matrices role) := by
  rw [toStructure_matrix, matrixAt_role]

@[simp] theorem toStructure_constraintPolynomial
    {rows columns : Nat}
    (relation : FiniteRelation rows columns)
    (profile : Profile rows columns) :
    (relation.toStructure profile).constraintPolynomial =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial := by
  rfl

end FiniteRelation

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile
