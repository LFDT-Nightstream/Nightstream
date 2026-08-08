import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-! Exact small identity-first R1CS relation used by the production golden run. -/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.NifsProductionGolden.FixedRelation

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

def shape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape where
  rowVariables := 6
  logicalWidth := 54
  matrixCount := 4
  publicRingColumns := 1
  publicFits := by decide

def matrixValue (matrix : Fin 4) (vertex : BooleanVertex 6)
    (column : Fin 54) : F :=
  let row := NumericBooleanDomain.index vertex
  match matrix.val with
  | 0 => if row = column.val then 1 else 0
  | 1 => if row == 0 && column.val == 1 then 1 else 0
  | 2 => if row == 0 && column.val == 0 then 1 else 0
  | _ => if row == 0 && column.val == 1 then 1 else 0

def multiplicationTerm : Monomial F 4 where
  coefficient := 1
  exponents := fun matrix =>
    if matrix.val = 1 || matrix.val = 2 then 1 else 0

def subtractionTerm : Monomial F 4 where
  coefficient := -1
  exponents := fun matrix => if matrix.val = 3 then 1 else 0

def polynomial : ConstraintPolynomial F 4 where
  degreeBound := 3
  terms := [multiplicationTerm, subtractionTerm]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> native_decide

def system : Structure shape where
  matrices := fun matrix vertex column => matrixValue matrix vertex column
  constraintPolynomial := polynomial

/-- Verifier-row key is irrelevant to public recomposition, but the concrete
algebras keep its exact type visible. -/
def zeroKey :
    PiRLCAlgebra.Commitment.Key shape 18 :=
  fun _ _ _ => 0

end Nightstream.Implementation.Rust.NifsProductionGolden.FixedRelation
