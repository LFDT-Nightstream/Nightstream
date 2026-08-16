import Nightstream.Assurance.Nebula.AjtaiBinding

set_option autoImplicit false

namespace tests.NebulaAjtaiBinding

open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Assurance.Nebula.AjtaiBinding

def tinyShape : Shape where
  rows := 1
  columns := 1
  degree := 1

def zeroMatrix : Matrix Int tinyShape := fun _ _ => 0

def coefficientMap : CoefficientVector tinyShape →+ Int where
  toFun coefficients := coefficients ⟨0, by decide⟩
  map_zero' := rfl
  map_add' := by intro left right; rfl

def zeroWitness : Witness tinyShape := fun _ _ => 0
def oneWitness : Witness tinyShape := fun _ _ => 1

theorem zeroWitness_ne_oneWitness : zeroWitness ≠ oneWitness := by
  intro equal
  have atColumn := congrFun equal (0 : Fin 1)
  have atZero := congrFun atColumn (0 : Fin 1)
  change 0 = 1 at atZero
  omega

def boolWitness : Bool → Witness tinyShape
  | false => zeroWitness
  | true => oneWitness

def boolMap (value : Bool) : Commitment Int tinyShape :=
  commit zeroMatrix coefficientMap (boolWitness value)

def boolRefinement :
    MapRefinement Bool (Commitment Int tinyShape) Int tinyShape boolMap where
  matrix := zeroMatrix
  coefficientMap := coefficientMap
  witness := boolWitness
  witnessInjective := by
    intro left right equal
    cases left <;> cases right
    · rfl
    · exact False.elim (zeroWitness_ne_oneWitness equal)
    · exact False.elim (zeroWitness_ne_oneWitness equal.symm)
    · rfl
  outputEquiv := Equiv.refl _
  correct := fun _ => rfl

/-- A zero matrix gives an explicit nonzero short kernel. This countermodel
shows why map linearity and distinct setup labels cannot replace Module-SIS
hardness. -/
theorem zero_matrix_collision_exposes_kernel :
    Nonempty (KernelWitness boolRefinement.matrix
      boolRefinement.coefficientMap 3) := by
  apply signed_unit_collision_to_kernel boolRefinement
    (by decide : false ≠ true)
  · rfl
  · intro input column coefficient
    cases input <;> fin_cases column <;> fin_cases coefficient <;>
      decide

theorem compact_shapes_are_fixed :
    primaryShape.rows = 2 ∧ primaryShape.columns = 738 ∧
      primaryShape.degree = 54 ∧
      shortShape.rows = 1 ∧ shortShape.columns = 82 ∧
      shortShape.degree = 54 :=
  exact_compact_shapes

end tests.NebulaAjtaiBinding
