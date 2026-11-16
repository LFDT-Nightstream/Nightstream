import NightstreamFPrime.Gadgets.Multilinear.PointEqualitySupport
import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner
import NightstreamFPrime.Gadgets.Polynomial.HornerSupport

namespace NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial

theorem flatConstraints_varsSatisfy {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (allowed : Nat → Prop)
    (leftSupport : ∀ coordinate,
      Horner.KSupported (interface.left offset coordinate) allowed)
    (rightSupport : ∀ coordinate,
      Horner.KSupported (interface.right offset coordinate) allowed)
    (hornerPointSupport :
      Horner.KSupported (interface.hornerPoint offset) allowed)
    (coefficientsSupport : ∀ coefficient ∈ interface.coefficients offset,
      Horner.KSupported coefficient allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit interface positive).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface positive).main offset),
      expression.VarsSatisfy allowed := by
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsSatisfy allowed
  rw [flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with pointMember | hornerMember
  · apply PointEquality.Owned.flatConstraints_varsSatisfy
      (pointInterfaceAt interface offset) offset allowed
    · intro coordinate
      simpa [pointInterfaceAt] using leftSupport coordinate
    · intro coordinate
      simpa [pointInterfaceAt] using rightSupport coordinate
    · intro index lower upper
      apply localSupport index lower
      rw [PointEquality.Owned.program_recipes_length_of_positive
        (pointInterfaceAt interface offset) offset positive] at upper
      rw [localLength_eq interface positive offset]
      omega
    · exact pointMember
  · apply Horner.Owned.flatConstraints_varsSatisfy
      (hornerInterfaceAt interface offset) (hornerOffset interface offset)
      allowed
    · simpa [hornerInterfaceAt] using hornerPointSupport
    · intro coefficient coefficientMember
      simpa [hornerInterfaceAt] using
        coefficientsSupport coefficient coefficientMember
    · intro index lower upper
      apply localSupport index
      · exact Nat.le_trans (by
          unfold hornerOffset
          omega) lower
      · have endEq :
            hornerOffset interface offset +
                (Horner.Owned.program (hornerInterfaceAt interface offset)
                  (hornerOffset interface offset)).recipes.length =
              offset + localLength
                (Circuit.ops (circuit interface positive).main offset) := by
          rw [localLength_eq interface positive offset]
          unfold hornerOffset
          rw [pointLength_eq_of_positive interface offset positive]
          simp only [Horner.Owned.program, hornerInterfaceAt]
          rw [Horner.compile_recipes_length]
          omega
        rw [← endEq]
        exact upper
    · exact hornerMember

theorem output_varsSatisfy {variableCount : Nat}
    (interface : Interface variableCount) (positive : 0 < variableCount)
    (offset : Nat) (allowed : Nat → Prop)
    (leftSupport : ∀ coordinate,
      Horner.KSupported (interface.left offset coordinate) allowed)
    (rightSupport : ∀ coordinate,
      Horner.KSupported (interface.right offset coordinate) allowed)
    (hornerPointSupport :
      Horner.KSupported (interface.hornerPoint offset) allowed)
    (coefficientsSupport : ∀ coefficient ∈ interface.coefficients offset,
      Horner.KSupported coefficient allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit interface positive).main offset) →
      allowed index) :
    Horner.KSupported (output interface offset) allowed := by
  have pointSupport := PointEquality.Owned.output_varsSatisfy
    (pointInterfaceAt interface offset) offset allowed
    (by intro coordinate; simpa [pointInterfaceAt] using leftSupport coordinate)
    (by intro coordinate; simpa [pointInterfaceAt] using rightSupport coordinate)
    (by
      intro index lower upper
      apply localSupport index lower
      rw [PointEquality.Owned.program_recipes_length_of_positive
        (pointInterfaceAt interface offset) offset positive] at upper
      rw [localLength_eq interface positive offset]
      omega)
  have hornerSupport := Horner.Owned.output_varsSatisfy
    (hornerInterfaceAt interface offset) (hornerOffset interface offset)
    allowed (by simpa [hornerInterfaceAt] using hornerPointSupport)
    (by
      intro coefficient member
      simpa [hornerInterfaceAt] using coefficientsSupport coefficient member)
    (by
      intro index lower upper
      apply localSupport index
      · exact Nat.le_trans (by unfold hornerOffset; omega) lower
      · have endEq :
            hornerOffset interface offset +
                (Horner.Owned.program (hornerInterfaceAt interface offset)
                  (hornerOffset interface offset)).recipes.length =
              offset + localLength
                (Circuit.ops (circuit interface positive).main offset) := by
          rw [localLength_eq interface positive offset]
          unfold hornerOffset
          rw [pointLength_eq_of_positive interface offset positive]
          simp only [Horner.Owned.program, hornerInterfaceAt]
          rw [Horner.compile_recipes_length]
          omega
        rw [← endEq]
        exact upper)
  exact Horner.KSupported.mul pointSupport hornerSupport

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem marks_nil {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) (offset : Nat) :
    marks (Circuit.ops (circuit interface positive).main offset) = [] := by
  show marks (opsAt interface offset) = []
  rw [opsAt, marks_cons, marks_cons, NightstreamFPrime.Circuit.marks_nil, pointOp, hornerOp,
    Op.marks, Op.marks, FormalCircuit.asSubcircuit_marks, FormalCircuit.asSubcircuit_marks,
    pointCircuitAt, hornerCircuitAt, PointEquality.Owned.marks_nil, Horner.Owned.marks_nil]
  rfl

/-- The rows read both points, the Horner point and coefficients, and the gadget's own
locals. -/
theorem free {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) (offset : Nat) {live : Nat → Prop}
    (left : ∀ coordinate, Horner.KSupported (interface.left offset coordinate) live)
    (right : ∀ coordinate, Horner.KSupported (interface.right offset coordinate) live)
    (hornerPoint : Horner.KSupported (interface.hornerPoint offset) live)
    (coefficients : ∀ coefficient ∈ interface.coefficients offset,
      Horner.KSupported coefficient live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface positive).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface positive).main offset),
      row.VarsSatisfy live :=
  flatConstraints_varsSatisfy interface positive offset live left right hornerPoint coefficients
    locals

theorem output_live {variableCount : Nat} (interface : Interface variableCount)
    (positive : 0 < variableCount) (offset : Nat) {live : Nat → Prop}
    (left : ∀ coordinate, Horner.KSupported (interface.left offset coordinate) live)
    (right : ∀ coordinate, Horner.KSupported (interface.right offset coordinate) live)
    (hornerPoint : Horner.KSupported (interface.hornerPoint offset) live)
    (coefficients : ∀ coefficient ∈ interface.coefficients offset,
      Horner.KSupported coefficient live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface positive).main offset) →
        live column) :
    Horner.KSupported (output interface offset) live :=
  output_varsSatisfy interface positive offset live left right hornerPoint coefficients locals

end NightstreamFPrime.Gadgets.Multilinear.PointWeightedHorner.Owned
