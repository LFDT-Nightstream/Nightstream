import NightstreamFPrime.Gadgets.Multilinear.PointEqualitySupport
import NightstreamFPrime.Gadgets.Polynomial.PowerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem productionCubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, RingMatrixSource.ringShape, cubeVariables]

private theorem equalities_supported (left right : KExpr)
    (allowed : Nat → Prop) (leftSupport : Horner.KSupported left allowed)
    (rightSupport : Horner.KSupported right allowed) :
    ∀ expression ∈ KExpr.equalities left right,
      expression.VarsSatisfy allowed := by
  intro expression member
  simp only [KExpr.equalities, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact ⟨leftSupport.1, ⟨trivial, rightSupport.1⟩⟩
  · exact ⟨leftSupport.2, ⟨trivial, rightSupport.2⟩⟩

theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (roundPointSupport : ∀ coordinate,
      Horner.KSupported (interface.roundPoint offset coordinate) allowed)
    (alphaSupport : ∀ coordinate,
      Horner.KSupported (interface.alpha offset coordinate) allowed)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (evalKSupport : Horner.KSupported (interface.eval_K offset) allowed)
    (evalASupport : Horner.KSupported (interface.eval_A offset) allowed)
    (ccsSupport : Horner.KSupported (interface.ccs offset) allowed)
    (normSupport : Horner.KSupported (interface.norm offset) allowed)
    (terminalSupport : Horner.KSupported (interface.terminal offset) allowed)
    (localSupport : ∀ index,
      offset ≤ index → index < offset + privateCount → allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have pointLocal : ∀ index,
      offset ≤ index →
      index < offset +
        (PointEquality.Owned.program
          (pointInterfaceAt interface offset) offset).recipes.length →
      allowed index := by
    intro index lower upper
    apply localSupport index lower
    rw [PointEquality.Owned.program_recipes_length_of_positive
      (pointInterfaceAt interface offset) offset
        productionCubeVariables_positive] at upper
    norm_num [productionShape, RingMatrixSource.ringShape,
      cubeVariables] at upper
    simpa [privateCount] using (Nat.lt_trans upper (by omega :
      offset + 110 < offset + 2813))
  have matrixLocal : ∀ index,
      matrixOffset interface offset ≤ index →
      index < matrixOffset interface offset + localLength
        (Circuit.ops (matrixPowerCircuitAt interface offset).main
          (matrixOffset interface offset)) →
      allowed index := by
    intro index lower upper
    apply localSupport index
    · exact Nat.le_trans (by unfold matrixOffset; omega) lower
    · unfold matrixPowerCircuitAt at upper
      rw [Power.localLength_eq] at upper
      unfold matrixOffset pointLength pointCircuitAt at upper
      rw [PointEquality.Owned.localLength_eq_of_positive
        (pointInterfaceAt interface offset) offset
          productionCubeVariables_positive] at upper
      rw [matrixExponent_eq] at upper
      norm_num [productionShape, RingMatrixSource.ringShape,
        cubeVariables] at upper
      simpa [privateCount] using (Nat.lt_trans upper (by omega :
        offset + 110 + 2688 < offset + 2813))
  have constraintLocal : ∀ index,
      constraintOffset interface offset ≤ index →
      index < constraintOffset interface offset + localLength
        (Circuit.ops (constraintPowerCircuitAt interface offset).main
          (constraintOffset interface offset)) →
      allowed index := by
    intro index lower upper
    apply localSupport index
    · exact Nat.le_trans (by
        unfold constraintOffset matrixOffset
        omega) lower
    · unfold constraintPowerCircuitAt at upper
      rw [Power.localLength_eq] at upper
      unfold constraintOffset matrixOffset matrixLength matrixPowerCircuitAt
        pointLength pointCircuitAt at upper
      rw [Power.localLength_eq] at upper
      rw [PointEquality.Owned.localLength_eq_of_positive
        (pointInterfaceAt interface offset) offset
          productionCubeVariables_positive] at upper
      rw [matrixExponent_eq, constraintFactor_eq] at upper
      norm_num [productionShape, RingMatrixSource.ringShape,
        cubeVariables] at upper
      simpa [privateCount] using upper
  have pointRows := PointEquality.Owned.flatConstraints_varsSatisfy
    (pointInterfaceAt interface offset) offset allowed
    (by intro coordinate; simpa [pointInterfaceAt] using
      roundPointSupport coordinate)
    (by intro coordinate; simpa [pointInterfaceAt] using alphaSupport coordinate)
    pointLocal
  have matrixRows := Power.flatConstraints_varsSatisfy (matrixExponentOf productionShape)
    (matrixPowerInterfaceAt interface offset) (matrixOffset interface offset)
    allowed (by simpa [matrixPowerInterfaceAt] using gammaSupport) matrixLocal
  have pointOutputSupport := PointEquality.Owned.output_varsSatisfy
    (pointInterfaceAt interface offset) offset allowed
    (by intro coordinate; simpa [pointInterfaceAt] using
      roundPointSupport coordinate)
    (by intro coordinate; simpa [pointInterfaceAt] using alphaSupport coordinate)
    pointLocal
  have matrixOutputSupport := Power.output_varsSatisfy (matrixExponentOf productionShape)
    (matrixPowerInterfaceAt interface offset) (matrixOffset interface offset)
    allowed (by simpa [matrixPowerInterfaceAt] using gammaSupport) matrixLocal
  have constraintPointSupport : Horner.KSupported
      ((constraintPowerInterfaceAt interface offset).point
        (constraintOffset interface offset)) allowed := by
    simpa [constraintPowerInterfaceAt, gammaMatrixOutput] using
      matrixOutputSupport
  have constraintRows := Power.flatConstraints_varsSatisfy (constraintFactorOf productionShape)
    (constraintPowerInterfaceAt interface offset)
    (constraintOffset interface offset) allowed constraintPointSupport
    constraintLocal
  have constraintOutputSupport := Power.output_varsSatisfy (constraintFactorOf productionShape)
    (constraintPowerInterfaceAt interface offset)
    (constraintOffset interface offset) allowed constraintPointSupport
    constraintLocal
  have terminalExprSupport :
      Horner.KSupported (terminalExpr interface offset) allowed := by
    unfold terminalExpr gammaMatrixOutput gammaConstraintOutput
      pointEqualityOutput
    exact Horner.KSupported.add evalKSupport
      (Horner.KSupported.add
        (Horner.KSupported.mul matrixOutputSupport evalASupport)
        (Horner.KSupported.mul constraintOutputSupport
          (Horner.KSupported.mul pointOutputSupport
            (Horner.KSupported.add ccsSupport
              (Horner.KSupported.mul gammaSupport normSupport)))))
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsSatisfy allowed
  rw [flatConstraints_opsAt]
  intro expression member
  rcases List.mem_append.mp member with coreMember | terminalMember
  · rcases List.mem_append.mp coreMember with firstTwoMember |
        constraintMember
    · rcases List.mem_append.mp firstTwoMember with pointMember |
          matrixMember
      · exact pointRows expression pointMember
      · exact matrixRows expression matrixMember
    · exact constraintRows expression constraintMember
  · exact equalities_supported (interface.terminal offset)
      (terminalExpr interface offset) allowed terminalSupport
      terminalExprSupport expression (by
        simpa [terminalAssertions] using terminalMember)

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

section Exporter

variable {shape : Shape}

theorem shapeMarks_nil (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables)
    (offset : Nat) :
    marks (Circuit.ops (shapeCircuit interface positive).main offset) = [] := by
  show marks (opsAt interface offset) = []
  have assertions : marks ((terminalAssertions interface offset).map Op.assertZero) = [] := by
    simp [marks, Op.marks]
  rw [opsAt, marks_append, assertions, List.append_nil, marks_cons, marks_cons, marks_cons,
    NightstreamFPrime.Circuit.marks_nil, pointOp, matrixPowerOp, constraintPowerOp,
    pointSubcircuit, matrixPowerSubcircuit, constraintPowerSubcircuit, Op.marks, Op.marks,
    Op.marks, FormalCircuit.asSubcircuit_marks, FormalCircuit.asSubcircuit_marks,
    FormalCircuit.asSubcircuit_marks, pointCircuitAt, matrixPowerCircuitAt,
    constraintPowerCircuitAt, PointEquality.Owned.marks_nil, Power.marks_nil, Power.marks_nil]
  rfl

/-- The rows read the round point, α, γ, the four claims, the terminal and the gadget's own
locals. -/
theorem shapeFree (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables)
    (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (alpha : ∀ coordinate, Horner.KSupported (interface.alpha offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (evalK : Horner.KSupported (interface.eval_K offset) live)
    (evalA : Horner.KSupported (interface.eval_A offset) live)
    (ccs : Horner.KSupported (interface.ccs offset) live)
    (norm : Horner.KSupported (interface.norm offset) live)
    (terminal : Horner.KSupported (interface.terminal offset) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (shapeCircuit interface positive).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (shapeCircuit interface positive).main offset),
      row.VarsSatisfy live := by
  have length : localLength (Circuit.ops (shapeCircuit interface positive).main offset) =
      pointLength interface offset + matrixLength interface offset +
        constraintLength interface offset :=
    (shapeCircuitFast interface positive).privateCount_eq offset
  have pointLocals : ∀ column, offset ≤ column →
      column < offset + pointLength interface offset → live column :=
    fun column lower upper => locals column lower (by rw [length]; omega)
  have matrixLocals : ∀ column, matrixOffset interface offset ≤ column →
      column < matrixOffset interface offset + matrixLength interface offset → live column := by
    intro column lower upper
    unfold matrixOffset at lower upper
    exact locals column (by omega) (by rw [length]; omega)
  have constraintLocals : ∀ column, constraintOffset interface offset ≤ column →
      column < constraintOffset interface offset + constraintLength interface offset →
        live column := by
    intro column lower upper
    unfold constraintOffset matrixOffset at lower upper
    exact locals column (by omega) (by rw [length]; omega)
  have pointRows := PointEquality.Owned.free (pointInterfaceAt interface offset) offset
    roundPoint alpha pointLocals
  have pointOutput : Horner.KSupported (pointEqualityOutput interface offset) live :=
    PointEquality.Owned.output_live (pointInterfaceAt interface offset) offset roundPoint alpha
      pointLocals
  have matrixRows := Power.free (matrixExponentOf shape) (matrixPowerInterfaceAt interface offset)
    (matrixOffset interface offset) gamma matrixLocals
  have matrixOutput : Horner.KSupported (gammaMatrixOutput interface offset) live :=
    Power.output_live (matrixExponentOf shape) (matrixPowerInterfaceAt interface offset)
      (matrixOffset interface offset) gamma matrixLocals
  have constraintRows := Power.free (constraintFactorOf shape)
    (constraintPowerInterfaceAt interface offset) (constraintOffset interface offset)
    matrixOutput constraintLocals
  have constraintOutput : Horner.KSupported (gammaConstraintOutput interface offset) live :=
    Power.output_live (constraintFactorOf shape) (constraintPowerInterfaceAt interface offset)
      (constraintOffset interface offset) matrixOutput constraintLocals
  have terminalExprLive : Horner.KSupported (terminalExpr interface offset) live :=
    Horner.KSupported.add evalK
      (Horner.KSupported.add (Horner.KSupported.mul matrixOutput evalA)
        (Horner.KSupported.mul constraintOutput
          (Horner.KSupported.mul pointOutput
            (Horner.KSupported.add ccs (Horner.KSupported.mul gamma norm)))))
  change ∀ row ∈ flatConstraints (opsAt interface offset), row.VarsSatisfy live
  rw [flatConstraints_opsAt]
  intro row member
  simp only [List.mem_append] at member
  rcases member with ((pointMember | matrixMember) | constraintMember) | terminalMember
  · exact pointRows row pointMember
  · exact matrixRows row matrixMember
  · exact constraintRows row constraintMember
  · exact equalities_supported (interface.terminal offset) (terminalExpr interface offset) live
      terminal terminalExprLive row terminalMember

theorem marks_nil (interface : Interface) (offset : Nat) :
    marks (Circuit.ops (circuit interface).main offset) = [] :=
  shapeMarks_nil interface productionCubeVariables_positive offset

/-- The rows read the round point, α, γ, the four claims, the terminal and the gadget's own
locals. -/
theorem free (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (alpha : ∀ coordinate, Horner.KSupported (interface.alpha offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (evalK : Horner.KSupported (interface.eval_K offset) live)
    (evalA : Horner.KSupported (interface.eval_A offset) live)
    (ccs : Horner.KSupported (interface.ccs offset) live)
    (norm : Horner.KSupported (interface.norm offset) live)
    (terminal : Horner.KSupported (interface.terminal offset) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main offset), row.VarsSatisfy live :=
  shapeFree interface productionCubeVariables_positive offset roundPoint alpha gamma evalK evalA
    ccs norm terminal locals

end Exporter

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity
