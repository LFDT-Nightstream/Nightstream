import NightstreamFPrime.Gadgets.Multilinear.PointEqualitySupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.GammaPowers
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity

/-!
Owns variable-support propagation for the complete production PiCCS final
identity. It changes no exponent, child order, circuit, or row.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem productionCubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

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

/-- Exact support propagation through both owned children and the final
two extension-component equality rows. -/
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
    norm_num [productionShape, Phi81MatrixSource.phi81Shape,
      cubeVariables] at upper
    simpa [privateCount] using (Nat.lt_trans upper (by omega :
      offset + 110 < offset + 142))
  have gammaLocal : ∀ index, gammaOffset interface offset ≤ index →
      index < gammaOffset interface offset + 32 → allowed index := by
    intro index lower upper
    apply localSupport index
    · unfold gammaOffset at lower
      omega
    · unfold gammaOffset at upper
      rw [pointLength_eq] at upper
      simpa [privateCount, Nat.add_assoc] using upper
  have pointRows := PointEquality.Owned.flatConstraints_varsSatisfy
    (pointInterfaceAt interface offset) offset allowed
    (by intro coordinate; simpa [pointInterfaceAt] using
      roundPointSupport coordinate)
    (by intro coordinate; simpa [pointInterfaceAt] using alphaSupport coordinate)
    pointLocal
  have gammaRows := GammaPowers.flatConstraints_varsSatisfy
    (interface.gamma offset) (gammaOffset interface offset) allowed gammaSupport gammaLocal
  have pointOutputSupport := PointEquality.Owned.output_varsSatisfy
    (pointInterfaceAt interface offset) offset allowed
    (by intro coordinate; simpa [pointInterfaceAt] using
      roundPointSupport coordinate)
    (by intro coordinate; simpa [pointInterfaceAt] using alphaSupport coordinate)
    pointLocal
  have matrixOutputSupport := GammaPowers.wire_varsSatisfy
    (interface.gamma offset) (gammaOffset interface offset) 11 allowed gammaSupport (by decide) gammaLocal
  have constraintOutputSupport := GammaPowers.wire_varsSatisfy
    (interface.gamma offset) (gammaOffset interface offset) 16 allowed gammaSupport (by decide) gammaLocal
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
  · rcases List.mem_append.mp coreMember with pointMember | gammaMember
    · exact pointRows expression pointMember
    · exact gammaRows expression gammaMember
  · exact equalities_supported (interface.terminal offset)
      (terminalExpr interface offset) allowed terminalSupport
      terminalExprSupport expression (by
        simpa [terminalAssertions] using terminalMember)

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity
