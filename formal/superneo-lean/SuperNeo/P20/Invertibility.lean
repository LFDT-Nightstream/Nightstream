import SuperNeo.P20.Checks

/-! P20 invertibility witness extraction wrappers. -/

namespace SuperNeo

open F

/--
Assumption-driven invertibility witness extraction for the concrete delta tracked in P20.
-/
theorem p20InvertibilityWitness_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption
    hInv (p20ArithmeticBundle_invertibilityWindow (hP20 := hP20))

/--
Helper: derive `0 < normInfCoeffs (mulRq aDelta bDelta)` from the P20 bundle's
invertibility-window obligation for `invDelta = mulRq aDelta bDelta`.
-/
theorem p20_pos_normInfCoeffs_mulRq_of_p20ArithmeticBundle
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 :
    p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples
      invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta) :
  0 < normInfCoeffs (mulRq aDelta bDelta) := by
  exact p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow
    (invDelta := invDelta) (aDelta := aDelta) (bDelta := bDelta)
    (p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)) hDeltaEq

/--
Core wrapper for the product-based invertibility-witness specializations below.

Given `invDelta = mulRq aDelta bDelta`, this:
1. extracts the required positivity side-condition from the P20 bundle, and
2. transports any witness for `mulRq aDelta bDelta` back to the tracked `invDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_assumption_core
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 :
    p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples
      invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hInvMul :
    0 < normInfCoeffs (mulRq aDelta bDelta) →
      ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_p20ArithmeticBundle (hP20 := hP20) (hDeltaEq := hDeltaEq)
  have hInvMul' : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    hInvMul hPosMul
  simpa [hDeltaEq] using hInvMul'

/--
Generic operand/raw-coeff-bound witness extraction when the tracked `invDelta`
is identified with a product `mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_rawCoeff_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_rawCoeff_of_assumption
    hInv hA hB hPosMul hRawCoeff hAddSub hSub hBLt

/--
Field-op collapse variant of generic operand/raw-coeff witness extraction when
`invDelta = mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp_of_assumption
    hInv hA hB hPosMul hRawCoeff hOps hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BTerm BRaw B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_assumption
    hInv hA hB hPosMul hMul hAdd hZero hAddSub hSub hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BTerm B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub : rawAddSubCollapseBound ((D * D) * BTerm) B)
  (hSub : rawSubCollapseBound ((D * D) * BTerm) B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_of_assumption
    hInv hA hB hPosMul hMul hAddTri hAddSub hSub hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BTerm : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm))
  (hBLt : ((D * D) * BTerm) < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp_of_assumption
    hInv hA hB hPosMul hMul hAddTri hOps hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_fieldOp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BTerm B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ B → normInfF y ≤ BTerm → normInfF (x + y) ≤ B)
  (hZero : normInfF (0 : F) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_fieldOp_of_assumption
    hInv hA hB hPosMul hMul hAdd hZero hOps hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BTerm BRaw B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le_of_assumption
    hInv hA hB hPosMul hMul hTermLe hAddCollapse hAddSub hSub hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_of_assumption
    hInv hA hB hPosMul hMul hAddCollapse hAddSub hSub hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp_of_assumption
    hInv hA hB hPosMul hMul hOps hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
    hInv hA hB hPosMul hMulUniv hAddTri hSubTri hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption
    hInv hA hB hPosMul hMulUniv hAddTri hSubTri hBLt

/--
Triangle-bundle variant of
`p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption`.
-/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
    hInv hA hB hPosMul hMulUniv hTri hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
    hInv hA hB hPosMul hMulUniv hAddTri hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    hInv hA hB hPosMul hMulRep hAddRep hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := centeredRepMulAddBounds_mul hRep)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hBLt := hBLt)

/-- Assumption-free native invertibility-witness wrapper over a P20 bundle. -/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
    (hInv := hInv) (hP20 := hP20) (hDeltaEq := hDeltaEq)
    (hA := hA) (hB := hB)
    (hMulUniv := schoolbookMulUniversalBound_theorem)
    (hAddTri := schoolbookTriangleBounds_add schoolbookTriangleBounds_theorem)
    (hSubTri := schoolbookTriangleBounds_sub schoolbookTriangleBounds_theorem)
    (hBLt := hBLt)

/-- Assumption-free native-tight invertibility-witness wrapper over a P20 bundle. -/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
    (hInv := hInv) (hP20 := hP20) (hDeltaEq := hDeltaEq)
    (hA := hA) (hB := hB)
    (hMulUniv := schoolbookMulUniversalBound_theorem)
    (hTri := schoolbookTriangleBounds_theorem)
    (hBLt := hBLt)

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_tight_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := hMulRep)
    (hAddRep := centeredRepAddTriangleBound_theorem)
    (hBLt := hBLt)

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    hInv hA hB hPosMul hRawFromOperands hAddTri hSubTri hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    hInv hA hB hPosMul hRawFromOperands hAddTri hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
    hInv hA hB hPosMul hRawFromOperands hAddRep hBLt

theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

/--
Concrete Goldilocks-bound witness extraction when the tracked `invDelta`
is identified with a product `mulRq a b` and operand-bound assumptions are provided.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
    hInv hA hB hPosMul hRaw hCollapse

/--
In-range raw-coefficient variant of the concrete Goldilocks-bound witness extraction.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_inRange_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hCollapse := hCollapse)

/--
Field-op-collapse variant of concrete Goldilocks-bound witness extraction when
`invDelta = mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := hRaw)
    (hCollapse := goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

/--
In-range raw-coefficient + field-op-collapse variant of concrete Goldilocks-bound
witness extraction.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hFieldOps := hFieldOps)

theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hCollapse := hCollapse)

theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
    (hInv := hInv)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hFieldOps := hFieldOps)


end SuperNeo
