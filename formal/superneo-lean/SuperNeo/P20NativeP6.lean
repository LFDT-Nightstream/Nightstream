import SuperNeo.P20
import SuperNeo.DecompNative

/-! P20 wrappers that consume theorem-native P6 base-2 decomposition closures. -/

namespace SuperNeo

theorem p20DecompProp_of_base_two_of_state_zero_of_allCanonical
  {z : Array F} {k : Nat}
  (hKPos : 0 < k)
  (hZero : splitScalarTerminalZeroProp z 2 k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  p20DecompProp z 2 k := by
  exact p20DecompProp_of_splitRoundTrip
    (splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
      (z := z) (k := k) hKPos hZero hCanon)

theorem p20DecompProp_of_base_two_of_residue_fold_eq_centeredInt_of_allCanonical
  {z : Array F} {k : Nat}
  (hKPos : 0 < k)
  (hEq : ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  p20DecompProp z 2 k := by
  exact p20DecompProp_of_splitRoundTrip
    (splitRoundTrip_true_of_base_two_of_residue_fold_eq_centeredInt_of_allCanonical
      (z := z) (k := k) hKPos hEq hCanon)

/--
Concrete native base-2/k=8 decomposition closure from challenge-coefficient rows.
-/
theorem p20DecompProp_of_base_two_k8_of_allChallenge
  {z : Array F}
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  p20DecompProp z 2 8 := by
  exact p20DecompProp_of_splitRoundTrip
    (splitRoundTrip_true_of_base_two_k8_of_allChallenge
      (z := z) hChallenge)

/--
Native base-2 decomposition closure from challenge-coefficient rows for any
`k ≥ 8`.
-/
theorem p20DecompProp_of_base_two_of_allChallenge_of_ge8
  {z : Array F} {k : Nat}
  (hk : 8 ≤ k)
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  p20DecompProp z 2 k := by
  exact p20DecompProp_of_splitRoundTrip
    (splitRoundTrip_true_of_base_two_of_allChallenge_of_ge8
      (z := z) (k := k) hk hChallenge)

theorem p20ArithmeticBundle_of_assumptions_with_native_p6_base2
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hKPos : 0 < k)
  (hZero : splitScalarTerminalZeroProp zDecomp 2 k)
  (hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 k hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := p20DecompProp_of_base_two_of_state_zero_of_allCanonical
      (z := zDecomp) (k := k) hKPos hZero hCanon)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_assumptions_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hKPos : 0 < k)
  (hEq : ∀ j (_hj : j < zDecomp.size), centeredInt zDecomp[j]! = splitScalarResidueFoldInt zDecomp[j]! 2 k)
  (hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 k hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions_with_native_p6_base2
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
      (z := zDecomp) (k := k) hEq)
    (hCanon := hCanon)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_checks_with_native_p6_base2
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hKPos : 0 < k)
  (hZero : splitScalarTerminalZeroProp zDecomp 2 k)
  (hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true)
  (hP12 : matrixTransformIdentity bar m z = true)
  (hP14 : evalHom2 bar m z1 z2 r ρ1 ρ2 = true)
  (hVecAdd : preservesAddVec hVec z1 z2 = true)
  (hVecScale : preservesScaleVec hVec ρ1 z1 = true)
  (hScalAdd : preservesAddScalar hScal z1 z2 = true)
  (hScalScale : preservesScaleScalar hScal ρ1 z1 = true)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 k hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP6 : splitRoundTrip zDecomp 2 k = true := by
    exact splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
      (z := zDecomp) (k := k) hKPos hZero hCanon
  exact p20ArithmeticBundle_of_checks
    (hP6 := hP6)
    (hP12 := hP12)
    (hP14 := hP14)
    (hVecAdd := hVecAdd)
    (hVecScale := hVecScale)
    (hScalAdd := hScalAdd)
    (hScalScale := hScalScale)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18Eq := hP18Eq)
    (hP18SZ := hP18SZ)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_checks_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hKPos : 0 < k)
  (hEq : ∀ j (_hj : j < zDecomp.size), centeredInt zDecomp[j]! = splitScalarResidueFoldInt zDecomp[j]! 2 k)
  (hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true)
  (hP12 : matrixTransformIdentity bar m z = true)
  (hP14 : evalHom2 bar m z1 z2 r ρ1 ρ2 = true)
  (hVecAdd : preservesAddVec hVec z1 z2 = true)
  (hVecScale : preservesScaleVec hVec ρ1 z1 = true)
  (hScalAdd : preservesAddScalar hScal z1 z2 = true)
  (hScalScale : preservesScaleScalar hScal ρ1 z1 = true)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 k hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_checks_with_native_p6_base2
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
      (z := zDecomp) (k := k) hEq)
    (hCanon := hCanon)
    (hP12 := hP12)
    (hP14 := hP14)
    (hVecAdd := hVecAdd)
    (hVecScale := hVecScale)
    (hScalAdd := hScalAdd)
    (hScalScale := hScalScale)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18Eq := hP18Eq)
    (hP18SZ := hP18SZ)
    (hP19 := hP19)

/--
Concrete native base-2/k=8 bundle constructor from challenge-coefficient
decomposition rows.
-/
theorem p20ArithmeticBundle_of_assumptions_with_native_p6_base2_k8_of_allChallenge
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hChallenge : ∀ j (hj : j < zDecomp.size), IsChallengeCoeff (zDecomp[j]'hj))
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 8 hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := p20DecompProp_of_base_two_k8_of_allChallenge
      (z := zDecomp) hChallenge)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Native base-2 arithmetic-bundle constructor from challenge-coefficient
decomposition rows for any `k ≥ 8`.
-/
theorem p20ArithmeticBundle_of_assumptions_with_native_p6_base2_of_allChallenge_of_ge8
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hk : 8 ≤ k)
  (hChallenge : ∀ j (hj : j < zDecomp.size), IsChallengeCoeff (zDecomp[j]'hj))
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 2 k hVec hScal cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions_with_native_p6_base2
    (hKPos := Nat.lt_of_lt_of_le (by decide : 0 < 8) hk)
    (hZero := splitScalarTerminalZeroProp_of_allChallenge_base2_of_ge8
      (z := zDecomp) (k := k) hk hChallenge)
    (hCanon := allCanonical_of_allChallenge (z := zDecomp) hChallenge)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

end SuperNeo
