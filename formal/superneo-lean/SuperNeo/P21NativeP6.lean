import SuperNeo.P21
import SuperNeo.DecompNative

/-! P21 wrappers that consume theorem-native P6 residue-fold equalities (base 2). -/

namespace SuperNeo

theorem p21ProtocolTarget_of_assumptions_with_native_p6_base2_of_residue_fold_eq_centeredInt
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
  (hEq : ∀ j (_hj : j < zDecomp.size),
    centeredInt zDecomp[j]! = splitScalarResidueFoldInt zDecomp[j]! 2 k)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_assumptions_with_native_p6_base2
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

theorem p21ProtocolTarget_of_checks_with_native_p6_base2_of_residue_fold_eq_centeredInt
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
  (hEq : ∀ j (_hj : j < zDecomp.size),
    centeredInt zDecomp[j]! = splitScalarResidueFoldInt zDecomp[j]! 2 k)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_checks_with_native_p6_base2
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

theorem p21FullMathTarget_of_checks_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreCheck bar a b = true)
  (hKPos : 0 < kSplit)
  (hEq : ∀ j (_hj : j < zDecomp.size),
    centeredInt zDecomp[j]! = splitScalarResidueFoldInt zDecomp[j]! 2 kSplit)
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
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 2 kSplit cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_checks_with_native_p6_base2
    (hP10 := hP10)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
      (z := zDecomp) (k := kSplit) hEq)
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
Concrete P21 assumption-surface constructor for `(b,k)=(2,8)` from challenge
decomposition rows.
-/
theorem p21ProtocolTarget_of_assumptions_with_native_p6_base2_k8_of_allChallenge
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 8 cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hKPos : 0 < (8 : Nat) := by decide
  have hZero : splitScalarTerminalZeroProp zDecomp 2 8 :=
    splitScalarTerminalZeroProp_of_allChallenge_base2_k8 (z := zDecomp) hChallenge
  have hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true :=
    allCanonical_of_allChallenge (z := zDecomp) hChallenge
  exact p21ProtocolTarget_of_assumptions_with_native_p6_base2
    (hKPos := hKPos)
    (hZero := hZero)
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

/-- Check-surface counterpart of the concrete `(b,k)=(2,8)` challenge closure. -/
theorem p21ProtocolTarget_of_checks_with_native_p6_base2_k8_of_allChallenge
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 8 cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hKPos : 0 < (8 : Nat) := by decide
  have hZero : splitScalarTerminalZeroProp zDecomp 2 8 :=
    splitScalarTerminalZeroProp_of_allChallenge_base2_k8 (z := zDecomp) hChallenge
  have hCanon : zDecomp.all (fun x => decide (F.Canonical x)) = true :=
    allCanonical_of_allChallenge (z := zDecomp) hChallenge
  exact p21ProtocolTarget_of_checks_with_native_p6_base2
    (hKPos := hKPos)
    (hZero := hZero)
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
P21 assumption-surface constructor from challenge decomposition rows for
base-2 and any `k ≥ 8`.
-/
theorem p21ProtocolTarget_of_assumptions_with_native_p6_base2_of_allChallenge_of_ge8
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_assumptions_with_native_p6_base2
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

/--
P21 check-surface constructor from challenge decomposition rows for base-2 and
any `k ≥ 8`.
-/
theorem p21ProtocolTarget_of_checks_with_native_p6_base2_of_allChallenge_of_ge8
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_checks_with_native_p6_base2
    (hKPos := Nat.lt_of_lt_of_le (by decide : 0 < 8) hk)
    (hZero := splitScalarTerminalZeroProp_of_allChallenge_base2_of_ge8
      (z := zDecomp) (k := k) hk hChallenge)
    (hCanon := allCanonical_of_allChallenge (z := zDecomp) hChallenge)
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

end SuperNeo
