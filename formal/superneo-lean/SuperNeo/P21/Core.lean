import SuperNeo.P20
import SuperNeo.Thm3Core
import SuperNeo.DecompNative

/-! Protocol-level composition target built from P10 and P20. -/


namespace SuperNeo

open F

/--
First protocol-facing target extracted from the arithmetic P20 bundle.
This is not the full SuperNeo theorem yet; it is the bridge interface that
the eventual protocol proof can consume.
-/
def p21ProtocolTarget
  (bar : Array (Array F))
  (m : Array (Array F))
  (z z1 z2 zDecomp r : Array F)
  (ρ1 ρ2 : F)
  (b k : Nat)
  (cset samples : Array Coeffs)
  (invDelta : Coeffs)
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p20DecompProp zDecomp b k ∧
    MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z ∧
    p20EvalHomProp bar m z1 z2 r ρ1 ρ2 ∧
    invertibilityPreconditionsProp ∧
    p20InvertibilityWindowProp invDelta ∧
    p20SamplingProp cset samples ∧
    p20PolyProp qVals ell totalDegree setSize ∧
    p20InterpProp xs ys expectedCoeffs evalPoint expectedEval

/-- P21 full target = P10 core proposition plus the protocol target bundle. -/
def p21FullMathTarget
  (bar : Array (Array F))
  (a b : Array F)
  (m : Array (Array F))
  (z z1 z2 zDecomp r : Array F)
  (ρ1 ρ2 : F)
  (bSplit kSplit : Nat)
  (cset samples : Array Coeffs)
  (invDelta : Coeffs)
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p10CoreProp bar a b ∧
    p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize

/-- Primary constructor for `p21ProtocolTarget`: theorem-native conjuncts. -/
theorem p21ProtocolTarget_of_props
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta
    qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP6, hP12Rows, hP12Eq, hP14, hP16, hP16Win, hP17, hP18, hP19⟩

/-- Primary constructor for `p21FullMathTarget`: theorem-native `P10` + protocol target. -/
theorem p21FullMathTarget_of_props
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP21 :
    p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta
      qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta
    qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP10, hP21⟩

theorem p21ProtocolTarget_decomp
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP21 : p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p20DecompProp zDecomp b k := by
  exact hP21.1

theorem p21ProtocolTarget_decomp_digit_row_size
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k i : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP21 : p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hi : i < (splitBalancedVec zDecomp b k).size) :
  ((splitBalancedVec zDecomp b k)[i]'hi).size = zDecomp.size := by
  exact p20DecompProp_digit_row_size (p21ProtocolTarget_decomp hP21) hi

theorem p21ProtocolTarget_decomp_recompose_eq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP21 : p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  recomposeSplitDigits (splitBalancedVec zDecomp b k) b = zDecomp := by
  exact p20DecompProp_recompose_eq (p21ProtocolTarget_decomp hP21)

theorem p21ProtocolTarget_decomp_digit_bound
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k i j : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP21 : p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hi : i < (splitBalancedVec zDecomp b k).size)
  (hj : j < ((splitBalancedVec zDecomp b k)[i]'hi).size) :
  normInfF (((splitBalancedVec zDecomp b k)[i]'hi)[j]'hj) < b := by
  exact p20DecompProp_digit_bound (p21ProtocolTarget_decomp hP21) hi hj

theorem p21FullMathTarget_protocol
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hFull : p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact hFull.2

theorem p21FullMathTarget_decomp_digit_row_size
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit i : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hFull : p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hi : i < (splitBalancedVec zDecomp bSplit kSplit).size) :
  ((splitBalancedVec zDecomp bSplit kSplit)[i]'hi).size = zDecomp.size := by
  exact p21ProtocolTarget_decomp_digit_row_size (p21FullMathTarget_protocol hFull) hi

theorem p21FullMathTarget_decomp_recompose_eq
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hFull : p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  recomposeSplitDigits (splitBalancedVec zDecomp bSplit kSplit) bSplit = zDecomp := by
  exact p21ProtocolTarget_decomp_recompose_eq (p21FullMathTarget_protocol hFull)

theorem p21FullMathTarget_decomp_digit_bound
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit i j : Nat}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hFull : p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hi : i < (splitBalancedVec zDecomp bSplit kSplit).size)
  (hj : j < ((splitBalancedVec zDecomp bSplit kSplit)[i]'hi).size) :
  normInfF (((splitBalancedVec zDecomp bSplit kSplit)[i]'hi)[j]'hj) < bSplit := by
  exact p21ProtocolTarget_decomp_digit_bound (p21FullMathTarget_protocol hFull) hi hj

theorem p21ProtocolTarget_of_p20
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
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  rcases hP20 with ⟨hDecomp, hRows, hMat, hEval, _hVec, _hScal, hInv, hInvWin, hSamp, hPoly, hInterp⟩
  exact p21ProtocolTarget_of_props
    (hP6 := hDecomp)
    (hP12Rows := hRows)
    (hP12Eq := hMat)
    (hP14 := hEval)
    (hP16 := hInv)
    (hP16Win := hInvWin)
    (hP17 := hSamp)
    (hP18 := hPoly)
    (hP19 := hInterp)

theorem p21ProtocolTarget_of_assumptions_with_p6DecompAssumption
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
  (hP6Assm : p6DecompAssumption b k)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_assumptions_with_p6DecompAssumption
      (hP6Assm := hP6Assm)
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
      (hP19 := hP19))

theorem p21ProtocolTarget_of_assumptions_with_p6DecompCheckAssumption
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
  (hP6Check : p6DecompCheckAssumption b k)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_assumptions_with_p6DecompCheckAssumption
      (hP6Check := hP6Check)
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
      (hP19 := hP19))

theorem p21ProtocolTarget_of_assumptions_with_native_p6_base2
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 2 k cset samples invDelta qVals
    xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP6 : p20DecompProp zDecomp 2 k := by
    exact p20DecompProp_of_splitRoundTrip
      (z := zDecomp) (b := 2) (k := k)
      (splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
        (z := zDecomp) (k := k) hKPos hZero hCanon)
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_assumptions
      (hP6 := hP6)
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
      (hP19 := hP19))


end SuperNeo
