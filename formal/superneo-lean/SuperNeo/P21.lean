import SuperNeo.P20
import SuperNeo.Thm3Core

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

theorem p21ProtocolTarget_of_props_with_sampling_operand_assumptions
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
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_sampling_operand_assumptions
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hAddSub := hAddSub)
      (hSub := hSub)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_sampling_goldilocks_operand_assumptions
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_assumptions
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hCollapse := hCollapse)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hFieldOps := hFieldOps)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawInRange := hRawInRange)
      (hFieldOps := hFieldOps)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_thm3CoreAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_thm3CoreAssumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hThm3 := hThm3)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_thm3CoreAssumption_with_evalHom_assumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hThm3 := hThm3)
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

theorem p21ProtocolTarget_of_props_with_thm3CoreAssumption_with_evalHom_checkAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hThm3 := hThm3)
      (hP14Check := hP14Check)
      (hP14Size := hP14Size)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_evalHom_assumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
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

theorem p21ProtocolTarget_of_props_with_evalHom_checkAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_props_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21ProtocolTarget_of_props_with_matrixTransform_assumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_matrixTransform_assumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Assm := hP12Assm)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16 := invertibilityPreconditions_from_constants)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21ProtocolTarget_of_props_with_matrixTransform_checkAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_props_with_matrixTransform_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_assumption
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
  (hP6 : p20DecompProp zDecomp b k)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_props_with_matrixTransform_assumption_with_evalHom_assumption
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

theorem p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21ProtocolTarget_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21ProtocolTarget_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
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
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_p10_p20
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_props (hP10 := hP10) (hP21 := p21ProtocolTarget_of_p20 hP20)

theorem p21FullMathTarget_of_p10_props_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_props
    (hP10 := hP10)
    (hP21 := p21ProtocolTarget_of_props_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Eq := hP12Eq)
      (hP14Assm := hP14Assm)
      (hP14Size := hP14Size)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21FullMathTarget_of_p10_props_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_evalHom_assumption
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
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
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_props
    (hP10 := hP10)
    (hP21 := p21ProtocolTarget_of_props_with_matrixTransform_assumption_with_evalHom_assumption
      (hP6 := hP6)
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
      (hP19 := hP19))

theorem p21FullMathTarget_of_p10_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_props
    (hP10 := hP10)
    (hP21 := p21ProtocolTarget_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := hP12Rows)
      (hP12Check := hP12Check)
      (hP14Check := hP14Check)
      (hP14Size := hP14Size)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19))

theorem p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_p10_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_preconditions
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_p20 (p10Core_of_preconditions hBar ha hb hP10Check) hP20

theorem p21FullMathTarget_of_thm3_preconditions_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_evalHom_assumption
    (hP10 := p10Core_of_preconditions hBar ha hb hP10Check)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_preconditions_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_evalHom_checkAssumption
    (hP10 := p10Core_of_preconditions hBar ha hb hP10Check)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_preconditions_with_matrixTransform_assumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
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
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP10 := p10Core_of_preconditions hBar ha hb hP10Check)
    (hP6 := hP6)
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

theorem p21FullMathTarget_of_thm3_preconditions_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
    (hP10 := p10Core_of_preconditions hBar ha hb hP10Check)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Check := hP12Check)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_p20 (p10Core_of_preconditions_props hBar ha hb hThm3) hP20

theorem p21FullMathTarget_of_thm3_assumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_evalHom_assumption
    (hP10 := p10Core_of_preconditions_props hBar ha hb hThm3)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_assumption_with_evalHom_assumption_of_rows
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_thm3_assumption_with_evalHom_assumption
    (hBar := hBar)
    (ha := ha)
    (hb := hb)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3 hP12Rows)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_assumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_evalHom_checkAssumption
    (hP10 := p10Core_of_preconditions_props hBar ha hb hThm3)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_assumption_with_evalHom_checkAssumption_of_rows
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_thm3_assumption_with_evalHom_checkAssumption
    (hBar := hBar)
    (ha := ha)
    (hb := hb)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3 hP12Rows)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21FullMathTarget_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
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
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP10 := p10Core_of_preconditions_props hBar ha hb hThm3)
    (hP6 := hP6)
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

theorem p21FullMathTarget_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP6 : p20DecompProp zDecomp bSplit kSplit)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
    (hP10 := p10Core_of_preconditions_props hBar ha hb hThm3)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Check := hP12Check)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p21ProtocolTarget_of_check_subset
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
  (hP6 : splitRoundTrip zDecomp b k = true)
  (hP12 : matrixTransformIdentity bar m z = true)
  (hP14 : evalHom2 bar m z1 z2 r ρ1 ρ2 = true)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Full : MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
    matrixTransformIdentity_sound_full hP12
  have hSZ := schwartzZippelBoundLeOne_sound hP18SZ
  exact p21ProtocolTarget_of_props
    (hP6 := p20DecompProp_of_splitRoundTrip hP6)
    (hP12Rows := hP12Full.1)
    (hP12Eq := hP12Full.2)
    (hP14 := evalHom2_sound_full hP14)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := samplingSetBoundCheck_sound hP17)
    (hP18 := ⟨hP18Eq, hSZ.1, hSZ.2⟩)
    (hP19 := interpolationCase_sound hP19)

theorem p21ProtocolTarget_props_imply_check_subset
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
  splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hP21 with ⟨hP6, hP12Rows, hP12Eq, hP14, _hP16, hP16Win, hP17, hP18, hP19⟩
  rcases hP18 with ⟨hP18Eq, hSetNonzero, hDegBound⟩
  exact ⟨
    splitRoundTrip_of_p20DecompProp hP6,
    matrixTransformIdentity_complete_of_rowsCompatible hP12Rows hP12Eq,
    evalHom2_complete hP14,
    hP16Win,
    samplingSetBoundCheck_complete hP17,
    hP18Eq,
    schwartzZippelBoundLeOne_complete hSetNonzero hDegBound,
    interpolationCase_complete hP19
  ⟩

theorem p21ProtocolTarget_iff_check_subset
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
  {ell totalDegree setSize : Nat} :
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize ↔
    splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  constructor
  · exact p21ProtocolTarget_props_imply_check_subset
  · intro hChecks
    rcases hChecks with ⟨hP6, hP12, hP14, hP16Win, hP17, hP18Eq, hP18SZ, hP19⟩
    exact p21ProtocolTarget_of_check_subset
      (hP6 := hP6)
      (hP12 := hP12)
      (hP14 := hP14)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18Eq := hP18Eq)
      (hP18SZ := hP18SZ)
      (hP19 := hP19)

theorem p21ProtocolTarget_of_checks
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
  (hP6 : splitRoundTrip zDecomp b k = true)
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  let _ := hVecAdd
  let _ := hVecScale
  let _ := hScalAdd
  let _ := hScalScale
  exact p21ProtocolTarget_of_check_subset
    (hP6 := hP6)
    (hP12 := hP12)
    (hP14 := hP14)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18Eq := hP18Eq)
    (hP18SZ := hP18SZ)
    (hP19 := hP19)

theorem p21FullMathTarget_of_checks
  {bar : Array (Array F)}
  {a b : Array F}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {bSplit kSplit : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreCheck bar a b = true)
  (hP6 : splitRoundTrip zDecomp bSplit kSplit = true)
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
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_props
    (hP10 := p10CoreCheck_sound hP10)
    (hP21 := p21ProtocolTarget_of_checks
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
      (hP19 := hP19))

theorem p21FullMathTarget_props_imply_check_subset
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
  p10CoreCheck bar a b = true ∧
    splitRoundTrip zDecomp bSplit kSplit = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hFull with ⟨hP10, hP21⟩
  have hP21Checks :
      splitRoundTrip zDecomp bSplit kSplit = true ∧
      matrixTransformIdentity bar m z = true ∧
      evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
      p20InvertibilityWindowProp invDelta ∧
      samplingSetBoundCheck cset samples = true ∧
      eqLiftAllBoolean qVals ell = true ∧
      schwartzZippelBoundLeOne totalDegree setSize = true ∧
      interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
    exact p21ProtocolTarget_props_imply_check_subset hP21
  exact ⟨p10CoreCheck_complete hP10, hP21Checks⟩

theorem p21FullMathTarget_iff_check_subset
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
  {ell totalDegree setSize : Nat} :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize ↔
    p10CoreCheck bar a b = true ∧
    splitRoundTrip zDecomp bSplit kSplit = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  constructor
  · exact p21FullMathTarget_props_imply_check_subset
  · intro hChecks
    rcases hChecks with ⟨hP10, hP6, hP12, hP14, hP16Win, hP17, hP18Eq, hP18SZ, hP19⟩
    exact p21FullMathTarget_of_props
      (hP10 := p10CoreCheck_sound hP10)
      (hP21 := p21ProtocolTarget_of_check_subset
        (hP6 := hP6)
        (hP12 := hP12)
        (hP14 := hP14)
        (hP16Win := hP16Win)
        (hP17 := hP17)
        (hP18Eq := hP18Eq)
        (hP18SZ := hP18SZ)
        (hP19 := hP19))

end SuperNeo
