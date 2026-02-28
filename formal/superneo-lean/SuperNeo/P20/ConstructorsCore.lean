import SuperNeo.P20.Spec

/-! P20 theorem-native and compatibility constructors (core segment). -/

namespace SuperNeo

open F

/--
Proposition-native constructor for the P20 bundle.
-/
theorem p20ArithmeticBundle_of_props
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP6, hP12Rows, hP12Eq, hP14, hP15Vec, hP15Scal, hP16, hP16Win, hP17, hP18, hP19⟩

/--
Theorem-native core constructor for the P20 bundle.

This is the single intended "theorem-first" entry point:
* P12 is supplied via `p12MatrixTransformAssumption`.
* P14 is supplied via `p14EvalHomAssumption`.

All other P20 constructors in this file are compatibility wrappers that either
package already-proved equalities/props or convert check-style assumptions into
these theorem-style interfaces.
-/
theorem p20ArithmeticBundle_of_assumptions
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals
      xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := p20MatrixTransformProp_of_assumption hP12Assm hP12Rows)
    (hP14 := p20EvalHomProp_of_assumption hP14Assm hP14Size hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_assumptions_with_p6DecompAssumption
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals
      xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := p20DecompProp_of_p6DecompAssumption (z := zDecomp) hP6Assm)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_assumptions_with_p6DecompCheckAssumption
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals
      xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions_with_p6DecompAssumption
    (hP6Assm := p6DecompAssumption_of_checkAssumption hP6Check)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions hCset hSamples hRaw hAddSub hSub)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_fieldOp_assumptions
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_fieldOp hCset hSamples hRaw hOps)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_fieldOp_assumptions_inRange
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange :
    mulRqRawInRangeBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_fieldOp_inRange hCset hSamples hRawInRange hOps)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hAddTri hSubTri hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_tight
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hAddTri hSubTri hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Triangle-bundle variant of
`p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight`.
-/
theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hTri hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := hMulUniv)
    (hTri := schoolbookTriangleBounds_of_add hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (hAddTri := schoolbookAddTriangleBound_of_centeredRep hAddRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulRep := centeredRepMulAddBounds_mul hRep)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Assumption-free native P20 bundle constructor.
Uses theorem-native P5 blockers proved in `Norm.lean`.
-/
theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_native
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := schoolbookMulUniversalBound_theorem)
    (hAddTri := schoolbookTriangleBounds_add schoolbookTriangleBounds_theorem)
    (hSubTri := schoolbookTriangleBounds_sub schoolbookTriangleBounds_theorem)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Assumption-free native-tight P20 bundle constructor.
Uses theorem-native P5 blockers proved in `Norm.lean`.
-/
theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_native_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := schoolbookMulUniversalBound_theorem)
    (hTri := schoolbookTriangleBounds_theorem)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_tight
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
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulRep := hMulRep)
    (hAddRep := centeredRepAddTriangleBound_theorem)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_and_raw
      (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
      hCset hSamples hRaw hAddTri hSubTri hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_rawCoeff
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddTri := hAddTri)
    (hSubTri := schoolbookSubTriangleBound_of_add hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_rawCoeff
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddTri := hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
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
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddTri := schoolbookAddTriangleBound_of_centeredRep hAddRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)


end SuperNeo
