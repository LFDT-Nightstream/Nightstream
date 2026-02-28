import SuperNeo.P20.ConstructorsExtra

/-! P20 check/prop bridges and check-driven constructor. -/

namespace SuperNeo

open F

/--
Bridge theorem: executable checks imply the proposition-native P20 bundle.
-/
theorem p20ArithmeticBundle_checks_imply_props
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
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Full : MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
    matrixTransformIdentity_sound_full hP12
  have hP15VecProp : p20VecModuleProp hVec ρ1 z1 z2 := by
    exact vecModulePropPair_of_checkPair (hCheck := ⟨hVecAdd, hVecScale⟩)
  have hP15ScalProp : p20ScalarModuleProp hScal ρ1 z1 z2 := by
    exact scalarModulePropPair_of_checkPair (hCheck := ⟨hScalAdd, hScalScale⟩)
  exact p20ArithmeticBundle_of_props
    (hP6 := p20DecompProp_of_splitRoundTrip hP6)
    (hP12Rows := hP12Full.1)
    (hP12Eq := hP12Full.2)
    (hP14 := evalHom2_sound_full hP14)
    (hP15Vec := hP15VecProp)
    (hP15Scal := hP15ScalProp)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := samplingSetBoundCheck_sound hP17)
    (hP18 := ⟨hP18Eq, (schwartzZippelBoundLeOne_sound hP18SZ).1, (schwartzZippelBoundLeOne_sound hP18SZ).2⟩)
    (hP19 := interpolationCase_sound hP19)

/--
Subset bridge in the opposite direction: proposition-level P20 assumptions imply
check-level obligations for P6/P17/P18/P19.
-/
theorem p20ArithmeticBundle_props_imply_check_subset
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
  splitRoundTrip zDecomp b k = true ∧
  matrixTransformIdentity bar m z = true ∧
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
  interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hP20 with ⟨hP6, hP12Rows, hP12Eq, hP14, _hP15Vec, _hP15Scal, _hP16, _hP16Win, hP17, hP18, hP19⟩
  rcases hP18 with ⟨hP18Eq, hSetNonzero, hDegBound⟩
  refine ⟨splitRoundTrip_of_p20DecompProp hP6, ?_, ?_, ?_, hP18Eq, ?_, ?_⟩
  · exact matrixTransformIdentity_complete_of_rowsCompatible hP12Rows hP12Eq
  · exact evalHom2_complete hP14
  · exact samplingSetBoundCheck_complete hP17
  · exact schwartzZippelBoundLeOne_complete hSetNonzero hDegBound
  · exact interpolationCase_complete hP19

/--
Additional proposition -> check bridge for P15 obligations, requiring the
size guard used by additivity checks.
-/
theorem p20ArithmeticBundle_props_imply_module_checks
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
  preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true := by
  rcases hP20 with ⟨_hP6, _hP12Rows, _hP12Eq, hP14, hP15Vec, hP15Scal, _hP16, _hP16Win, _hP17, _hP18, _hP19⟩
  have hSize : z1.size = z2.size := evalHom2Prop_size_eq hP14
  have hVecChecks : vecModuleCheckPair hVec ρ1 z1 z2 := by
    exact vecModuleCheckPair_of_propPair hSize hP15Vec
  have hScalChecks : scalarModuleCheckPair hScal ρ1 z1 z2 := by
    exact scalarModuleCheckPair_of_propPair hSize hP15Scal
  exact ⟨hVecChecks.1, hVecChecks.2, hScalChecks.1, hScalChecks.2⟩

/--
Full proposition -> check bridge for the check-side obligations used by the
backward-compatible P20 constructor.
-/
theorem p20ArithmeticBundle_props_imply_checks
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
  splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  have hSubset :
      splitRoundTrip zDecomp b k = true ∧
      matrixTransformIdentity bar m z = true ∧
      evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
      samplingSetBoundCheck cset samples = true ∧
      eqLiftAllBoolean qVals ell = true ∧
      schwartzZippelBoundLeOne totalDegree setSize = true ∧
      interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
    exact p20ArithmeticBundle_props_imply_check_subset hP20
  have hModule :
      preservesAddVec hVec z1 z2 = true ∧
      preservesScaleVec hVec ρ1 z1 = true ∧
      preservesAddScalar hScal z1 z2 = true ∧
      preservesScaleScalar hScal ρ1 z1 = true := by
    exact p20ArithmeticBundle_props_imply_module_checks hP20
  rcases hSubset with ⟨hSplit, hMat, hEvalHom, hSamp, hEq, hSZ, hInterp⟩
  rcases hModule with ⟨hVecAdd, hVecScale, hScalAdd, hScalScale⟩
  exact ⟨
    hSplit,
    hMat,
    hEvalHom,
    hVecAdd,
    hVecScale,
    hScalAdd,
    hScalScale,
    hSamp,
    hEq,
    hSZ,
    hInterp
  ⟩

/--
P20 check/prop equivalence for the concrete backward-compatible check surface.
-/
theorem p20ArithmeticBundle_iff_checks
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
  {ell totalDegree setSize : Nat} :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize ↔
    splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  constructor
  · intro hP20
    have hChecks :
        splitRoundTrip zDecomp b k = true ∧
        matrixTransformIdentity bar m z = true ∧
        evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
        preservesAddVec hVec z1 z2 = true ∧
        preservesScaleVec hVec ρ1 z1 = true ∧
        preservesAddScalar hScal z1 z2 = true ∧
        preservesScaleScalar hScal ρ1 z1 = true ∧
        samplingSetBoundCheck cset samples = true ∧
        eqLiftAllBoolean qVals ell = true ∧
        schwartzZippelBoundLeOne totalDegree setSize = true ∧
        interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
      exact p20ArithmeticBundle_props_imply_checks hP20
    rcases hP20 with ⟨_hP6, _hP12Rows, _hP12Eq, _hP14, _hP15Vec, _hP15Scal, _hP16, hInvWin, _hP17, _hP18, _hP19⟩
    rcases hChecks with
      ⟨hSplit, hMat, hEvalHom, hVecAdd, hVecScale, hScalAdd, hScalScale, hSamp, hEq, hSZ, hInterp⟩
    exact ⟨
      hSplit,
      hMat,
      hEvalHom,
      hVecAdd,
      hVecScale,
      hScalAdd,
      hScalScale,
      hInvWin,
      hSamp,
      hEq,
      hSZ,
      hInterp
    ⟩
  · intro hChecks
    rcases hChecks with
      ⟨hSplit, hMat, hEvalHom, hVecAdd, hVecScale, hScalAdd, hScalScale, hInvWin, hSamp, hEq, hSZ, hInterp⟩
    exact p20ArithmeticBundle_checks_imply_props
      hSplit hMat hEvalHom hVecAdd hVecScale hScalAdd hScalScale hInvWin hSamp hEq hSZ hInterp

/--
Backward-compatible check-driven constructor for P20.
-/
theorem p20ArithmeticBundle_of_checks
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
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_checks_imply_props
    hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP16Win hP17 hP18Eq hP18SZ hP19


end SuperNeo
