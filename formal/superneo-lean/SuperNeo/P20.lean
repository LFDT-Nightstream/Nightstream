import SuperNeo.MatrixTransform
import SuperNeo.EvalHom
import SuperNeo.ModuleHom
import SuperNeo.InvertibilityAxioms
import SuperNeo.SamplingSet
import SuperNeo.PolyLemmas
import SuperNeo.Decomp
import SuperNeo.Interp

namespace SuperNeo

open F

/-- P14 consequence packaged as a proposition (evaluation homomorphism equality). -/
def p20EvalHomProp
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  let y1 := evalBarMzAt bar m z1 r
  let y2 := evalBarMzAt bar m z2 r
  let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
  let yDirect := evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r
  yLin = yDirect ∧ ct yLin = ρ1 * ct y1 + ρ2 * ct y2

/-- P15 consequence packaged as vector-module linearity obligations. -/
def p20VecModuleProp (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧
    h.map (vecScale s x) = vecScale s (h.map x)

/-- P15 consequence packaged as scalar-module linearity obligations. -/
def p20ScalarModuleProp (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = h.map x + h.map y ∧
    h.map (vecScale s x) = s * h.map x

def p20SamplingProp (cset samples : Array Coeffs) : Prop :=
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)

def p20PolyProp (qVals : Array F) (ell totalDegree setSize : Nat) : Prop :=
  eqLiftAllBoolean qVals ell = true ∧ setSize ≠ 0 ∧ totalDegree <= setSize

def p20DecompProp (z : Array F) (b k : Nat) : Prop :=
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true

def p20InterpProp
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) : Prop :=
  let coeffs := interpolateFromEvals xs ys
  coeffs = expectedCoeffs ∧ polyEval coeffs evalPoint = expectedEval

/--
P20 arithmetic bundle: composition obligations for P6/P12/P14/P15 plus
invertibility/sampling/polynomial/interpolation side conditions (P16/P17/P18/P19).
-/
def p20ArithmeticBundle
  (bar : Array (Array F))
  (m : Array (Array F))
  (z z1 z2 zDecomp r : Array F)
  (ρ1 ρ2 : F)
  (b k : Nat)
  (hVec : VecModuleHom)
  (hScal : ScalarModuleHom)
  (cset samples : Array Coeffs)
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p20DecompProp zDecomp b k ∧
    MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z ∧
    p20EvalHomProp bar m z1 z2 r ρ1 ρ2 ∧
    p20VecModuleProp hVec ρ1 z1 z2 ∧
    p20ScalarModuleProp hScal ρ1 z1 z2 ∧
    invertibilityPreconditionsProp ∧
    p20SamplingProp cset samples ∧
    p20PolyProp qVals ell totalDegree setSize ∧
    p20InterpProp xs ys expectedCoeffs evalPoint expectedEval

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
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP6, hP12Rows, hP12Eq, hP14, hP15Vec, hP15Scal, hP16, hP17, hP18, hP19⟩

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
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Full : MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
    matrixTransformIdentity_sound_full hP12
  exact p20ArithmeticBundle_of_props
    (hP6 := splitRoundTrip_sound hP6)
    (hP12Rows := hP12Full.1)
    (hP12Eq := hP12Full.2)
    (hP14 := evalHom2_sound hP14)
    (hP15Vec := ⟨preservesAddVec_sound hVecAdd, preservesScaleVec_sound hVecScale⟩)
    (hP15Scal := ⟨preservesAddScalar_sound hScalAdd, preservesScaleScalar_sound hScalScale⟩)
    (hP16 := invertibilityPreconditions_from_constants)
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
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hP20 with ⟨hP6, hP12Rows, hP12Eq, _hP14, _hP15Vec, _hP15Scal, _hP16, hP17, hP18, hP19⟩
  rcases hP18 with ⟨hP18Eq, hSetNonzero, hDegBound⟩
  refine ⟨splitRoundTrip_complete hP6, ?_, ?_, hP18Eq, ?_, ?_⟩
  · exact matrixTransformIdentity_complete_of_rowsCompatible hP12Rows hP12Eq
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
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hSize : z1.size = z2.size)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true := by
  rcases hP20 with ⟨_hP6, _hP12Rows, _hP12Eq, _hP14, hP15Vec, hP15Scal, _hP16, _hP17, _hP18, _hP19⟩
  exact ⟨
    preservesAddVec_complete hSize hP15Vec.1,
    preservesScaleVec_complete hP15Vec.2,
    preservesAddScalar_complete hSize hP15Scal.1,
    preservesScaleScalar_complete hP15Scal.2
  ⟩

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
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_checks_imply_props
    hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP17 hP18Eq hP18SZ hP19

end SuperNeo
