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
def arithmeticEvalHomProp
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  evalHom2Prop bar m z1 z2 r ρ1 ρ2

/-- P15 consequence packaged as vector-module linearity obligations. -/
def arithmeticVecModuleProp (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧
    h.map (vecScale s x) = vecScale s (h.map x)

/-- P15 consequence packaged as scalar-module linearity obligations. -/
def arithmeticScalarModuleProp (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = h.map x + h.map y ∧
    h.map (vecScale s x) = s * h.map x

def arithmeticSamplingProp (cset samples : Array Coeffs) : Prop :=
  samplingExpansionProp cset samples

def arithmeticPolyProp (qVals : Array F) (ell totalDegree setSize : Nat) : Prop :=
  eqLiftAllBoolean qVals ell = true ∧ setSize ≠ 0 ∧ totalDegree <= setSize

def arithmeticDecompProp (z : Array F) (b k : Nat) : Prop :=
  splitBalancedRoundTripProp z b k

/-- Canonical equivalence between the proposition surface and executable P6 check. -/
theorem arithmeticDecompProp_iff_splitRoundTrip_true
  {z : Array F} {b k : Nat} :
  arithmeticDecompProp z b k ↔ splitRoundTrip z b k = true := by
  unfold arithmeticDecompProp
  exact Iff.symm splitRoundTrip_eq_true_iff_prop

def arithmeticInterpProp
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) : Prop :=
  interpolationProp xs ys expectedCoeffs evalPoint expectedEval

/--
Arithmetic bundle: composition obligations for split/matrix/eval/module plus
invertibility/sampling/polynomial/interpolation side conditions.
-/
def arithmeticBundleProp
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
  arithmeticDecompProp zDecomp b k ∧
    MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z ∧
    arithmeticEvalHomProp bar m z1 z2 r ρ1 ρ2 ∧
    arithmeticVecModuleProp hVec ρ1 z1 z2 ∧
    arithmeticScalarModuleProp hScal ρ1 z1 z2 ∧
    invertibilityPreconditionsProp ∧
    arithmeticSamplingProp cset samples ∧
    arithmeticPolyProp qVals ell totalDegree setSize ∧
    arithmeticInterpProp xs ys expectedCoeffs evalPoint expectedEval

/--
Proposition-native constructor for the arithmetic bundle.
-/
theorem arithmeticBundleProp_of_props
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
  (hP6 : arithmeticDecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : arithmeticEvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : arithmeticVecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : arithmeticScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP17 : arithmeticSamplingProp cset samples)
  (hP18 : arithmeticPolyProp qVals ell totalDegree setSize)
  (hP19 : arithmeticInterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP6, hP12Rows, hP12Eq, hP14, hP15Vec, hP15Scal, hP16, hP17, hP18, hP19⟩

/--
Theorem-native constructor for `P20` from `P10` plus theorem boundaries.

This avoids check-driven plumbing for the P12/P13/P14 path:
- derive P12 from `thm3CoreAssumption`,
- derive P13 from P12,
- derive P14 from P13 + module-hom assumptions,
then assemble the arithmetic bundle proposition.
-/
theorem arithmeticBundleProp_of_theorem_stack
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
  (hP6 : arithmeticDecompProp zDecomp b k)
  (hRowsZ : MatrixRowsCompatible m z)
  (hRowsZ1 : MatrixRowsCompatible m z1)
  (hSize12 : z1.size = z2.size)
  (hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal)
  (hP16 : invertibilityPreconditionsProp)
  (hP17 : arithmeticSamplingProp cset samples)
  (hP18 : arithmeticPolyProp qVals ell totalDegree setSize)
  (hP19 : arithmeticInterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z :=
    (matrixTransformAssumption_of_p10 hThm3) z hRowsZ
  have hEvalHomAssm : evalHomAssumption bar m r ρ1 ρ2 :=
    evalHomAssumption_of_p10_and_moduleAssumptions
      (hThm3 := hThm3)
      (hVecAssm := hVecAssm) (hScalAssm := hScalAssm)
  have hP14 : arithmeticEvalHomProp bar m z1 z2 r ρ1 ρ2 :=
    hEvalHomAssm z1 z2 hSize12 hRowsZ1
  exact arithmeticBundleProp_of_props
    (hP6 := hP6)
    (hP12Rows := hRowsZ)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := ⟨hVecAssm.1 z1 z2, hVecAssm.2 ρ1 z1⟩)
    (hP15Scal := ⟨hScalAssm.1 z1 z2, hScalAssm.2 ρ1 z1⟩)
    (hP16 := hP16)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Native theorem-stack constructor for `P20`: canonical Theorem-3 closure path
without threading an explicit `hThm3`.
-/
theorem arithmeticBundleProp_of_native_theorem_stack
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
  (hP6 : arithmeticDecompProp zDecomp b k)
  (hRowsZ : MatrixRowsCompatible m z)
  (hRowsZ1 : MatrixRowsCompatible m z1)
  (hSize12 : z1.size = z2.size)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal)
  (hP16 : invertibilityPreconditionsProp)
  (hP17 : arithmeticSamplingProp cset samples)
  (hP18 : arithmeticPolyProp qVals ell totalDegree setSize)
  (hP19 : arithmeticInterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  arithmeticBundleProp nativeBarMatrix m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact arithmeticBundleProp_of_theorem_stack
    (bar := nativeBarMatrix)
    (m := m)
    (z := z) (z1 := z1) (z2 := z2) (zDecomp := zDecomp) (r := r)
    (ρ1 := ρ1) (ρ2 := ρ2)
    (b := b) (k := k)
    (hVec := hVec) (hScal := hScal)
    (cset := cset) (samples := samples)
    (qVals := qVals)
    (xs := xs) (ys := ys) (expectedCoeffs := expectedCoeffs)
    (evalPoint := evalPoint) (expectedEval := expectedEval)
    (ell := ell) (totalDegree := totalDegree) (setSize := setSize)
    hP6 hRowsZ hRowsZ1 hSize12
    thm3CoreAssumption_native
    hVecAssm hScalAssm hP16 hP17 hP18 hP19

/--
Bridge theorem: executable checks imply the proposition-native arithmetic bundle.
-/
theorem arithmeticBundleProp_checks_imply_props
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
  arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Full : MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
    matrixTransformIdentity_sound_full hP12
  exact arithmeticBundleProp_of_props
    (hP6 := splitRoundTrip_sound_prop hP6)
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
Subset bridge in the opposite direction: proposition-level arithmetic assumptions
imply check-level obligations for split/sampling/polynomial/interpolation checks.
-/
theorem arithmeticBundleProp_props_imply_check_subset
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
  (hP20 : arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hP20 with ⟨hP6, hP12Rows, hP12Eq, _hP14, _hP15Vec, _hP15Scal, _hP16, hP17, hP18, hP19⟩
  rcases hP18 with ⟨hP18Eq, hSetNonzero, hDegBound⟩
  refine ⟨splitRoundTrip_complete_prop hP6, ?_, ?_, hP18Eq, ?_, ?_⟩
  · exact matrixTransformIdentity_complete_of_rowsCompatible hP12Rows hP12Eq
  · exact samplingSetBoundCheck_complete hP17
  · exact schwartzZippelBoundLeOne_complete hSetNonzero hDegBound
  · exact interpolationCase_complete hP19

/--
Additional proposition -> check bridge for P15 obligations, requiring the
size guard used by additivity checks.
-/
theorem arithmeticBundleProp_props_imply_module_checks
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
  (hP20 : arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
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
Check-driven constructor for the arithmetic bundle.
-/
theorem arithmeticBundleProp_of_checks
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
  arithmeticBundleProp bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact arithmeticBundleProp_checks_imply_props
    hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP17 hP18Eq hP18SZ hP19

end SuperNeo
