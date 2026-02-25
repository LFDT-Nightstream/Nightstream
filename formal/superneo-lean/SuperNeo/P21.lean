import SuperNeo.P20
import SuperNeo.Thm3Core

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
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p20DecompProp zDecomp b k ∧
    MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z ∧
    p20EvalHomProp bar m z1 z2 r ρ1 ρ2 ∧
    invertibilityPreconditionsProp ∧
    p20SamplingProp cset samples ∧
    p20PolyProp qVals ell totalDegree setSize ∧
    p20InterpProp xs ys expectedCoeffs evalPoint expectedEval

def p21FullMathTarget
  (bar : Array (Array F))
  (a b : Array F)
  (m : Array (Array F))
  (z z1 z2 zDecomp r : Array F)
  (ρ1 ρ2 : F)
  (bSplit kSplit : Nat)
  (cset samples : Array Coeffs)
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p10CoreProp bar a b ∧
    p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize

theorem p21ProtocolTarget_of_p20
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  rcases hP20 with ⟨hDecomp, hRows, hMat, hEval, _hVec, _hScal, hInv, hSamp, hPoly, hInterp⟩
  exact ⟨hDecomp, hRows, hMat, hEval, hInv, hSamp, hPoly, hInterp⟩

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
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP10 : p10CoreProp bar a b)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP10, p21ProtocolTarget_of_p20 hP20⟩

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
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hP10Check : p10CoreCheck bar a b = true)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_p20 (p10Core_of_preconditions hBar ha hb hP10Check) hP20

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
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit hVec hScal cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21FullMathTarget_of_p10_p20 (p10Core_of_preconditions_props hBar ha hb hThm3) hP20

theorem p21ProtocolTarget_of_checks
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
  p21ProtocolTarget bar m z z1 z2 zDecomp r ρ1 ρ2 b k cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p21ProtocolTarget_of_p20
    (p20ArithmeticBundle_of_checks
      (bar := bar) (m := m)
      (z := z) (z1 := z1) (z2 := z2) (zDecomp := zDecomp) (r := r)
      (ρ1 := ρ1) (ρ2 := ρ2)
      (b := b) (k := k)
      (hVec := hVec) (hScal := hScal)
      (cset := cset) (samples := samples)
      (qVals := qVals)
      (xs := xs) (ys := ys) (expectedCoeffs := expectedCoeffs)
      (evalPoint := evalPoint) (expectedEval := expectedEval)
      (ell := ell) (totalDegree := totalDegree) (setSize := setSize)
      hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP17 hP18Eq hP18SZ hP19)

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
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p21FullMathTarget bar a b m z z1 z2 zDecomp r ρ1 ρ2 bSplit kSplit cset samples qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  refine ⟨p10CoreCheck_sound hP10, ?_⟩
  exact p21ProtocolTarget_of_checks
    (bar := bar) (m := m)
    (z := z) (z1 := z1) (z2 := z2) (zDecomp := zDecomp) (r := r)
    (ρ1 := ρ1) (ρ2 := ρ2)
    (b := bSplit) (k := kSplit)
    (hVec := hVec) (hScal := hScal)
    (cset := cset) (samples := samples)
    (qVals := qVals)
    (xs := xs) (ys := ys) (expectedCoeffs := expectedCoeffs)
    (evalPoint := evalPoint) (expectedEval := expectedEval)
    (ell := ell) (totalDegree := totalDegree) (setSize := setSize)
    hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP17 hP18Eq hP18SZ hP19

end SuperNeo
