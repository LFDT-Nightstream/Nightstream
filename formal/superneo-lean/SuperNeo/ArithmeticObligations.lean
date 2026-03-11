import SuperNeo.Decomp
import SuperNeo.MatrixTransform
import SuperNeo.EvalHom
import SuperNeo.ModuleHom
import SuperNeo.SamplingSet
import SuperNeo.MLE
import SuperNeo.Interp

/-!
Arithmetic obligation bundle used by protocol composition.

This file contains only typed theorem-facing obligations that protocol layers
consume directly.
-/

namespace SuperNeo

/-- Arithmetic obligations required before entering protocol reductions. -/
structure ArithmeticObligations
  (bar m : Array (Array F))
  (r : Array F)
  (rho1 rho2 : F)
  (hVec : VecModuleHom)
  (hScal : ScalarModuleHom)
  (splitScalar : F)
  (kSplit : Nat)
  (cset samples : Array Coeffs)
  (xs ys qVals coeffs : Array F)
  (xEval expectedEval : F) where
  splitScalarBelowPow : splitScalar.val < 2 ^ kSplit
  evalHom : evalHomAssumption bar m r rho1 rho2
  vecModule : vecModuleAssumption hVec
  scalarModule : scalarModuleAssumption hScal
  sampling : samplingExpansionProp cset samples
  mleTableSize : qVals.size = (2 ^ r.size)
  mleIdentityAtR : mleEval qVals r = mleInnerProductForm qVals r
  interpolation : interpolationProp xs ys coeffs xEval expectedEval

/--
Theorem-native constructor for arithmetic obligations.

This uses the constructive eval-hom closure directly and derives the local MLE
identity from the theorem-native size theorem.
-/
def ArithmeticObligations.of_constructive
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (hSplit : splitScalar.val < 2 ^ kSplit)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal)
  (hSampling : samplingExpansionProp cset samples)
  (hMleSize : qVals.size = (2 ^ r.size))
  (hInterp : interpolationProp xs ys coeffs xEval expectedEval) :
  ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval where
  splitScalarBelowPow := hSplit
  evalHom := evalHomAssumption_constructive
  vecModule := hVecAssm
  scalarModule := hScalAssm
  sampling := hSampling
  mleTableSize := hMleSize
  mleIdentityAtR := by
    exact mleIdentityAssumption_holds qVals r hMleSize
  interpolation := hInterp

/--
Compatibility constructor deriving arithmetic obligations from `P10`.

This keeps the historical dependency shape while routing `evalHom` through the
constructive theorem-native closure.
-/
def ArithmeticObligations.of_p10
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (hSplit : splitScalar.val < 2 ^ kSplit)
  (_hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal)
  (hSampling : samplingExpansionProp cset samples)
  (hMleSize : qVals.size = (2 ^ r.size))
  (hInterp : interpolationProp xs ys coeffs xEval expectedEval) :
  ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval := by
  exact ArithmeticObligations.of_constructive
    (hSplit := hSplit)
    (hVecAssm := hVecAssm)
    (hScalAssm := hScalAssm)
    (hSampling := hSampling)
    (hMleSize := hMleSize)
    (hInterp := hInterp)

/--
Compatibility constructor keeping the historical `(P10 + P11)` signature.
-/
def ArithmeticObligations.of_p10_p11
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (hSplit : splitScalar.val < 2 ^ kSplit)
  (hThm3 : thm3CoreAssumption bar)
  (_hLift : barLiftLinearityAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal)
  (hSampling : samplingExpansionProp cset samples)
  (hMleSize : qVals.size = (2 ^ r.size))
  (hInterp : interpolationProp xs ys coeffs xEval expectedEval) :
  ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval := by
  exact ArithmeticObligations.of_p10
    hSplit
    hThm3
    hVecAssm
    hScalAssm
    hSampling
    hMleSize
    hInterp

/--
Compatibility accessor: recover the terminal-quotient-zero obligation from the
stored scalar bound `splitScalar.val < 2^kSplit`.
-/
theorem ArithmeticObligations.splitTerminalZero
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (h : ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval) :
  splitBase2TerminalZeroProp splitScalar kSplit :=
  splitBase2TerminalZeroProp_of_val_lt_pow splitScalar kSplit h.splitScalarBelowPow

/--
The scalar split decomposition identity is derivable directly from definitions,
so it is intentionally not stored as an explicit assumption field.
-/
theorem splitDecompositionNat_of_obligations
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (_h : ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval) :
  splitBase2LowPartNat splitScalar kSplit +
    (2 ^ kSplit) * splitBase2TerminalQuot splitScalar kSplit = splitScalar.val := by
  exact splitBase2DecompositionNat splitScalar kSplit

/-- Build the local MLE identity obligation from the global theorem surface. -/
theorem mleIdentityAtR_of_assumption
  {qVals r : Array F}
  (hSize : qVals.size = (2 ^ r.size))
  (hMLE : mleIdentityAssumption) :
  mleEval qVals r = mleInnerProductForm qVals r := by
  exact hMLE qVals r hSize

/-- Preferred constructor: local MLE identity follows directly from the proved MLE theorem. -/
theorem mleIdentityAtR_of_size
  {qVals r : Array F}
  (hSize : qVals.size = (2 ^ r.size)) :
  mleEval qVals r = mleInnerProductForm qVals r := by
  exact mleIdentityAssumption_holds qVals r hSize

/-- The constructive interpolation checker succeeds on the stored interpolation obligation. -/
theorem ArithmeticObligations.interpolationCase_eq_true
  {bar m : Array (Array F)}
  {r : Array F}
  {rho1 rho2 : F}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {splitScalar : F}
  {kSplit : Nat}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (h : ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    cset samples
    xs ys qVals coeffs
    xEval expectedEval) :
  interpolationCase xs ys coeffs xEval expectedEval = true := by
  exact interpolationCase_complete h.interpolation

end SuperNeo
