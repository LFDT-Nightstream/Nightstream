import SuperNeo.Decomp
import SuperNeo.MatrixTransform
import SuperNeo.EvalHom
import SuperNeo.ModuleHom
import SuperNeo.InvertibilityAxioms
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
  (invDelta : Coeffs)
  (cset samples : Array Coeffs)
  (xs ys qVals coeffs : Array F)
  (xEval expectedEval : F) where
  splitTerminalZero : splitBase2TerminalZeroProp splitScalar kSplit
  evalHom : evalHomAssumption bar m r rho1 rho2
  vecModule : vecModuleAssumption hVec
  scalarModule : scalarModuleAssumption hScal
  invertibilityWindow : invertibilityWindowProp Goldilocks.halfQ invDelta
  sampling : samplingExpansionProp cset samples
  mleTableSize : qVals.size = (2 ^ r.size)
  mleIdentityAtR : mleEval qVals r = mleInnerProductForm qVals r
  interpolation : interpolationProp xs ys coeffs xEval expectedEval

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
  {invDelta : Coeffs}
  {cset samples : Array Coeffs}
  {xs ys qVals coeffs : Array F}
  {xEval expectedEval : F}
  (_h : ArithmeticObligations
    bar m r rho1 rho2
    hVec hScal
    splitScalar kSplit
    invDelta cset samples
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

end SuperNeo
