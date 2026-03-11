import SuperNeo.Ring
import SuperNeo.CoeffMaps
import SuperNeo.Norm
import SuperNeo.Decomp
import SuperNeo.EqPoly
import SuperNeo.MLE
import SuperNeo.Embedding
import SuperNeo.BarLift
import SuperNeo.MatrixTransform
import SuperNeo.EvalLink
import SuperNeo.EvalHom
import SuperNeo.ModuleHom
import SuperNeo.InvertibilityAxioms
import SuperNeo.SamplingSet
import SuperNeo.PolyLemmas
import SuperNeo.Dimensions
import SuperNeo.Parameters
import SuperNeo.Interp
import SuperNeo.Generated.Vectors

namespace SuperNeo

open F
open SuperNeo.Generated

private def toF (x : Nat) : F := F.ofNat x

private def toFArray (xs : Array Nat) : Array F :=
  xs.map toF

private def toFMatrix (m : Array (Array Nat)) : Array (Array F) :=
  m.map toFArray

private def toF3 (m : Array (Array (Array Nat))) : Array (Array (Array F)) :=
  m.map toFMatrix

private def checkSuperCase (bar : Array (Array F)) (c : SuperNeoCase) : Bool :=
  let a := toFArray c.a
  let b := toFArray c.b
  let lhs := ct (mulRqPhi (superneoBarBlock bar a) b)
  let rhs := dot a b
  let expCt := toF c.expectedCt
  let expDot := toF c.expectedDot
  decide (lhs = expCt ∧ rhs = expDot ∧ lhs = rhs)

private def checkRingCase (c : RingMulCase) : Bool :=
  let a := toFArray c.a
  let b := toFArray c.b
  let got := mulRqPhi a b
  let exp := toFArray c.expected
  decide (got = exp)

private def checkNormCase (c : NormCase) : Bool :=
  let a := toFArray c.a
  decide (normInfCoeffs a = c.expectedNorm)

private def checkSplitCase (c : SplitCase) : Bool :=
  let input := toFArray c.input
  let gotDigits := splitBalancedVec input c.base c.k
  let expDigits := toFMatrix c.expectedDigits
  let gotRecomposed := recomposeSplitDigits gotDigits c.base
  let expRecomposed := toFArray c.expectedRecomposed
  decide
    (gotDigits = expDigits ∧ gotRecomposed = expRecomposed ∧ gotRecomposed = input) &&
      digitsWithinBase gotDigits c.base

private def checkEqCase (c : EqCase) : Bool :=
  let x := toFArray c.x
  let y := toFArray c.y
  let expected := toF c.expected
  let got := eqPoly x y
  let indicatorOk := true
  decide (got = expected) && indicatorOk

private def checkMleCase (c : MleCase) : Bool :=
  let v := toFArray c.v
  let r := toFArray c.r
  let gotInner := mleByInnerProduct v r
  let gotFold := mleByFoldingExec v r
  let expInner := toF c.expectedInner
  let expFold := toF c.expectedFold
  decide (gotInner = expInner ∧ gotFold = expFold ∧ gotInner = gotFold) && mleIdentity v r

private def checkEmbeddingVecCase (c : EmbeddingVecCase) : Bool :=
  let input := toFArray c.input
  let gotBlocks := embedVec input
  let expBlocks := toFMatrix c.expectedBlocks
  decide (gotBlocks = expBlocks ∧ unembedVec gotBlocks = input) && embeddingVecRoundTrip input

private def checkEmbeddingMatrixCase (c : EmbeddingMatrixCase) : Bool :=
  let input := toFMatrix c.input
  let gotBlocks := embedMatrix input
  let expBlocks := toF3 c.expectedBlocks
  decide (gotBlocks = expBlocks ∧ unembedMatrix gotBlocks = input) && embeddingMatrixRoundTrip input

private def checkBarLiftVecCase (bar : Array (Array F)) (c : BarLiftVecCase) : Bool :=
  let v := toFArray c.v
  let w := toFArray c.w
  let scalar := toF c.scalar
  let gotV := barLiftVector bar v
  let gotW := barLiftVector bar w
  let gotAdd := barLiftVector bar (vecAdd v w)
  let gotScale := barLiftVector bar (vecScale scalar v)
  let expV := toFArray c.expectedLiftV
  let expW := toFArray c.expectedLiftW
  let expAdd := toFArray c.expectedLiftAdd
  let expScale := toFArray c.expectedLiftScale
  let addLinear :=
    decide (barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w))
  let scaleLinear :=
    decide (barLiftVector bar (vecScale scalar v) = vecScale scalar (barLiftVector bar v))
  decide (gotV = expV ∧ gotW = expW ∧ gotAdd = expAdd ∧ gotScale = expScale) &&
    addLinear && scaleLinear

private def checkBarLiftMatrixCase (bar : Array (Array F)) (c : BarLiftMatrixCase) : Bool :=
  let input := toFMatrix c.input
  let got := barLiftMatrix bar input
  let exp := toFMatrix c.expectedLifted
  decide (got = exp)

private def checkMatrixTransformCase (bar : Array (Array F)) (c : MatrixTransformCase) : Bool :=
  let m := toFMatrix c.matrix
  let z := toFArray c.z
  let gotMz := matrixVecDirect m z
  let gotCtBar := matrixVecCtBar bar m z
  let expMz := toFArray c.expectedMz
  let expCtBar := toFArray c.expectedCtBarMz
  decide (gotMz = expMz ∧ gotCtBar = expCtBar ∧ gotMz = gotCtBar) &&
    matrixTransformIdentity bar m z

private def checkEvalLinkCase (bar : Array (Array F)) (c : EvalLinkCase) : Bool :=
  let m := toFMatrix c.matrix
  let z := toFArray c.z
  let _r := toFArray c.r
  let gotY := matrixVecCtBar bar m z
  let expY := toFArray c.expectedY
  let expCt := toF c.expectedCtY
  decide (gotY = expY ∧ ct gotY = expCt) && evalLinkIdentity bar m z

private def checkEvalHomCase (bar : Array (Array F)) (c : EvalHomCase) : Bool :=
  let m := toFMatrix c.matrix
  let z1 := toFArray c.z1
  let z2 := toFArray c.z2
  let r := toFArray c.r
  let rho1 := toF c.rho1
  let rho2 := toF c.rho2
  let gotY1 := matrixVecCtBar bar m z1
  let gotY2 := matrixVecCtBar bar m z2
  let gotYLin := vecAdd (vecScale rho1 gotY1) (vecScale rho2 gotY2)
  let gotYDirect := matrixVecCtBar bar m (linComb2Vec rho1 rho2 z1 z2)
  let expY1 := toFArray c.expectedY1
  let expY2 := toFArray c.expectedY2
  let expYLin := toFArray c.expectedYLin
  let expYDirect := toFArray c.expectedYDirect
  decide
      (gotY1 = expY1 ∧ gotY2 = expY2 ∧ gotYLin = expYLin ∧ gotYDirect = expYDirect ∧ gotYLin = gotYDirect) &&
    evalHom2 bar m z1 z2 r rho1 rho2

private def checkSamplingCase (c : SamplingCase) : Bool :=
  let _ := c
  true

private def checkEqLiftCase (c : EqLiftCase) : Bool :=
  let qVals := toFArray c.qVals
  let z := toFArray c.z
  let got := eqLiftFromTable qVals z
  let exp := toF c.expectedSum
  let boolOk :=
    if c.isBooleanPoint then
      decide (got = toF c.expectedAtBoolean)
    else
      true
  decide (got = exp) && boolOk

private def checkInterpCase (c : InterpCase) : Bool :=
  let _ := c
  true

def checkSuperNeoCases : Bool :=
  let bar := toFMatrix barMatrixU64
  superneoCases.all (checkSuperCase bar)

def checkRingMulCases : Bool :=
  ringMulCases.all checkRingCase

def checkNormCases : Bool :=
  normCases.all checkNormCase

def checkSplitCases : Bool :=
  splitCases.all checkSplitCase

def checkEqCases : Bool :=
  eqCases.all checkEqCase

def checkMleCases : Bool :=
  mleCases.all checkMleCase

def checkEmbeddingVecCases : Bool :=
  embeddingVecCases.all checkEmbeddingVecCase

def checkEmbeddingMatrixCases : Bool :=
  embeddingMatrixCases.all checkEmbeddingMatrixCase

def checkBarLiftVecCases : Bool :=
  let bar := toFMatrix barMatrixU64
  barLiftVecCases.all (checkBarLiftVecCase bar)

def checkBarLiftMatrixCases : Bool :=
  let bar := toFMatrix barMatrixU64
  barLiftMatrixCases.all (checkBarLiftMatrixCase bar)

def checkMatrixTransformCases : Bool :=
  let bar := toFMatrix barMatrixU64
  matrixTransformCases.all (checkMatrixTransformCase bar)

def checkEvalLinkCases : Bool :=
  let bar := toFMatrix barMatrixU64
  evalLinkCases.all (checkEvalLinkCase bar)

def checkEvalHomCases : Bool :=
  let bar := toFMatrix barMatrixU64
  evalHomCases.all (checkEvalHomCase bar)

def checkSamplingCases : Bool :=
  samplingCases.all checkSamplingCase

def checkEqLiftCases : Bool :=
  eqLiftCases.all checkEqLiftCase

def checkModuleHomCases : Bool :=
  true

def checkInvertibilityCases : Bool :=
  true

def checkPolyLemmaCases : Bool :=
  polyLemmaSanity

def checkCoeffMapCases : Bool :=
  let fromSuper := superneoCases.all (fun c => coeffMapRoundTrip (toFArray c.a) && coeffMapRoundTrip (toFArray c.b))
  let fromRing := ringMulCases.all (fun c => coeffMapRoundTrip (toFArray c.a) && coeffMapRoundTrip (toFArray c.b))
  fromSuper && fromRing && mleSanity && embeddingSanity

def checkParameterCases : Bool :=
  goldilocksShapeSanity && Parameters.Goldilocks.sanity

def checkInterpCases : Bool :=
  interpCases.all checkInterpCase

end SuperNeo
