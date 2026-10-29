import CompressionLean.Circuit.Early

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Relation foldVars runOracles generator twoAdicGenerator)

def setupPoint (point : Array ExtVar) (vars : ℕ) : Build (Array ExtVar) := do
  let zero ← Algebra.extZero
  pure (point.extract 0 vars ++ Array.replicate (vars - point.size) zero)

def combinePartials (partials point : Array ExtVar) : Build ExtVar := do
  let eq ← Algebra.eqTable point
  Algebra.dot eq partials

def setupCheck (root : DigestVar) (vars : ℕ) (bits : Array Var) (leaf : LeafView)
    (groups : Array (Array ExtVar)) (alpha : Array ExtVar) : Build (Array ExtVar) := do
  let leafSize := 2 ^ foldVars
  let digest ← hashLeaf leaf.values
  let computed ← merkleRoot digest bits leaf.path
  for lane in List.finRange 4 do
    Build.assertEqual (computed lane) (root lane)
  let rotation ← Algebra.powerFromBits (twoAdicGenerator (vars + 1)).inv bits
  let yInverse ← Build.scale rotation generator.inv
  let mut monomials := #[← Algebra.extOne]
  for a in alpha do
    let mut scaled := #[]
    for m in monomials do
      scaled := scaled.push (← Build.extMul m a)
    monomials := monomials ++ scaled
  let mut scale ← Build.const (Gl.ofNat leafSize).inv
  let mut coefficients := #[]
  for monomial in monomials do
    coefficients := coefficients.push (← Build.extScale monomial scale)
    scale ← Build.mul scale yInverse
  let omega := twoAdicGenerator foldVars
  let mut weight := #[]
  for v in [0:leafSize] do
    let mut total ← Algebra.extZero
    for (coefficient, u) in coefficients.zipIdx do
      let term ← Algebra.extScaleConstant coefficient (omega.pow ((leafSize - u) * v % leafSize))
      total ← Build.extAdd total term
    weight := weight.push total
  let mut oracle := 0
  let mut folded := #[]
  for weights in groups do
    let zero ← Algebra.extZero
    let mut combined := Array.replicate leafSize zero
    for w in weights do
      let values := leaf.values.extract (leafSize * oracle) (leafSize * oracle + leafSize)
      oracle := oracle + 1
      for (value, v) in values.zipIdx do
        let term ← Build.extScale w value
        combined := combined.set! v (← Build.extAdd combined[v]! term)
    folded := folded.push (← Algebra.dot weight combined)
  pure folded

def queryPoint (vars : ℕ) (bits : Array Var) : Build (Array ExtVar) := do
  let rotation ← Algebra.powerFromBits (twoAdicGenerator (vars + 1 - foldVars)) bits
  let mut q ← Build.scale rotation (generator.pow (2 ^ foldVars))
  let mut point := #[]
  for _ in [0:vars - foldVars] do
    point := point.push (← Algebra.lift q)
    q ← Build.mul q q
  pure point

structure HonestInput where
  lanePoint : Array ExtVar
  blockPoint : Array ExtVar
  runPoint : Array ExtVar
  setup : ExtVar × ExtVar × ExtVar
  block : ExtVar
  commitmentRows : Array ExtVar
  keyValue : ExtVar
  partials : Array (Array ExtVar)
  p1 : DigestVar
  leaves : Array LeafView

structure HonestOut where
  alpha : Array ExtVar
  keyPoint : Array ExtVar
  runPoint : Array ExtVar
  indexBits : Array (Array Var)
  queried : Array ExtVar

def honestStage (rel : Relation) (input : HonestInput) : Transcript HonestOut := do
  let mu ← Duplex.sampleExt
  for values in input.partials do
    for value in values do
      Duplex.observeExt value
  let alpha ← (Array.range foldVars).mapM fun _ => Duplex.sampleExt
  let vars := rel.setupVars
  let keyPoint ← setupPoint (input.lanePoint ++ input.blockPoint) vars
  let runPoint ← setupPoint input.runPoint vars
  let (row, slot, coefficient) := input.setup
  let mut embedded := input.block
  for x in input.runPoint.extract rel.key.layout.slotVariables input.runPoint.size do
    let low ← Algebra.oneMinus x
    embedded ← Build.extMul embedded low
  let mut runValue := embedded
  for value in #[coefficient, slot, row] do
    let scaled ← Build.extMul mu runValue
    runValue ← Build.extAdd value scaled
  let keyPartial ← combinePartials input.partials[0]! (keyPoint.extract 0 foldVars)
  Build.assertExtEqual keyPartial input.keyValue
  let runPartial ← combinePartials input.partials[1]! (runPoint.extract 0 foldVars)
  Build.assertExtEqual runPartial runValue
  observeRoot rel.commitmentSeed input.p1
  let indexBits ← (Array.range rel.queries).mapM fun _ => Duplex.sampleBits (vars + 1 - foldVars)
  let mut runWeights := #[← Algebra.extOne]
  for _ in [1:runOracles] do
    runWeights := runWeights.push (← Build.extMul runWeights.back! mu)
  let mut root := #[]
  for word in rel.key.root do
    root := root.push (← Build.const word)
  let mut queried := #[]
  for (bits, leaf) in indexBits.zip input.leaves do
    queried := queried ++ (← setupCheck (fun lane => root[lane]!) vars bits leaf
      #[input.commitmentRows, runWeights] alpha)
  pure { alpha, keyPoint, runPoint, indexBits, queried }

end CompressionLean.Circuit
