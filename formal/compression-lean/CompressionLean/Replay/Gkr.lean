import CompressionLean.Replay.Formulas

/-!
Owns the sum-check replay (`sumcheck::replay`) and the batched fraction-sum
GKR verifier (`gkr::verify`, `GkrView::read`), with the model's names
(`Layer1/Gkr.lean`, `Layer1/GkrLayer.lean`): `GkrClaims`, `combineVals`,
`reduce`. A round message is `h(0), h(2), …, h(degree)`; `h(1)` is the claim
minus `h(0)`.
-/

namespace CompressionLean.Replay

/-- Replay `rounds` against `claim`; return the point and the final claim. -/
def sumcheckReplay (rounds : Array (Array Ext)) (claim : Ext) : ReplayM (Array Ext × Ext) := do
  let mut point := #[]
  let mut claim := claim
  for message in rounds do
    let values := #[message[0]!, claim - message[0]!] ++ message.extract 1 message.size
    message.forM observeExt
    let r ← sampleExt
    claim := interpolate values r
    point := point.push r
  return (point, claim)

/-- The verifier's claims during the batched GKR (the model's `GkrClaims`). -/
structure GkrClaims where
  roots : Array (Ext × Ext)
  point : Array Ext
  current : Array (Ext × Ext)
  leafPoint : Array (Array Ext)
  leafValues : Array (Array Ext)

/-- Part `j` of tree `t` at its leaf point. -/
def GkrClaims.leafValue (claims : GkrClaims) (t j : Nat) : Ext := (claims.leafValues[t]!)[j]!

/-- Factors of each tree that reaches layer `k`: one inside, its own at its leaves. -/
def layerFactors (shapes : Array TreeShape) (k : Nat) : Array Nat :=
  (shapes.filter (k ≤ ·.depth)).map fun s => if s.depth = k then s.factors else 1

/-- The batched round degree: 3, or `factors + 2` at a product leaf layer. -/
def layerDegree (factors : Array Nat) : Nat := max 3 (factors.foldl (fun d f => max d (f + 2)) 0)

def layerCount (shapes : Array TreeShape) : Nat := shapes.foldl (fun d s => max d s.depth) 0

/-- The proof has the shapes' layers, rounds and children (`GkrView::read`). -/
def readGkr (proof : GkrProof) (shapes : Array TreeShape) : Except String Unit := do
  if shapes.any (fun s => s.depth == 0 || s.factors > 2) || proof.roots.size != shapes.size ||
      proof.layers.size != layerCount shapes then
    throw "GKR shape"
  for k in [1:layerCount shapes + 1] do
    let factors := layerFactors shapes k
    let layer := proof.layers[k - 1]!
    if layer.rounds.size != k - 1 || layer.rounds.any (·.size != layerDegree factors) ||
        layer.children.size != factors.size ||
        (layer.children.zip factors).any (fun pair => pair.1.size != 2 * (pair.2 + 1)) then
      throw "GKR layer shape"

/-- `p₀·q₁ + p₁·q₀ + μ·q₀·q₁` from one tree's part pairs: factors, then the
denominator (`combine`). -/
def combineVals (children : Array Ext) (mu : Ext) : Ext :=
  let f := children.size / 2 - 1
  let p (side : Nat) : Ext := (List.range f).foldl (fun total j => total * children[2 * j + side]!) 1
  let q0 := children[2 * f]!
  let q1 := children[2 * f + 1]!
  p 0 * q1 + p 1 * q0 + mu * (q0 * q1)

/-- One pair reduced at the top variable. -/
def reduce (pair : Ext × Ext) (tau : Ext) : Ext := pair.1 + tau * (pair.2 - pair.1)

/-- Verify the batched GKR for trees of `shapes`: the roots, then every layer's
`μ`, `ν`, sum-check, check, children and `τ`. -/
def gkrVerify (proof : GkrProof) (shapes : Array TreeShape) : ReplayM GkrClaims := do
  for root in proof.roots do
    observeExt root.1
    observeExt root.2
  let mut claims : GkrClaims :=
    { roots := proof.roots, point := #[], current := proof.roots,
      leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] }
  for index in [0:proof.layers.size] do
    let k := index + 1
    let layer := proof.layers[index]!
    let active := (Array.range shapes.size).filter (k ≤ shapes[·]!.depth)
    let mu ← sampleExt
    let nu ← sampleExt
    let (claim, expected, _) := (active.zip layer.children).foldl
      (fun (acc : Ext × Ext × Ext) pair =>
        let current := claims.current[pair.1]!
        (acc.1 + acc.2.2 * (current.1 + mu * current.2),
          acc.2.1 + acc.2.2 * combineVals pair.2 mu, acc.2.2 * nu))
      (0, 0, 1)
    let (next, last) ← sumcheckReplay layer.rounds claim
    ensure (last == eqEval claims.point next * expected) "GKR layer"
    for values in layer.children do
      values.forM observeExt
    let tau ← sampleExt
    let next := next.push tau
    for pair in active.zip layer.children do
      let values := pair.2
      let reduced := (Array.range (values.size / 2)).map fun j => reduce (values[2 * j]!, values[2 * j + 1]!) tau
      if shapes[pair.1]!.depth == k then
        claims := { claims with
          leafPoint := claims.leafPoint.set! pair.1 next, leafValues := claims.leafValues.set! pair.1 reduced }
      else
        claims := { claims with current := claims.current.set! pair.1 (reduced[0]!, reduced[1]!) }
    claims := { claims with point := next }
  return claims

end CompressionLean.Replay
