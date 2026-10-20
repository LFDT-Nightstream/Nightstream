import CompressionLean.Replay.Formulas

/-!
Owns the sum-check replay (`sumcheck::replay`) and the batched fraction-sum
GKR verifier (`gkr::verify`, `GkrView::read`), with the model's names
(`Layer1/Gkr.lean`, `Layer1/GkrLayer.lean`): `GkrClaims`, `combineVals`,
`reduce`, `advance`. A round message is `h(0), h(2), …, h(degree)`; `h(1)` is
the claim minus `h(0)`. Rounds and layers are replayed by recursion over the
proof's lists, one draw per model draw.
-/

namespace CompressionLean.Replay

/-- The node values `h(0), claim - h(0), h(2), …` of one round message. -/
def roundValues (message : Array Ext) (claim : Ext) : Array Ext :=
  #[message[0]!, claim - message[0]!] ++ message.extract 1 message.size

/-- Replay `rounds` against `claim` from `point`; return the point and the final claim. -/
def sumcheckRounds : List (Array Ext) → Array Ext → Ext → ReplayM (Array Ext × Ext)
  | [], point, claim => pure (point, claim)
  | message :: rest, point, claim => do
      observeExts message
      let r ← sampleExt
      sumcheckRounds rest (point.push r) (interpolate (roundValues message claim) r)

def sumcheckReplay (rounds : Array (Array Ext)) (claim : Ext) : ReplayM (Array Ext × Ext) :=
  sumcheckRounds rounds.toList #[] claim

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

/-- Layer `k` has `k - 1` rounds of the batched degree and each active tree's parts. -/
def layerShapeOk (shapes : Array TreeShape) (k : Nat) (layer : GkrLayer) : Bool :=
  let factors := layerFactors shapes k
  layer.rounds.size == k - 1 && layer.rounds.all (·.size == layerDegree factors) &&
    layer.children.size == factors.size &&
    (layer.children.zip factors).all fun pair => pair.1.size == 2 * (pair.2 + 1)

/-- The proof has the shapes' layers, rounds and children (`GkrView::read`). -/
def readGkr (proof : GkrProof) (shapes : Array TreeShape) : Except String Unit := do
  if shapes.any (fun s => s.depth == 0 || s.factors > 2) || proof.roots.size != shapes.size ||
      proof.layers.size != layerCount shapes then
    throw "GKR shape"
  ensure ((List.range (layerCount shapes)).all fun i => layerShapeOk shapes (i + 1) proof.layers[i]!)
    "GKR layer shape"

/-- `p₀·q₁ + p₁·q₀ + μ·q₀·q₁` from one tree's part pairs: factors, then the
denominator (`combine`). -/
def combineVals (children : Array Ext) (mu : Ext) : Ext :=
  let f := children.size / 2 - 1
  let q0 := children[2 * f]!
  let q1 := children[2 * f + 1]!
  prodRange f (fun j => children[2 * j]!) * q1 + prodRange f (fun j => children[2 * j + 1]!) * q0 + mu * q0 * q1

/-- One pair reduced at the top variable. -/
def reduce (pair : Ext × Ext) (tau : Ext) : Ext := pair.1 + tau * (pair.2 - pair.1)

/-- The trees that reach layer `k`, in tree order. -/
def activeTrees (shapes : Array TreeShape) (k : Nat) : Array Nat :=
  (Array.range shapes.size).filter (k ≤ shapes[·]!.depth)

/-- `Σ_i ν^i·(P_i + μ·Q_i)` and `Σ_i ν^i·combine_i` over the active trees, by
Horner's rule (the model's `wsum`). -/
def batchClaims (current : Array (Ext × Ext)) (mu nu : Ext) : List (Nat × Array Ext) → Ext × Ext
  | [] => (0, 0)
  | (t, values) :: rest =>
      let (claim, expected) := batchClaims current mu nu rest
      ((current[t]!).1 + mu * (current[t]!).2 + nu * claim, combineVals values mu + nu * expected)

/-- Reduce at `τ`: inner trees keep `(P, Q)`; trees of depth `k` end with
their leaf point and values (the model's `advance`). -/
def advance (shapes : Array TreeShape) (k : Nat) (claims : GkrClaims) (pairs : List (Nat × Array Ext))
    (next : Array Ext) (tau : Ext) : GkrClaims :=
  let point := next.push tau
  let claims := pairs.foldl (fun claims pair =>
    let reduced := (Array.range (pair.2.size / 2)).map fun j => reduce (pair.2[2 * j]!, pair.2[2 * j + 1]!) tau
    if shapes[pair.1]!.depth == k then
      { claims with leafPoint := claims.leafPoint.set! pair.1 point, leafValues := claims.leafValues.set! pair.1 reduced }
    else { claims with current := claims.current.set! pair.1 (reduced[0]!, reduced[1]!) }) claims
  { claims with point }

/-- Layers `k, k + 1, …` of the batched GKR: per layer `μ`, `ν`, the
sum-check, the check, the children and `τ`. -/
def gkrLayers (shapes : Array TreeShape) : Nat → List GkrLayer → GkrClaims → ReplayM GkrClaims
  | _, [], claims => pure claims
  | k, layer :: rest, claims => do
      let pairs := ((activeTrees shapes k).zip layer.children).toList
      let mu ← sampleExt
      let nu ← sampleExt
      let (claim, expected) := batchClaims claims.current mu nu pairs
      let (next, last) ← sumcheckReplay layer.rounds claim
      ensure (last == eqEval claims.point next * expected) "GKR layer"
      observeExts layer.children.flatten
      let tau ← sampleExt
      gkrLayers shapes (k + 1) rest (advance shapes k claims pairs next tau)

/-- Verify the batched GKR for trees of `shapes`: the roots, then every layer. -/
def gkrVerify (proof : GkrProof) (shapes : Array TreeShape) : ReplayM GkrClaims := do
  observeExts (flatPairs proof.roots)
  gkrLayers shapes 1 proof.layers.toList
    { roots := proof.roots, point := #[], current := proof.roots,
      leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] }

end CompressionLean.Replay
