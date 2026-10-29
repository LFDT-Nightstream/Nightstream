import CompressionLean.Ports.Replay.Formulas

namespace CompressionLean.Replay

def roundValues (message : Array Ext) (claim : Ext) : Array Ext :=
  #[message[0]!, claim - message[0]!] ++ message.extract 1 message.size

def sumcheckRounds : List (Array Ext) → Array Ext → Ext → ReplayM (Array Ext × Ext)
  | [], point, claim => pure (point, claim)
  | message :: rest, point, claim => do
      observeExts message
      let r ← sampleExt
      sumcheckRounds rest (point.push r) (interpolate (roundValues message claim) r)

def sumcheckReplay (rounds : Array (Array Ext)) (claim : Ext) : ReplayM (Array Ext × Ext) :=
  sumcheckRounds rounds.toList #[] claim

structure GkrClaims where
  roots : Array (Ext × Ext)
  point : Array Ext
  current : Array (Ext × Ext)
  leafPoint : Array (Array Ext)
  leafValues : Array (Array Ext)

def GkrClaims.leafValue (claims : GkrClaims) (t j : Nat) : Ext := (claims.leafValues[t]!)[j]!

def activeTrees (shapes : Array TreeShape) (k : Nat) : Array Nat :=
  (Array.range shapes.size).filter (k ≤ shapes[·]!.depth)

def partFactors (shapes : Array TreeShape) (k t : Nat) : Nat :=
  if shapes[t]!.depth = k then shapes[t]!.factors else 1

def layerFactors (shapes : Array TreeShape) (k : Nat) : Array Nat :=
  (activeTrees shapes k).map (partFactors shapes k)

def layerDegree (factors : Array Nat) : Nat := factors.foldr (fun f d => max d (f + 2)) 3

def layerCount (shapes : Array TreeShape) : Nat := (List.range shapes.size).foldr (fun t d => max d shapes[t]!.depth) 0

def layerShapeOk (shapes : Array TreeShape) (k : Nat) (layer : GkrLayer) : Bool :=
  layer.rounds.size == k - 1 && layer.rounds.all (·.size == layerDegree (layerFactors shapes k)) &&
    layer.children.size == (activeTrees shapes k).size &&
    ((activeTrees shapes k).zip layer.children).all fun pair => pair.2.size == 2 * (partFactors shapes k pair.1 + 1)

def readGkr (proof : GkrProof) (shapes : Array TreeShape) : Except String Unit := do
  if shapes.any (fun s => s.depth == 0 || s.factors > 2) || proof.roots.size != shapes.size ||
      proof.layers.size != layerCount shapes then
    throw "GKR shape"
  ensure ((List.range (layerCount shapes)).all fun i => layerShapeOk shapes (i + 1) proof.layers[i]!)
    "GKR layer shape"

def combineVals (children : Array Ext) (mu : Ext) : Ext :=
  let f := children.size / 2 - 1
  let q0 := children[2 * f]!
  let q1 := children[2 * f + 1]!
  prodRange f (fun j => children[2 * j]!) * q1 + prodRange f (fun j => children[2 * j + 1]!) * q0 + mu * q0 * q1

def reduce (pair : Ext × Ext) (tau : Ext) : Ext := pair.1 + tau * (pair.2 - pair.1)

def batchClaim (current : Array (Ext × Ext)) (mu nu : Ext) : List (Nat × Array Ext) → Ext
  | [] => 0
  | (t, _) :: rest => (current[t]!).1 + mu * (current[t]!).2 + nu * batchClaim current mu nu rest

def batchExpected (mu nu : Ext) : List (Nat × Array Ext) → Ext
  | [] => 0
  | (_, values) :: rest => combineVals values mu + nu * batchExpected mu nu rest

def reducedOf (pairs : List (Nat × Array Ext)) (tau : Ext) (t : Nat) : Array Ext :=
  match pairs.find? (·.1 == t) with
  | some pair => (Array.range (pair.2.size / 2)).map fun j => reduce (pair.2[2 * j]!, pair.2[2 * j + 1]!) tau
  | none => #[]

def advance (shapes : Array TreeShape) (k : Nat) (claims : GkrClaims) (pairs : List (Nat × Array Ext))
    (next : Array Ext) (tau : Ext) : GkrClaims where
  roots := claims.roots
  point := next.push tau
  current := (Array.range claims.current.size).map fun t =>
    if k < shapes[t]!.depth then ((reducedOf pairs tau t)[0]!, (reducedOf pairs tau t)[1]!) else claims.current[t]!
  leafPoint := (Array.range claims.leafPoint.size).map fun t =>
    if shapes[t]!.depth = k then next.push tau else claims.leafPoint[t]!
  leafValues := (Array.range claims.leafValues.size).map fun t =>
    if shapes[t]!.depth = k then reducedOf pairs tau t else claims.leafValues[t]!

def gkrLayers (shapes : Array TreeShape) : Nat → List GkrLayer → GkrClaims → ReplayM GkrClaims
  | _, [], claims => pure claims
  | k, layer :: rest, claims => do
      let pairs := ((activeTrees shapes k).zip layer.children).toList
      let mu ← sampleExt
      let nu ← sampleExt
      let result ← sumcheckReplay layer.rounds (batchClaim claims.current mu nu pairs)
      ensure (result.2 == eqEval claims.point result.1 * batchExpected mu nu pairs) "GKR layer"
      observeExts layer.children.flatten
      let tau ← sampleExt
      gkrLayers shapes (k + 1) rest (advance shapes k claims pairs result.1 tau)

def gkrVerify (proof : GkrProof) (shapes : Array TreeShape) : ReplayM GkrClaims := do
  observeExts (flatPairs proof.roots)
  gkrLayers shapes 1 proof.layers.toList
    { roots := proof.roots, point := #[], current := proof.roots,
      leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] }

end CompressionLean.Replay
