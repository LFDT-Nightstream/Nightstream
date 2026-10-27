import CompressionLean.Ports.Replay.Whir

/-!
Owns the replay of the layer-1 verifier (`crates/neo-spartan/src/verifier.rs`)
in the stage order of the model (`Layer1/Stages.lean`, `Layer1/Soundness.lean`):
the proof shape, early, linear, honest fold, late, then the two openings.
Each stage draws its challenges as the model draws them, checks what Rust
checks in Rust's order, rejects with Rust's name for the first failed check,
and returns the values the model's transcript holds (`EarlyOut`,
`LinearOut`, `HonestOut`, `LateOut`). WHIR's own checks are not replayed
(`Whir.lean`), and `verify` returns the points that it gives WHIR
(`OpeningPoints`).

The check families are pure functions (`checkEarly`, `checkLate`,
`setupCheck`, `openingsAgree`); `Bridge/` relates each one to the model's
conditions (`EarlyChecks`, `LinearCheck`, `HonestConsistent`, `LateChecks`,
`Opens`).
-/

namespace CompressionLean.Replay

/-! ## The proof shape (`ProofView::read`) -/

def readProof (rel : Relation) (proof : Proof) : Except String Unit := do
  ensure (proof.histogram.size == 2 * rel.normBound + 1 && proof.earlyValues.pairs.size == rel.matrices &&
    proof.quotient.size == D - 1 && proof.linear.size == rel.cubeVars && proof.linear.all (·.size == 2) &&
    proof.leaves.size == rel.queries && proof.lateValues.size == rel.matrices &&
    proof.p0.size == 4 && proof.p1.size == 4 && proof.finals.size == 3 &&
    proof.partials.size == 2 && proof.partials.all (·.size == 2 ^ foldVars)) "proof shape"
  readGkr proof.early rel.earlyShapes
  ensure (proof.leaves.all fun leaf => leaf.values.size == (rel.kappa + runOracles) * 2 ^ foldVars &&
    leaf.path.size == rel.setupVars + 1 - foldVars) "setup leaf shape"
  readGkr proof.late rel.lateShapes
  readOpening rel.p1Whir proof.p1Opening
  readOpening rel.p0Whir proof.p0Opening

/-! ## Early (`verifier.rs` to `observe_early`, `matrix.rs::Structure::check_early`) -/

/-- `β` (norm), `(β, γ)` (row lookup), `(β, μ)` (slot scatter). -/
structure EarlyChallenges where
  beta : Ext
  lookup : Ext × Ext
  scatter : Ext × Ext

/-- One slot of `table_sum`: `total + m_t/(β - (t - H))`, rejecting a pole. -/
def tableSumStep (bound : Nat) (counts : Array Word) (beta total : Ext) (slot : Nat) : Except String Ext := do
  let denominator := beta - Ext.ofGl (Word.ofNat slot - Word.ofNat bound)
  ensure (!denominator.isZero) "histogram pole"
  pure (total + denominator.inv.scale counts[slot]!)

/-- `Σ_t m_t/(β - (t - H))`; rejects a pole (`norm::table_sum`, the model's `tableSum`). -/
def tableSum (bound : Nat) (counts : Array Word) (beta : Ext) : Except String Ext :=
  (List.range counts.size).foldlM (tableSumStep bound counts beta) 0

/-- `P_x·Q_y = P_y·Q_x` with both denominators nonzero (`check_pair`, the model's `PairCheck`). -/
def pairCheck (x y : Ext × Ext) (name : String) : Except String Unit := do
  ensure (!x.2.isZero) name
  ensure (!y.2.isZero) name
  ensure (x.1 * y.2 == y.1 * x.2) name

/-- `r + γ·(e₁ + γ·e₂)` (`tuple`, the model's `tuple`). -/
def tuple (gamma r e₁ e₂ : Ext) : Ext := r + gamma * (e₁ + gamma * e₂)

/-- `j~` over runs: matrix `j` on its run range, zero on padding runs (`matrix_step`). -/
def matrixStep (layout : Layout) (point : Array Ext) : Ext :=
  step point ((layout.runBounds.extract 0 layout.matrices).toList.zip
    ((List.range layout.matrices).map fun j => Ext.ofGl (Word.ofNat j))) layout.runBounds[layout.matrices]!

/-- What the early leaf checks leave to openings and to the honest fold. -/
structure EarlyOpenings where
  runPoint : Array Ext
  pairPoint : Array Ext
  rowPoint : Array Ext
  mult : Ext
  rowClaim : Ext
  slotClaim : Ext
  coefClaim : Ext

/-- The early roots and leaf claims against the early values (`check_early`);
the trees are numbered as in `Relation.earlyShapes`. -/
def checkEarly (layout : Layout) (r : Array (Word × Word)) (ch : EarlyChallenges) (claims : GkrClaims)
    (v : EarlyValues) : Except String EarlyOpenings := do
  pairCheck claims.roots[1]! claims.roots[2]! "row lookup"
  pairCheck claims.roots[3]! claims.roots[4]! "slot scatter"
  ensure (claims.leafValue 1 0 == ch.lookup.1 - tuple ch.lookup.2 v.row v.eRe v.eIm) "row lookup run leaf"
  let rowPoint := claims.leafPoint[2]!
  let chiValue := chi r rowPoint
  ensure (claims.leafValue 2 1 == ch.lookup.1 - tuple ch.lookup.2 (identityValue rowPoint) chiValue.re chiValue.im)
    "row lookup row leaf"
  ensure (claims.leafValue 3 1 == v.eRe + ch.scatter.2 * v.eIm) "slot scatter run leaf"
  let runPoint := claims.leafPoint[3]!
  let tag := (matrixStep layout runPoint).scale (Word.ofNat (2 ^ layout.slotVariables))
  let slotClaim := ch.scatter.1 - claims.leafValue 3 2 - tag
  let pairPoint := claims.leafPoint[4]!
  let eq := eqTable (pairPoint.extract layout.slotVariables pairPoint.size)
  let combined := sumRange (min v.pairs.size eq.size) fun j =>
    eq[j]! * ((v.pairs[j]!).1 + ch.scatter.2 * (v.pairs[j]!).2)
  ensure (claims.leafValue 4 0 == combined) "slot scatter pair leaf"
  ensure (claims.leafValue 4 1 == ch.scatter.1 - identityValue pairPoint) "slot scatter pair leaf"
  return { runPoint, pairPoint := pairPoint.extract 0 layout.slotVariables, rowPoint,
           mult := claims.leafValue 2 0, rowClaim := v.row, slotClaim,
           coefClaim := claims.leafValue 3 0 }

/-- The early stage's transcript and what later stages read. -/
structure EarlyOut where
  challenges : EarlyChallenges
  gkr : GkrClaims
  values : EarlyValues
  /-- `z~` at the norm leaf: `β - q~`. -/
  zLeaf : Ext
  openings : EarlyOpenings

def earlyStage (rel : Relation) (st : Statement) (proof : Proof) : ReplayM EarlyOut := do
  observeAll rel.commitmentSeed
  observeAll proof.p0
  observeAll proof.histogram
  let beta ← sampleExt
  let lookup ← samplePair
  let scatter ← samplePair
  let challenges : EarlyChallenges := { beta, lookup, scatter }
  let total ← liftExcept (tableSum rel.normBound proof.histogram beta)
  let claims ← gkrVerify proof.early rel.earlyShapes
  ensure (!(claims.roots[0]!).2.isZero) "GKR root against the histogram"
  ensure ((claims.roots[0]!).1 == (claims.roots[0]!).2 * total) "GKR root against the histogram"
  let openings ← liftExcept (checkEarly rel.key.layout st.point challenges claims proof.earlyValues)
  let v := proof.earlyValues
  observeExts (#[v.eRe, v.eIm, v.row] ++ flatPairs v.pairs ++ #[v.block])
  return { challenges, gkr := claims, values := v, zLeaf := beta - claims.leafValue 0 0, openings }

/-! ## Linear (`verifier.rs` from `λ` to the linear claim) -/

/-- `Λ~(x) = Σ_{l<D} ζ^l·eq(x, l)` (`lane_value`, the model's `laneTable` extension). -/
def laneValue (zeta : Ext) (point : Array Ext) : Ext :=
  sumRange D fun l => (eqTable point)[l]! * zeta.pow l

/-- The linear stage's transcript (the model's `LinearOut`) and what later
stages read. -/
structure LinearOut where
  lam : Ext
  zeta : Ext
  blockPoint : Array Ext
  middle : Ext
  lanePart : Array Ext
  last : Ext
  keyValue : Ext
  omegaValue : Ext
  zValue : Ext
  lanePoint : Array Ext
  mixing : Mixing
  tau : Array Ext

/-- `Ω~(s_b) = κ_ζ·A_λ~(x_ζ, s_b) + Eval_K + Ω^A~(s_b) + public` (the model's `omegaComputed`). -/
def omegaComputed (rel : Relation) (st : Statement) (m : Mixing) (tau blockPoint : Array Ext)
    (scale keyValue omegaValue : Ext) : Ext :=
  let evalKValue := evalK st.point blockPoint tau rel.key.blocks
  let pub := sumRange rel.publicBlocks fun b => eqAt blockPoint b * m.public b
  scale * keyValue + m.evalK.1 * evalKValue.re + m.evalK.2 * evalKValue.im + omegaValue + pub

def linearStage (rel : Relation) (st : Statement) (proof : Proof) : ReplayM LinearOut := do
  let lam ← sampleExt
  let mixing : Mixing := { lam, kappa := rel.kappa, matrices := rel.matrices }
  observeExts proof.quotient
  let zeta ← sampleExt
  let value := liftedTarget (targets st mixing) proof.quotient zeta
  -- The block bits bind first, then the lane bits: one replay, split at the block bits.
  let (blockPoint, middle) ← sumcheckReplay (proof.linear.extract 0 rel.blockVars) value
  let (lanePart, last) ← sumcheckReplay (proof.linear.extract rel.blockVars proof.linear.size) middle
  observeExts proof.finals
  let (scale, lanePoint) ← liftExcept (scaledEqPoint zeta laneVars)
  let tau := tau zeta
  let (keyValue, omegaValue, zValue) := (proof.finals[0]!, proof.finals[1]!, proof.finals[2]!)
  let omega := omegaComputed rel st mixing tau blockPoint scale keyValue omegaValue
  ensure (last == omega * laneValue zeta lanePart * zValue) "linear claim"
  return { lam, zeta, blockPoint, middle, lanePart, last, keyValue, omegaValue, zValue, lanePoint, mixing, tau }

/-! ## Honest fold (`verifier.rs` from `μ` to the setup queries, `setup.rs`) -/

/-- A point padded with zeros to the setup variables (`setup_point`, the model's `padded`). -/
def padded (n : Nat) (point : Array Ext) : Array Ext := point ++ Array.replicate (n - point.size) 0

/-- `Σ_u eq(point, u)·partials[u]` (`combine`). -/
def combine (partials point : Array Ext) : Ext := dot (eqTable point) partials

/-- The leaf's path reaches the key root, lane by lane (`merkle_root`). -/
def pathOk (root : Array Word) (index : Nat) (leaf : Leaf) : Bool :=
  let computed := merkleRoot (hashLeaf leaf.values) index leaf.path
  (List.range 4).all fun lane => computed[lane]! == root[lane]!

/-- `α^[u] = Π_i α_i^{bit i of u}`, low bit first (the model's `monomial`). -/
def monomial : List Ext → Nat → Ext
  | [], _ => 1
  | a :: rest, u => (if u % 2 = 1 then a else 1) * monomial rest (u / 2)

/-- The weight of coset value `v`: `Σ_u α^[u]·2^-f·y^-u·ω^-uv` (the model's `leaf_formula`). -/
def leafWeight (alpha : Array Ext) (yInverse omega : Word) (v : Nat) : Ext :=
  sumRange (2 ^ foldVars) fun u =>
    ((monomial alpha.toList u).scale ((Word.ofNat (2 ^ foldVars)).inv * yInverse.pow u)).scale
      (omega.pow ((2 ^ foldVars - u) * v % 2 ^ foldVars))

/-- One queried setup leaf: its path to the key root, then each group's folded
value `Σ_v w_v·Σ_k weight_k·Ô_k(y·ω^v)` (`setup::check`, the model's `leaf_formula`). -/
def setupCheck (root : Array Word) (setupVariables index : Nat) (leaf : Leaf) (groups : Array (Array Ext))
    (alpha : Array Ext) : Except String (Array Ext) := do
  ensure (pathOk root index leaf) "setup leaf path"
  let yInverse := powerFromBits (twoAdicGenerator (setupVariables + 1)).inv index (setupVariables + 1 - foldVars) *
    generator.inv
  let omega := twoAdicGenerator foldVars
  let offsets := groups.foldl (fun offsets weights => offsets.push (offsets.back! + weights.size)) #[0]
  return (groups.zip offsets).map fun pair =>
    sumRange (2 ^ foldVars) fun v => leafWeight alpha yInverse omega v *
      sumRange pair.1.size fun k => (pair.1[k]!).scale leaf.values[(pair.2 + k) * 2 ^ foldVars + v]!

/-- The honest-fold stage's transcript (the model's `HonestOut`) and the
queried folded values. -/
structure HonestOut where
  mu : Ext
  alpha : Array Ext
  words : Array Word
  queries : Array Nat
  queried : Array Ext

def honestStage (rel : Relation) (proof : Proof) (early : EarlyOut) (lin : LinearOut) : ReplayM HonestOut := do
  let mu ← sampleExt
  observeExts proof.partials.flatten
  let a0 ← sampleExt
  let a1 ← sampleExt
  let a2 ← sampleExt
  let alpha := #[a0, a1, a2]
  let setupVariables := rel.setupVars
  let keyPoint := padded setupVariables (lin.lanePoint ++ lin.blockPoint)
  let runPoint := padded setupVariables early.openings.runPoint
  let embedded := (early.openings.runPoint.extract rel.key.layout.slotVariables early.openings.runPoint.size).foldl
    (fun total x => total * oneMinus x) early.values.block
  let o := early.openings
  let runValue := #[o.coefClaim, o.slotClaim, o.rowClaim].foldl (fun total value => value + mu * total) embedded
  ensure (combine proof.partials[0]! (keyPoint.extract 0 foldVars) == lin.keyValue) "honest-fold partials"
  ensure (combine proof.partials[1]! (runPoint.extract 0 foldVars) == runValue) "honest-fold partials"
  observeAll rel.commitmentSeed
  observeAll proof.p1
  let words ← sampleWords rel.queries
  let queries := words.map fun word => word.toNat % 2 ^ (setupVariables + 1 - foldVars)
  let groups := #[(Array.range rel.kappa).map lin.mixing.commitment, (Array.range runOracles).map mu.pow]
  let queried ← liftExcept ((queries.zip proof.leaves).toList.foldlM (fun queried pair => do
    pure (queried ++ (← setupCheck rel.key.root setupVariables pair.1 pair.2 groups alpha))) #[])
  return { mu, alpha, words, queries, queried }

/-! ## Late (`verifier.rs` after the setup queries, `matrix.rs::Structure::check_late`) -/

/-- `Ĥ_t~` over slots: the class's lane table at the slot's lane (`slot_step`). -/
def slotStep (layout : Layout) (point : Array Ext) (lanes : Array (Array Ext × Array Ext)) (t : Nat) : Ext :=
  step point (layout.segments.toList.map fun s =>
    (s.first, let pair := lanes[s.classIndex]!; (if t = 0 then pair.1 else pair.2)[s.lane]!)) layout.slots

/-- What the late leaf checks leave to openings. -/
structure LateOpenings where
  pairPoint : Array Ext
  block : Ext
  blockPoint : Array Ext
  omega : Ext

/-- The late roots and leaf claims against the late values (`check_late`). -/
def checkLate (layout : Layout) (beta : Ext) (m : Mixing) (lanes : Array (Array Ext × Array Ext))
    (claims : GkrClaims) (pairs : Array (Ext × Ext)) : Except String LateOpenings := do
  pairCheck claims.roots[0]! claims.roots[1]! "block scatter"
  let point := claims.leafPoint[0]!
  let slotPart := point.extract 0 layout.slotVariables
  let t := point[layout.slotVariables]!
  let ubar := sumRange pairs.size fun j => (m.evalA j).1 * (pairs[j]!).1 + (m.evalA j).2 * (pairs[j]!).2
  let lane := oneMinus t * slotStep layout slotPart lanes 0 + t * slotStep layout slotPart lanes 1
  ensure (claims.leafValue 0 0 == ubar) "block scatter pair leaf"
  ensure (claims.leafValue 0 1 == lane) "block scatter pair leaf"
  ensure (claims.leafValue 1 1 == beta - identityValue claims.leafPoint[1]!) "block scatter block leaf"
  return { pairPoint := slotPart, block := beta - claims.leafValue 0 2 - t,
           blockPoint := claims.leafPoint[1]!, omega := claims.leafValue 1 0 }

/-- The late stage's transcript (the model's `LateOut`) and its openings. -/
structure LateOut where
  beta : Ext
  gkr : GkrClaims
  pairs : Array (Ext × Ext)
  openings : LateOpenings

def lateStage (rel : Relation) (proof : Proof) (lin : LinearOut) : ReplayM LateOut := do
  let beta ← sampleExt
  let claims ← gkrVerify proof.late rel.lateShapes
  let lanes := rel.key.layout.classes.map fun c => laneSplit lin.tau c.len c.ratio
  let openings ← liftExcept (checkLate rel.key.layout beta lin.mixing lanes claims proof.lateValues)
  observeExts (flatPairs proof.lateValues)
  return { beta, gkr := claims, pairs := proof.lateValues, openings }

/-! ## Openings (the model's `Opens`, idealized) -/

/-- The `Ext` value whose coordinate columns opened to `values` (`from_coordinates`). -/
def fromCoordinates (values : Array Ext) : Ext :=
  sumRange values.size fun c => Ext.basis c * values[c]!

/-- The P1 values: every batch read as triples of coordinate openings. -/
def p1Values (opened : Array (Array Ext)) : Array Ext :=
  opened.flatMap fun batch =>
    (Array.range (batch.size / 3)).map fun i => fromCoordinates (batch.extract (3 * i) (3 * i + 3))

/-- Every opened value equals the value the verifier expects. -/
def openingsAgree (opened expected : Array (Array Ext)) (what : String) : Except String Unit := do
  ensure (opened.map (·.size) == expected.map (·.size)) what
  ensure ((opened.flatten.zip expected.flatten).all fun pair => pair.1 == pair.2) what

/-- The P1 openings: the two fold claims, the setup queries, `Ω^A` at the
linear and the late block points. -/
def p1Expected (proof : Proof) (honest : HonestOut) (lin : LinearOut) (late : LateOut) : Array Ext :=
  #[combine proof.partials[0]! honest.alpha, combine proof.partials[1]! honest.alpha] ++ honest.queried ++
    #[lin.omegaValue, late.openings.omega]

/-- The P0 openings, one batch per point: `z` at the norm leaf and the linear
point, `e`, the early pairs, the slot-block copy, the late pairs and copy,
the row counts. -/
def p0Expected (early : EarlyOut) (lin : LinearOut) (late : LateOut) : Array (Array Ext) :=
  #[#[early.zLeaf], #[lin.zValue], #[early.values.eRe, early.values.eIm], flatPairs early.values.pairs,
    #[early.values.block], flatPairs late.pairs ++ #[late.openings.block], #[early.openings.mult]]

/-- The query point of leaf `index`: `(q, q², q⁴, …)` with `q = ω^index·g^8`,
in `setupVariables - 3` coordinates (`setup::query_point`). -/
def leafQueryPoint (setupVariables index : Nat) : Array Ext :=
  let bits := setupVariables + 1 - foldVars
  let q := powerFromBits (twoAdicGenerator bits) index bits * generator.pow (2 ^ foldVars)
  ((List.range (setupVariables - foldVars)).map fun k => Ext.ofGl (q.pow (2 ^ k))).toArray

/-- The points that the verifier gives WHIR, one per batch (`verifier.rs`,
`p1_points`). P0: the norm leaf, the linear point, the run point, the early
pair point, the run point's slot part, the late pair point and the row
point. P1: the tails of the key and run points, one point per setup query,
and the linear and the late block points. -/
structure OpeningPoints where
  norm : Array Ext
  linear : Array Ext
  run : Array Ext
  pairs : Array Ext
  slots : Array Ext
  latePairs : Array Ext
  rows : Array Ext
  keyTail : Array Ext
  runTail : Array Ext
  queries : Array (Array Ext)
  block : Array Ext
  lateBlock : Array Ext

def openingPoints (rel : Relation) (early : EarlyOut) (lin : LinearOut) (honest : HonestOut) (late : LateOut) :
    OpeningPoints :=
  let keyPoint := padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)
  let runPoint := padded rel.setupVars early.openings.runPoint
  { norm := early.gkr.leafPoint[0]!, linear := lin.lanePart ++ lin.blockPoint, run := early.openings.runPoint,
    pairs := early.openings.pairPoint, slots := early.openings.runPoint.extract 0 rel.key.layout.slotVariables,
    latePairs := late.openings.pairPoint, rows := early.openings.rowPoint,
    keyTail := keyPoint.extract foldVars keyPoint.size, runTail := runPoint.extract foldVars runPoint.size,
    queries := honest.queries.map (leafQueryPoint rel.setupVars), block := lin.blockPoint,
    lateBlock := late.openings.blockPoint }

/-! ## The verifier -/

/-- Replay Rust's layer-1 verifier on `proof` for `st` after `seed`. It
returns the points that it gives WHIR, which the WHIR assumption names
(`Bridge/Openings.lean`). -/
def verify (rel : Relation) (seed : Array Word) (st : Statement) (proof : Proof) : ReplayM OpeningPoints := do
  ensure (st.fits rel) "claim words"
  liftExcept (readProof rel proof)
  observeAll seed
  observeAll rel.words
  observeAll st.words
  let early ← earlyStage rel st proof
  let lin ← linearStage rel st proof
  let honest ← honestStage rel proof early lin
  let late ← lateStage rel proof lin
  whirTranscript rel.p1Whir proof.p1Opening
  liftExcept (openingsAgree #[p1Values proof.p1Opening.evals] #[p1Expected proof honest lin late] "P1 openings")
  whirTranscript rel.p0Whir proof.p0Opening
  liftExcept (openingsAgree proof.p0Opening.evals (p0Expected early lin late) "P0 openings")
  return openingPoints rel early lin honest late

end CompressionLean.Replay
