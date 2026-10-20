import CompressionLean.Replay.Whir

/-!
Owns the replay of the layer-1 verifier (`crates/neo-spartan/src/verifier.rs`)
in the stage order of the model (`Layer1/Stages.lean`, `Layer1/Soundness.lean`):
the proof shape, early, linear, honest fold, late, then the two openings.
Each stage draws its challenges, checks what Rust checks in Rust's order,
rejects with Rust's name for the first failed check, and returns the values
the model's transcript holds (`EarlyOut`, `LinearOut`, `HonestOut`, `LateOut`).
WHIR's own checks are not replayed (`Whir.lean`).

The check families are pure functions (`checkEarly`, `checkLate`,
`setupCheck`), so a later bridge can relate each one to the model's
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
  for leaf in proof.leaves do
    ensure (leaf.values.size == (rel.kappa + runOracles) * 2 ^ foldVars &&
      leaf.path.size == rel.setupVars + 1 - foldVars) "setup leaf shape"
  readGkr proof.late rel.lateShapes
  readOpening rel.p1Whir proof.p1Opening
  readOpening rel.p0Whir proof.p0Opening

/-! ## Early (`verifier.rs` to `observe_early`, `matrix.rs::check_early`) -/

/-- `β` (norm), `(β, γ)` (row lookup), `(β, μ)` (slot scatter). -/
structure EarlyChallenges where
  beta : Ext
  lookup : Ext × Ext
  scatter : Ext × Ext

/-- `Σ_t m_t/(β - (t - H))`; rejects a pole (`norm::table_sum`, the model's `tableSum`). -/
def tableSum (bound : Nat) (counts : Array Gl) (beta : Ext) : Except String Ext := do
  let mut total : Ext := 0
  for slot in [0:counts.size] do
    let denominator := beta - Ext.ofGl (Gl.ofNat slot - Gl.ofNat bound)
    ensure (!denominator.isZero) "histogram pole"
    total := total + denominator.inv.scale counts[slot]!
  return total

/-- `P_x·Q_y = P_y·Q_x` with both denominators nonzero (`check_pair`, the model's `PairCheck`). -/
def pairCheck (x y : Ext × Ext) (name : String) : Except String Unit := do
  ensure (!x.2.isZero) name
  ensure (!y.2.isZero) name
  ensure (x.1 * y.2 == y.1 * x.2) name

/-- `r + γ·(e₁ + γ·e₂)` (`tuple`, the model's `tuple`). -/
def tuple (gamma r e₁ e₂ : Ext) : Ext := r + gamma * (e₁ + gamma * e₂)

/-- `j~` over runs: matrix `j` on its run range, zero on padding runs (`matrix_step`). -/
def matrixStep (layout : Layout) (point : Array Ext) : Ext :=
  step point (layout.runBounds.extract 0 layout.matrices)
    ((Array.range layout.matrices).map fun j => Ext.ofGl (Gl.ofNat j)) layout.runBounds[layout.matrices]!

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
def checkEarly (layout : Layout) (r : Array (Gl × Gl)) (ch : EarlyChallenges) (claims : GkrClaims)
    (v : EarlyValues) : Except String EarlyOpenings := do
  pairCheck claims.roots[1]! claims.roots[2]! "row lookup"
  pairCheck claims.roots[3]! claims.roots[4]! "slot scatter"
  let (beta, gamma) := ch.lookup
  ensure (claims.leafValue 1 0 == beta - tuple gamma v.row v.eRe v.eIm) "row lookup run leaf"
  let rowPoint := claims.leafPoint[2]!
  let chiValue := chi r rowPoint
  ensure (claims.leafValue 2 1 == beta - tuple gamma (identityValue rowPoint) chiValue.re chiValue.im)
    "row lookup row leaf"
  let (beta, mu) := ch.scatter
  ensure (claims.leafValue 3 1 == v.eRe + mu * v.eIm) "slot scatter run leaf"
  let runPoint := claims.leafPoint[3]!
  let tag := (matrixStep layout runPoint).scale (Gl.ofNat (2 ^ layout.slotVariables))
  let slotClaim := beta - claims.leafValue 3 2 - tag
  let pairPoint := claims.leafPoint[4]!
  let eq := eqTable (pairPoint.extract layout.slotVariables pairPoint.size)
  let combined := (v.pairs.zip eq).foldl (fun total pair => total + pair.2 * (pair.1.1 + mu * pair.1.2)) 0
  ensure (claims.leafValue 4 0 == combined) "slot scatter pair leaf"
  ensure (claims.leafValue 4 1 == beta - identityValue pairPoint) "slot scatter pair leaf"
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
  let lookup := (← sampleExt, ← sampleExt)
  let scatter := (← sampleExt, ← sampleExt)
  let challenges : EarlyChallenges := { beta, lookup, scatter }
  let total ← liftExcept (tableSum rel.normBound proof.histogram beta)
  let claims ← gkrVerify proof.early rel.earlyShapes
  let (p, q) := claims.roots[0]!
  ensure (!q.isZero) "GKR root against the histogram"
  ensure (p == q * total) "GKR root against the histogram"
  let openings ← liftExcept (checkEarly rel.key.layout st.point challenges claims proof.earlyValues)
  let v := proof.earlyValues
  observeExt v.eRe
  observeExt v.eIm
  observeExt v.row
  for pair in v.pairs do
    observeExt pair.1
    observeExt pair.2
  observeExt v.block
  return { challenges, gkr := claims, values := v, zLeaf := beta - claims.leafValue 0 0, openings }

/-! ## Linear (`verifier.rs` from `λ` to the linear claim) -/

/-- `Λ~(x) = Σ_{l<D} ζ^l·eq(x, l)` (`lane_value`, the model's `laneTable` extension). -/
def laneValue (zeta : Ext) (point : Array Ext) : Ext :=
  let eq := eqTable point
  ((List.range D).foldl (fun (acc : Ext × Ext) l => (acc.1 + eq[l]! * acc.2, acc.2 * zeta)) (0, 1)).1

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
  let pub := (List.range rel.publicBlocks).foldl (fun total b => total + eqAt blockPoint b * m.public b) 0
  scale * keyValue + m.evalK.1 * evalKValue.re + m.evalK.2 * evalKValue.im + omegaValue + pub

def linearStage (rel : Relation) (st : Statement) (proof : Proof) : ReplayM LinearOut := do
  let lam ← sampleExt
  let mixing := Mixing.new lam rel.rowCount rel.kappa rel.matrices
  proof.quotient.forM observeExt
  let zeta ← sampleExt
  let value := liftedTarget (targets st mixing) proof.quotient zeta
  -- The block bits bind first, then the lane bits: one replay, split at the block bits.
  let (blockPoint, middle) ← sumcheckReplay (proof.linear.extract 0 rel.blockVars) value
  let (lanePart, last) ← sumcheckReplay (proof.linear.extract rel.blockVars proof.linear.size) middle
  proof.finals.forM observeExt
  let (scale, lanePoint) ← liftExcept (scaledEqPoint zeta laneVars)
  let tau := tau zeta
  let (keyValue, omegaValue, zValue) := (proof.finals[0]!, proof.finals[1]!, proof.finals[2]!)
  let omega := omegaComputed rel st mixing tau blockPoint scale keyValue omegaValue
  ensure (last == omega * laneValue zeta lanePart * zValue) "linear claim"
  return { lam, zeta, blockPoint, middle, lanePart, last, keyValue, omegaValue, zValue, lanePoint, mixing, tau }

/-! ## Honest fold (`verifier.rs` from `μ` to the setup queries, `setup.rs`) -/

/-- A point padded with zeros to the setup setupVariables (`setup_point`, the model's `padded`). -/
def padded (n : Nat) (point : Array Ext) : Array Ext := point ++ Array.replicate (n - point.size) 0

/-- `Σ_u eq(point, u)·partials[u]` (`combine`). -/
def combine (partials point : Array Ext) : Ext := dot (eqTable point) partials

/-- One queried setup leaf: its path to the key root, then each group's folded
value `Σ_u α^[u]·Ô_u(q)` (`setup::check`, the model's `leaf_formula`). -/
def setupCheck (root : Array Gl) (setupVariables index : Nat) (leaf : Leaf) (groups : Array (Array Ext))
    (alpha : Array Ext) : Except String (Array Ext) := do
  let computed := merkleRoot (hashLeaf leaf.values) index leaf.path
  for lane in [0:4] do
    ensure (computed[lane]! == root[lane]!) "setup leaf path"
  let rotation := powerFromBits (twoAdicGenerator (setupVariables + 1)).inv index (setupVariables + 1 - foldVars)
  let yInverse := rotation * generator.inv
  let monomials := alpha.foldl (fun ms a => ms ++ ms.map (· * a)) #[(1 : Ext)]
  let coefficients := (monomials.foldl (fun (acc : Array Ext × Gl) m => (acc.1.push (m.scale acc.2), acc.2 * yInverse))
    (#[], (Gl.ofNat (2 ^ foldVars)).inv)).1
  let omega := twoAdicGenerator foldVars
  let leafSize := 2 ^ foldVars
  let weight := (Array.range leafSize).map fun v =>
    (List.range leafSize).foldl (fun total u => total + coefficients[u]!.scale (omega.pow ((leafSize - u) * v % leafSize))) 0
  let folded := groups.foldl (fun (acc : Array Ext × Nat) weights =>
    let combined := (Array.range leafSize).map fun v =>
      (List.range weights.size).foldl (fun total k => total + weights[k]!.scale leaf.values[(acc.2 + k) * leafSize + v]!) 0
    (acc.1.push (dot weight combined), acc.2 + weights.size)) (#[], 0)
  return folded.1

/-- The honest-fold stage's transcript (the model's `HonestOut`) and the
queried folded values. -/
structure HonestOut where
  mu : Ext
  alpha : Array Ext
  queries : Array Nat
  queried : Array Ext

def honestStage (rel : Relation) (proof : Proof) (early : EarlyOut) (lin : LinearOut) : ReplayM HonestOut := do
  let mu ← sampleExt
  for values in proof.partials do
    values.forM observeExt
  let alpha := #[← sampleExt, ← sampleExt, ← sampleExt]
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
  let mut queries := #[]
  for _ in [0:rel.queries] do
    queries := queries.push (← sampleBits (setupVariables + 1 - foldVars))
  let commitmentRows := (Array.range rel.kappa).map lin.mixing.commitment
  let runWeights := (List.range (runOracles - 1)).foldl (fun ws _ => ws.push (ws.back! * mu)) #[(1 : Ext)]
  let mut queried := #[]
  for pair in queries.zip proof.leaves do
    queried := queried ++ (← liftExcept (setupCheck rel.key.root setupVariables pair.1 pair.2
      #[commitmentRows, runWeights] alpha))
  return { mu, alpha, queries, queried }

/-! ## Late (`verifier.rs` after the setup queries, `matrix.rs::check_late`) -/

/-- `Ĥ_t~` over slots: the class's lane table at the slot's lane (`slot_step`). -/
def slotStep (layout : Layout) (point : Array Ext) (lanes : Array (Array Ext × Array Ext)) (t : Nat) : Ext :=
  step point (layout.segments.map (·.first))
    (layout.segments.map fun s => let pair := lanes[s.classIndex]!; (if t = 0 then pair.1 else pair.2)[s.lane]!)
    layout.slots

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
  let ubar := (List.range pairs.size).foldl
    (fun total j => total + ((m.evalA j).1 * pairs[j]!.1 + (m.evalA j).2 * pairs[j]!.2)) 0
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
  for pair in proof.lateValues do
    observeExt pair.1
    observeExt pair.2
  return { beta, gkr := claims, pairs := proof.lateValues, openings }

/-! ## Openings (the model's `Opens`, idealized) -/

/-- The `Ext` value whose coordinate columns opened to `values` (`from_coordinates`). -/
def fromCoordinates (values : Array Ext) : Ext :=
  (List.range values.size).foldl (fun total c => total + Ext.basis c * values[c]!) 0

def flatPairs (pairs : Array (Ext × Ext)) : Array Ext := pairs.flatMap fun pair => #[pair.1, pair.2]

/-- Every opened value equals the value the verifier expects. -/
def openingsAgree (opened expected : Array (Array Ext)) (what : String) : Except String Unit := do
  ensure (opened.map (·.size) == expected.map (·.size)) what
  for pair in opened.flatten.zip expected.flatten do
    ensure (pair.1 == pair.2) what

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

/-! ## The verifier -/

/-- Replay Rust's layer-1 verifier on `proof` for `st` after `seed`. -/
def verify (rel : Relation) (seed : Array Gl) (st : Statement) (proof : Proof) : ReplayM Unit := do
  ensure (st.fits rel) "claim words"
  liftExcept (readProof rel proof)
  observeAll seed
  observeAll rel.words
  observeAll st.words
  let early ← earlyStage rel st proof
  let lin ← linearStage rel st proof
  let honest ← honestStage rel proof early lin
  let late ← lateStage rel proof lin
  let p1 ← whirTranscript rel.p1Whir proof.p1Opening
  let values := p1.flatMap fun batch => (Array.range (batch.size / 3)).map fun i => fromCoordinates (batch.extract (3 * i) (3 * i + 3))
  liftExcept (openingsAgree #[values] #[p1Expected proof honest lin late] "P1 openings")
  let p0 ← whirTranscript rel.p0Whir proof.p0Opening
  liftExcept (openingsAgree p0 (p0Expected early lin late) "P0 openings")

end CompressionLean.Replay
