import CompressionLean.Circuit.Views

/-!
Owns our WHIR verifier of one opening (`crates/neo-spartan/src/whir/mod.rs`)
as a builder program: `whirVerify` (`whir/mod.rs:506-661`) with its helpers
in the `Whir` namespace. Each one calls `Build` in the order of its Rust
counterpart. Points are in Plonky3 order inside `whirVerify`; its callers
pass low-bit-first points.

The transcript constants (every separator of `whir/seeds.rs` and every draw
count) are the replay's `WhirShape`. What the verifier reads of the `Pcs`
beyond them is `WhirKey`: the stacked layout, the opening batches, each
round's parameters and the query sets. The Merkle paths are a hint
(`whir/paths.rs`), so the program reads only the hint's length.
-/

namespace CompressionLean.Circuit

open Lean (FromJson)
open CompressionLean.Replay (Gl Ext WhirShape SumcheckShape twoAdicGenerator)

/-- One column's slot in the stacked polynomial: `numVariables` selector bits
with value `index`, high bit first (`Selector`). -/
structure WhirSelector where
  numVariables : ℕ
  index : ℕ
deriving FromJson, Inhabited

/-- Table `table`'s selector for each of its columns (`TablePlacement`). -/
structure WhirPlacement where
  table : ℕ
  selectors : Array WhirSelector
deriving FromJson, Inhabited

/-- One opening batch: its table and its columns (`iter_openings`). -/
structure WhirBatchKey where
  table : ℕ
  columns : Array ℕ
deriving FromJson, Inhabited

/-- One round's `num_variables` and `log_folded_domain_size`
(`round_parameters`). -/
structure WhirRoundKey where
  numVariables : ℕ
  logFoldedDomain : ℕ
deriving FromJson, Inhabited

/-- What `whir::verify` reads of a `Pcs` beyond `WhirShape`: the stacked
variable count `k` and the placements of `plan_stacked_layout`, each table's
arity, the batches, the rounds, the final folded domain and the query sets. -/
structure WhirKey where
  numVariables : ℕ
  arities : Array ℕ
  batches : Array WhirBatchKey
  placements : Array WhirPlacement
  rounds : Array WhirRoundKey
  finalLogFoldedDomain : ℕ
  rows : OpeningRows
deriving FromJson, Inhabited

namespace Whir

/-- Each word as a constant, observed (`absorb`, `whir/mod.rs:221-226`). -/
def absorb (words : Array Gl) : Transcript Unit := do
  for word in words do
    Duplex.observe (← Build.const word)

/-- `(x^{2^{n-1}}, …, x², x)`; the last square is computed and unused
(`expand`, `whir/mod.rs:234-244`). -/
def expand (x : ExtVar) (n : ℕ) : Build (Array ExtVar) := do
  let mut point := #[]
  let mut current := x
  for _ in [0:n] do
    point := point.push current
    current ← Build.extMul current current
  pure point.reverse

/-- `Π (2·q·p − p − q + 1)` (`eval_eq`, `whir/mod.rs:246-260`). -/
def evalEq (p q : Array ExtVar) : Build ExtVar := do
  let one ← Algebra.extOne
  let mut total := one
  for (p, q) in p.zip q do
    let product ← Build.extMul p q
    let twice ← Build.extAdd product product
    let sum ← Build.extAdd p q
    let difference ← Build.extSub twice sum
    let term ← Build.extAdd difference one
    total ← Build.extMul total term
  pure total

/-- `Π_j (1 + r_j·(var^{2^j} − 1))` over `point` from the last coordinate
(`eval_select`, `whir/mod.rs:262-277`). -/
def evalSelect (var : Var) (point : Array ExtVar) : Build ExtVar := do
  let one ← Algebra.extOne
  let mut power := var
  let mut total := one
  for r in point.reverse do
    let oneBase ← Build.const 1
    let minus ← Build.sub power oneBase
    let scaled ← Build.extScale r minus
    let term ← Build.extAdd scaled one
    total ← Build.extMul total term
    power ← Build.mul power power
  pure total

/-- The multilinear extension of `values` at `r`, low bit first
(`mle`, `whir/mod.rs:279-294`). -/
def mle (values r : Array ExtVar) : Build ExtVar := do
  let mut layer := values
  for x in r do
    let mut next := #[]
    for c in [0:layer.size / 2] do
      let difference ← Build.extSub layer[2 * c + 1]! layer[2 * c]!
      let step ← Build.extMul x difference
      next := next.push (← Build.extAdd layer[2 * c]! step)
    layer := next
  pure layer[0]!

/-- `Σ_i values_i·var^i` (`horner`, `whir/mod.rs:296-304`). -/
def horner (values : Array ExtVar) (var : Var) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for value in values.reverse do
    let scaled ← Build.extScale total var
    total ← Build.extAdd scaled value
  pure total

/-- A quadratic sum-check: the separator, then per round `(h(0), h(∞))`, a
draw `r` and `h(r) = h(0)(1−r) + h(1)r + h(∞)r(r−1)`. No rounds, no calls
(`sumcheck`, `whir/mod.rs:306-332`). Returns the draws and the claim. -/
def sumcheck (shape : SumcheckShape) (rounds : Array (ExtVar × ExtVar)) (claim : ExtVar) :
    Transcript (Array ExtVar × ExtVar) := do
  if rounds.isEmpty then return (#[], claim)
  absorb shape.seed
  let one ← Algebra.extOne
  let mut claim := claim
  let mut point := #[]
  for (atZero, atInfinity) in rounds do
    Duplex.observeExt atZero
    Duplex.observeExt atInfinity
    let r ← Duplex.sampleExt
    let atOne ← Build.extSub claim atZero
    let oneMinus ← Build.extSub one r
    let rMinus ← Build.extSub r one
    let quadratic ← Build.extMul r rMinus
    let a ← Build.extMul atZero oneMinus
    let c ← Build.extMul atOne r
    let d ← Build.extMul atInfinity quadratic
    let sum ← Build.extAdd a c
    claim ← Build.extAdd sum d
    point := point.push r
  pure (point, claim)

/-- A batch of constraints: a challenge, its first power, and eq and select
statements with their claimed values (`Constraint`, `whir/mod.rs:334-342`). -/
structure Constraint where
  challenge : ExtVar
  initialPower : ℕ
  numVariables : ℕ
  eq : Array (Array ExtVar × ExtVar)
  select : Array (Var × ExtVar)

namespace Constraint

/-- The challenge powers in statement order from `challenge^initialPower`;
the power after the last is computed and unused (`whir/mod.rs:345-358`). -/
def powers (c : Constraint) : Build (Array ExtVar) := do
  let mut power ← Algebra.extOne
  for _ in [0:c.initialPower] do
    power ← Build.extMul power c.challenge
  let mut powers := #[]
  for _ in [0:c.eq.size + c.select.size] do
    powers := powers.push power
    power ← Build.extMul power c.challenge
  pure powers

/-- `Σ value·power` (`claimed`, `whir/mod.rs:360-372`). -/
def claimed (c : Constraint) (powers : Array ExtVar) : Build ExtVar := do
  let mut total ← Algebra.extZero
  for (value, power) in (c.eq.map (·.2) ++ c.select.map (·.2)).zip powers do
    let term ← Build.extMul value power
    total ← Build.extAdd total term
  pure total

/-- The batched weight at the last `numVariables` folding challenges, reversed
(`weight`, `whir/mod.rs:374-396`). -/
def weight (c : Constraint) (powers randomness : Array ExtVar) : Build ExtVar := do
  let tail := randomness.reverse.extract 0 c.numVariables
  let mut total ← Algebra.extZero
  for ((point, _), power) in c.eq.zip powers do
    let weight ← evalEq point tail
    let term ← Build.extMul weight power
    total ← Build.extAdd total term
  for ((var, _), power) in c.select.zip (powers.extract c.eq.size powers.size) do
    let weight ← evalSelect var tail
    let term ← Build.extMul weight power
    total ← Build.extAdd total term
  pure total

end Constraint

/-- One query index of `width` bits: reject the sample `p − 1` by inverting
it plus one, then the low bits of its 64 bits and their sum
(`query_index`, `whir/mod.rs:399-416`). -/
def queryIndex (width : ℕ) : Transcript (Array Var × Var) := do
  let sample ← Duplex.sample
  let one ← Build.const 1
  let shifted ← Build.add sample one
  let zeroWord ← Build.const 0
  let _ ← Build.extInv ![shifted, zeroWord, zeroWord]
  let bits := (Array.ofFn (← Build.bits sample 64)).extract 0 width
  let mut index ← Build.const 0
  for (bit, i) in bits.zipIdx do
    let weighted ← Build.scale bit (Gl.ofNat (2 ^ i))
    index ← Build.add index weighted
  pure (bits, index)

/-- The STIR checks of one query set against `root`: the indices (every one,
as constants, when `draws` is zero), a hint of the full paths, then per query
its leaf, its path, the fold at `folding` and its domain point
(`stir`, `whir/mod.rs:418-504`). The branch is on the key's draw count.
Returns the select statements. -/
def stir (queries : Array (Array Var)) (draws indexBits logFoldedDomain : ℕ) (root : DigestVar)
    (folding : Array ExtVar) (base : Bool) : Transcript (Array (Var × ExtVar)) := do
  let mut indices : Array (Array Var × Var) := #[]
  if draws = 0 then
    for index in [0:2 ^ indexBits] do
      let bits ← (Array.range indexBits).mapM fun i => Build.const (Gl.ofNat ((index >>> i) % 2))
      let word ← Build.const (Gl.ofNat index)
      indices := indices.push (bits, word)
  else
    for _ in [0:draws] do
      indices := indices.push (← queryIndex indexBits)
  let pathWords := Array.ofFn (← Build.hint (indices.size * indexBits * 4))
  let generator := twoAdicGenerator logFoldedDomain
  let mut statements := #[]
  for ((bits, _), d) in indices.zipIdx do
    let row := queries[d]!
    let leaf ← hashLeaf row
    let path : Array DigestVar := (Array.range indexBits).map fun level lane =>
      pathWords[(d * indexBits + level) * 4 + lane]!
    let computed ← merkleRoot leaf bits path
    for lane in List.finRange 4 do
      Build.assertEqual (computed lane) (root lane)
    let values ← if base then
        row.mapM fun value => do
          let zeroWord ← Build.const 0
          pure (![value, zeroWord, zeroWord] : ExtVar)
      else pure ((Array.range (row.size / 3)).map fun c => ![row[3 * c]!, row[3 * c + 1]!, row[3 * c + 2]!])
    let fold ← mle values folding
    let var ← Algebra.powerFromBits generator bits
    statements := statements.push (var, fold)
  pure statements

end Whir

open Whir in
/-- `whir::verify` (`whir/mod.rs:506-661`): the out-of-domain claims, the
opening claims per table, the batching challenge and the initial sum-check;
per round its root, out-of-domain claims, STIR queries against the previous
root and sum-check; the final polynomial, its queries and sum-check; then
the final claim. `points` are one per batch, low bit first. Returns the
opened batch values. -/
def whirVerify (shape : WhirShape) (key : WhirKey) (root : DigestVar) (view : OpeningView)
    (points : Array (Array ExtVar)) : Transcript (Array (Array ExtVar)) := do
  let mut virtualClaims := #[]
  for answer in view.initialOod do
    absorb shape.virtualSeed
    let x ← Duplex.sampleExt
    let point ← expand x key.numVariables
    Duplex.observeExt answer
    virtualClaims := virtualClaims.push (point, answer)
  let mut claims : Array (Array (ℕ × Array ExtVar × ExtVar)) := Array.replicate key.arities.size #[]
  for (batch, i) in key.batches.zipIdx do
    let values := view.evals[i]!
    let point := points[i]!
    absorb shape.batches[i]!.seed
    for value in values do
      Duplex.observeExt value
    let arity := key.arities[batch.table]!
    let zeroWord ← Algebra.extZero
    let padded := (point ++ Array.replicate (arity - point.size) zeroWord).reverse
    for (column, value) in batch.columns.zip values do
      claims := claims.modify batch.table (·.push (column, padded, value))
  absorb shape.whirSeed
  absorb shape.batchingSeed
  let alpha ← Duplex.sampleExt
  let mut eq := #[]
  for placement in key.placements do
    for (column, point, value) in claims[placement.table]! do
      let selector := placement.selectors[column]!
      let mut lifted := #[]
      for i in [0:selector.numVariables] do
        let bit := (selector.index >>> (selector.numVariables - 1 - i)) % 2
        lifted := lifted.push (← Algebra.extConstant (Ext.ofGl (Gl.ofNat bit)))
      eq := eq.push (lifted ++ point, value)
  let first : Constraint := { challenge := alpha, initialPower := 0, numVariables := key.numVariables,
                              eq := eq ++ virtualClaims, select := #[] }
  let powers ← first.powers
  let mut claimed ← first.claimed powers
  let mut constraints := #[(first, powers)]
  let (initial, next) ← sumcheck shape.initialSumcheck view.initialSumcheck claimed
  claimed := next
  let mut randomness := initial
  let mut lastFolding := initial
  let mut previousRoot := root
  for ((round, roundView), r) in (shape.rounds.zip view.rounds).zipIdx do
    let parameters := key.rounds[r]!
    Duplex.observeSlice (Array.ofFn roundView.root)
    let mut ood := #[]
    for answer in roundView.ood do
      let x ← Duplex.sampleExt
      let point ← expand x parameters.numVariables
      Duplex.observeExt answer
      ood := ood.push (point, answer)
    let select ← stir roundView.queries round.queryDraws key.rows.rounds[r]!.indexBits
      parameters.logFoldedDomain previousRoot lastFolding (r == 0)
    let beta ← Duplex.sampleExt
    let constraint : Constraint := { challenge := beta, initialPower := 1,
                                     numVariables := parameters.numVariables, eq := ood, select }
    let powers ← constraint.powers
    let added ← constraint.claimed powers
    claimed ← Build.extAdd claimed added
    constraints := constraints.push (constraint, powers)
    let (folding, next) ← sumcheck round.sumcheck roundView.sumcheck claimed
    claimed := next
    lastFolding := folding
    randomness := randomness ++ folding
    previousRoot := roundView.root
  for value in view.finalPoly do
    Duplex.observeExt value
  let finalSelect ← stir view.finalQueries shape.finalQueryDraws key.rows.final.indexBits
    key.finalLogFoldedDomain previousRoot lastFolding shape.rounds.isEmpty
  for (var, value) in finalSelect do
    let evaluated ← horner view.finalPoly var
    Build.assertExtEqual evaluated value
  let (finalRandomness, next) ← sumcheck shape.finalSumcheck view.finalSumcheck claimed
  claimed := next
  randomness := randomness ++ finalRandomness
  let mut weights ← Algebra.extZero
  for (constraint, powers) in constraints do
    let weight ← constraint.weight powers randomness
    weights ← Build.extAdd weights weight
  let finalValue ← mle view.finalPoly finalRandomness
  let expected ← Build.extMul weights finalValue
  Build.assertExtEqual claimed expected
  pure view.evals

end CompressionLean.Circuit
