import CompressionLean.Circuit.Linear
import CompressionLean.Circuit.Openings
import CompressionLean.Circuit.Record
import CompressionLean.Circuit.Verifier
import CompressionLean.Circuit.Late
import CompressionLean.Circuit.Parse
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
The `lowerCheck` executable. It reads one export of `neo-spartan`'s
`tests/internal/lowering.rs` (the statement, the IR, then Rust's trace),
lowers the IR in Goldilocks and compares the run with the trace: the glue
cells, the blocks, the statement words and every glue row, side by side and
term by term (`Call.computedRows`, which is `Call.rows`). It prints `match`
and the counts, or the first difference and exits with 1.

With a program name it also writes that program with the Lean builder and
compares it with the IR, instruction by instruction, as `Log` writes them
(`line`):

- `shared`: the Lean side of `Shared`, every helper of `Algebra.lean`,
  `Hash.lean`, `Fold.lean` and `Sumcheck.lean`. The whole IR must match.
- `early`: the Lean side of `Early`, the helpers of `Gkr.lean` and
  `Early.lean` on a structure that the toy relation does not have. The whole
  IR must match.
- `layer1 <input.json>`: the Lean side of `Layer1::run` for the relation in
  the file. Its instructions must be the first ones of the IR: the verifier
  is ported up to `verifier.rs:266`.
- `linear <input.json>`: the Lean side of `Linear`, the linear stage
  (`linearStage`) from the state after the early stage. The whole IR must
  match.
- `honest <input.json>`: the Lean side of `Honest`, the honest fold
  (`honestStage`) and the query points from the state before `μ`. The whole
  IR must match.
- `late <input.json>`: the Lean side of `Late`, `lateStage` from the toy
  proof's state after the setup queries, for the relation and duplex sizes
  in the file. The whole IR must match.
- `lane_split`: the Lean side of `LaneSplit`, `laneTables` on classes that
  the toy relation does not have. The whole IR must match.
- `openings <input.json>`: the Lean side of `Openings`, both openings
  (`openingsStage`) from the state after the query points. The whole IR must
  match.
-/

namespace CompressionLean.LowerCheck

open CompressionLean.Shrink CompressionLean.Circuit CompressionLean.Circuit.Parse NightstreamFPrime.Spec
open CompressionLean.Replay (Gl)

instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

/-- Goldilocks: there the lowering's zero tests are Rust's. -/
abbrev G := ZMod goldilocksModulus

def wire : Parser Wire := do
  match ← nat with
  | 0 => pure (.glue (← nat))
  | 1 => do
      let b ← nat
      pure (.block b (← nat))
  | 2 => pure (.pub (← nat))
  | tag => throw s!"unknown wire tag {tag}"

def traceSide : Parser (List (Wire × ℕ)) := do
  many (← nat) do
    let w ← wire
    pure (w, ← nat)

/-! ## Lean's own programs -/

def handlesText {n : ℕ} (a : Fin n → Var) : String :=
  String.join ((List.finRange n).map fun k => s!" {a k}")

def termsText (terms : List (ℕ × Gl)) : String :=
  s!" {terms.length}" ++ String.join (terms.map fun t => s!" {t.1} {t.2.toNat}")

/-- The line that `Log` writes for an instruction (`lowering.rs`). -/
def line : Instr → String
  | .const value => s!"c {value.toNat}"
  | .priv => "p"
  | .pub => "u"
  | .linear parts => s!"l{termsText parts}"
  | .mul a b => s!"m {a} {b}"
  | .extMul a b => s!"x{handlesText a}{handlesText b}"
  | .extInv a => s!"i{handlesText a}"
  | .assertZero a => s!"z {a}"
  | .bits a count => s!"b {a} {count}"
  | .hint count => s!"h {count}"
  | .permute state => s!"q{handlesText state}"
  | .r1cs rows columns =>
      s!"r {rows.length} {columns.length}" ++ String.join (columns.map (s!" {·}")) ++
        String.join (rows.map fun row => termsText row.a ++ termsText row.b ++ termsText row.c)

/-- `Shared::run` (`lowering.rs`): every shared helper, in Rust's order. -/
def shared : Build Unit := do
  let words ← (Array.range 14).mapM fun _ => Build.priv
  let x ← privateExts 4
  let scaled ← Build.scale words[0]! 3
  let lifted ← Algebra.lift scaled
  let weighted ← Algebra.extScaleConstant lifted 9
  let dot ← Algebra.dot (x.extract 0 2) (x.extract 2 4)
  let eq ← Algebra.eqEval (x.extract 0 2) (x.extract 2 4)
  let table ← Algebra.eqTable (x.extract 0 2)
  let at_ ← Algebra.eqAt (x.extract 1 4) 5
  let value ← Algebra.interpolate table weighted
  let horner ← Algebra.horner x dot
  let k ← KxVar.fromK (scaled, words[1]!)
  let one ← KxVar.one
  let zero ← KxVar.zero
  let k ← k.mul ⟨eq, at_⟩
  let k ← k.add one
  let k ← k.scale horner
  let k ← k.oneMinus
  let k ← k.add zero
  let bits := Array.ofFn (← Build.bits words[2]! 5)
  let power ← Algebra.powerFromBits 7 bits
  let leaf ← hashLeaf words
  let short ← hashLeaf (words.extract 0 3)
  let node ← compress leaf short
  let root ← merkleRoot leaf (bits.extract 0 2) #[short, node]
  let rest : Transcript Unit := do
    Duplex.observeSlice words
    Duplex.observeExt k.re
    let first ← Duplex.sample
    let drawn ← Duplex.sampleExt
    let index ← Duplex.sampleBits 10
    let more ← (Array.range 3).mapM fun _ => Duplex.sampleExt
    Duplex.observe power
    let (point, last) ← sumcheckReplay #[#[x[0]!, x[1]!], #[x[2]!, x[3]!, value]] drawn
    let fold ← FoldTranscript.new
    let fold ← fold.absorb words
    let fold ← fold.absorbConstants #[1, 2, 3]
    let (digest, fold) ← fold.squeezeDigest
    let seed ← (FoldTranscript.fromState fun lane => words[lane.val % 14]!).handoff
    Build.assertExtEqual last point[1]!
    Build.assertExtEqual k.im more[2]!
    Build.assertEqual first index[3]!
    for lane in List.finRange 4 do
      Build.assertEqual (root lane) (seed lane)
      Build.assertEqual (digest lane) (fold.statePrefix lane)
  let _ ← rest.run (← Duplex.new)

/-- `Early::run` (`lowering.rs`): the early stage's helpers on a small
structure, in Rust's order. -/
def early : Build Unit := do
  let layout : Replay.Layout := { matrices := 3, rowVariables := 1, runVariables := 3, slotVariables := 1,
                                  runBounds := #[0, 2, 2, 8], classes := #[], segments := #[], slots := 2 }
  let shapes : Array Replay.TreeShape := #[⟨2, 0⟩, ⟨3, 0⟩, ⟨1, 1⟩, ⟨3, 2⟩, ⟨3, 1⟩]
  let histogram ← (Array.range 5).mapM fun _ => Build.priv
  let r ← (Array.range 3).mapM fun _ => do
    let re ← Build.priv
    pure (re, ← Build.priv)
  let view ← GkrView.read shapes
  let runs ← privateExts 3
  let pairs ← privatePairs layout.matrices
  let block ← Algebra.privateExt
  let values : EarlyValuesView := { runs, pairs, block }
  let rest : Transcript Unit := do
    Duplex.observeSlice histogram
    let beta ← Duplex.sampleExt
    let lookupBeta ← Duplex.sampleExt
    let gamma ← Duplex.sampleExt
    let scatterBeta ← Duplex.sampleExt
    let mu ← Duplex.sampleExt
    let challenges : EarlyChallengesView := { lookup := (lookupBeta, gamma), scatter := (scatterBeta, mu) }
    let total ← tableSum histogram 2 beta
    let claims ← gkrVerify view shapes
    let early ← checkEarly layout r challenges (claims.extract 1 claims.size) values
    (early.runPoint ++ early.pairPoint ++ early.rowPoint).forM Duplex.observeExt
    let (row, slot, coefficient) := early.setup
    Build.assertExtEqual slot total
    Build.assertExtEqual row coefficient
    Build.assertExtEqual early.mult claims[0]!.point[1]!
  let _ ← rest.run (← Duplex.new)

/-- What `layer1` reads (`lowering.rs`, `layer1.json`): the replay's relation
and the query sets of the two openings. -/
structure Layer1Input where
  relation : Replay.Relation
  p1Rows : OpeningRows
  p0Rows : OpeningRows
deriving Lean.FromJson

/-- `statement_input` (`tests/internal/verifier.rs`): every statement word,
in `ClaimWords::words` order. -/
def statementInput (rel : Replay.Relation) : Build ClaimView := do
  let ring := (Array.range Replay.D).mapM fun _ => Build.pub
  let pair : Build KVar := do
    let re ← Build.pub
    pure (re, ← Build.pub)
  let commitment ← (Array.range rel.kappa).mapM fun _ => ring
  let publicInput ← (Array.range rel.publicBlocks).mapM fun _ => ring
  let point ← (Array.range rel.pointVariables).mapM fun _ => pair
  let evalK ← (Array.range Replay.D).mapM fun _ => pair
  let evalA ← (Array.range rel.matrices).mapM fun _ => (Array.range Replay.D).mapM fun _ => pair
  pure { commitment, publicInput, point, evalK, evalA }

/-- `Layer1::run` (`tests/internal/shrink.rs`): the seed and the statement as
statement words, the proof view, then the verifier. -/
def layer1 (input : Layer1Input) : Build Unit := do
  let seed ← (Array.range 4).mapM fun _ => Build.pub
  let statement ← statementInput input.relation
  let proof ← ProofView.read input.relation input.p1Rows input.p0Rows
  verify input.relation (fun lane => seed[lane]!) statement proof

/-- What `linear`, `honest` and `late` read (`lowering.rs`, `linear.json`,
`honest.json` and `late.json`): the replay's relation, and the duplex inputs
pending and the output lanes unread at the stage's start. -/
structure StageInput where
  relation : Replay.Relation
  pending : ℕ
  output : ℕ
deriving Lean.FromJson

/-- `Linear::run` (`lowering.rs`): the statement as statement words; the
duplex state, its pending inputs and the linear stage's proof values as proof
words; then the linear stage. The unread output is the state's first lanes. -/
def linear (input : StageInput) : Build Unit := do
  let rel := input.relation
  let statement ← statementInput rel
  let state ← (Array.range 16).mapM fun _ => Build.priv
  let pending ← (Array.range input.pending).mapM fun _ => Build.priv
  let quotient ← privateExts (Replay.D - 1)
  let rounds ← (Array.range rel.cubeVars).mapM fun _ => privateExts 2
  let finals ← privateExts 3
  let _ ← (linearStage rel statement quotient rounds finals).run ⟨state, pending, state.extract 0 input.output⟩

/-- `Honest::run` (`lowering.rs`): the duplex and every value of
`HonestInput` as proof words, in its order; then the honest fold and each
query's point. The unread output is the state's first lanes. -/
def honest (input : StageInput) : Build Unit := do
  let rel := input.relation
  let state ← (Array.range 16).mapM fun _ => Build.priv
  let pending ← (Array.range input.pending).mapM fun _ => Build.priv
  let lanePoint ← privateExts Replay.laneVars
  let blockPoint ← privateExts rel.blockVars
  let runPoint ← privateExts rel.key.layout.runVariables
  let row ← Algebra.privateExt
  let slot ← Algebra.privateExt
  let setup := (row, slot, ← Algebra.privateExt)
  let block ← Algebra.privateExt
  let commitmentRows ← privateExts rel.kappa
  let keyValue ← Algebra.privateExt
  let partials ← (Array.range 2).mapM fun _ => privateExts (2 ^ Replay.foldVars)
  let p1 ← privateDigest
  let leaves ← (Array.range rel.queries).mapM fun _ =>
    LeafView.read (rel.kappa + Replay.runOracles) rel.setupVars
  let run : Transcript Unit := do
    let out ← honestStage rel
      { lanePoint, blockPoint, runPoint, setup, block, commitmentRows, keyValue, partials, p1, leaves }
    for bits in out.indexBits do
      let _ ← queryPoint rel.setupVars bits
  let _ ← run.run ⟨state, pending, state.extract 0 input.output⟩

/-- `Late::run` (`lowering.rs`): the duplex, `τ`, the Eval_A weights, the
late GKR proof and the late values as proof words, then `lateStage`, then
the late openings into the duplex. -/
def late (input : StageInput) : Build Unit := do
  let rel := input.relation
  let state ← (Array.range 16).mapM fun _ => Build.priv
  let pending ← (Array.range input.pending).mapM fun _ => Build.priv
  let tau ← privateExts Replay.D
  let evalA ← privatePairs rel.matrices
  let view ← GkrView.read rel.lateShapes
  let lateValues ← privatePairs rel.matrices
  let rest : Transcript Unit := do
    let late ← lateStage rel tau evalA view lateValues
    (late.pairPoint ++ #[late.block] ++ late.blockPoint ++ #[late.omega]).forM Duplex.observeExt
  let _ ← rest.run { state, input := pending, output := state.extract 0 input.output }

/-- `LaneSplit::run` (`lowering.rs`): the lane tables of a run of `D` lanes
with ratio 3 and of a run of 30 lanes with ratio `p - 2`, then every value
into a duplex. -/
def laneSplit : Build Unit := do
  let tau ← privateExts Replay.D
  let classes := #[⟨Replay.D, 3⟩, ⟨30, Gl.ofNat (goldilocksModulus - 2)⟩]
  let layout : Replay.Layout := { matrices := 1, rowVariables := 1, runVariables := 1, slotVariables := 1,
                                  runBounds := #[0, 1], classes, segments := #[], slots := 1 }
  let lanes ← laneTables layout tau
  let rest : Transcript Unit := do
    for (head, tail) in lanes do
      (head ++ tail).forM Duplex.observeExt
    let _ ← Duplex.sample
  let _ ← rest.run (← Duplex.new)

/-- What `openings` reads (`lowering.rs`, `openings.json`): the replay's
relation, the `WhirKey` of each opening, and the duplex inputs pending and the output lanes unread after the
query points. -/
structure OpeningsJson where
  relation : Replay.Relation
  p1Key : WhirKey
  p0Key : WhirKey
  pending : ℕ
  output : ℕ
deriving Lean.FromJson

/-- `Openings::run` (`lowering.rs`): the duplex state, its pending inputs,
the stage values and the proof parts as proof words, in `OpeningsInput`
order; then both openings. The unread output is the state's first lanes. -/
def openings (input : OpeningsJson) : Build Unit := do
  let rel := input.relation
  let layout := rel.key.layout
  let state ← (Array.range 16).mapM fun _ => Build.priv
  let pending ← (Array.range input.pending).mapM fun _ => Build.priv
  let queryPoints ← (Array.range rel.queries).mapM fun _ => privateExts (rel.setupVars - Replay.foldVars)
  let keyPoint ← privateExts rel.setupVars
  let runPoint ← privateExts rel.setupVars
  let blockPoint ← privateExts rel.blockVars
  let lateBlockPoint ← privateExts rel.blockVars
  let alpha ← privateExts Replay.foldVars
  let queried ← privateExts (2 * rel.queries)
  let omegaValue ← Algebra.privateExt
  let lateOmega ← Algebra.privateExt
  let normPoint ← privateExts rel.cubeVars
  let linearPoint ← privateExts rel.cubeVars
  let earlyRunPoint ← privateExts layout.runVariables
  let pairPoint ← privateExts layout.slotVariables
  let latePairPoint ← privateExts layout.slotVariables
  let rowPoint ← privateExts layout.rowVariables
  let zLeaf ← Algebra.privateExt
  let zValue ← Algebra.privateExt
  let lateBlock ← Algebra.privateExt
  let mult ← Algebra.privateExt
  let p1 ← privateDigest
  let p0 ← privateDigest
  let partials ← (Array.range 2).mapM fun _ => privateExts (2 ^ Replay.foldVars)
  let runs ← privateExts 3
  let pairs ← privatePairs rel.matrices
  let block ← Algebra.privateExt
  let lateValues ← privatePairs rel.matrices
  let p1Opening ← OpeningView.read rel.p1Whir input.p1Key.rows
  let p0Opening ← OpeningView.read rel.p0Whir input.p0Key.rows
  let stage := openingsStage rel input.p1Key input.p0Key
    { queryPoints, keyPoint, runPoint, blockPoint, lateBlockPoint, alpha, queried, omegaValue, lateOmega,
      normPoint, linearPoint, earlyRunPoint, pairPoint, latePairPoint, rowPoint, zLeaf, zValue, lateBlock,
      mult, p1, p0, partials, earlyValues := { runs, pairs, block }, lateValues, p1Opening, p0Opening }
  let _ ← stage.run ⟨state, pending, state.extract 0 input.output⟩

/-- Lean's program for an export, and whether it is the whole program or its
start. -/
structure Mirror where
  instrs : Array Instr
  whole : Bool

/-- Lean's instructions against Rust's, one by one. -/
def compareMirror (mirror : Mirror) (rust : Array Instr) : Except String String := do
  for (instr, k) in mirror.instrs.zipIdx do
    match rust[k]? with
    | none => throw s!"instruction {k}: Lean `{line instr}`, Rust's program has ended"
    | some r => unless line instr == line r do throw s!"instruction {k}: Lean `{line instr}`, Rust `{line r}`"
  if mirror.whole && mirror.instrs.size != rust.size then
    throw s!"Lean writes {mirror.instrs.size} instructions, Rust {rust.size}"
  pure s!"Lean's program is {if mirror.whole then "all" else "the first"} {mirror.instrs.size} of Rust's \
    {rust.size} instructions"

/-- Compare `mirror` with the IR, then the lowered run with the trace that
follows the program. -/
def check (mirror : Option Mirror) : Parser String := do
  keyword "statement"
  let stmt ← many (← nat) glWord
  let instrs ← program
  let mirrored ← match mirror.map (compareMirror · instrs) with
    | none => pure ""
    | some (.ok message) => pure s!"; {message}"
    | some (.error message) => throw message
  let run := lower (F := G) instrs (stmt.map fun g => (g.toNat : G))
  keyword "cells"
  expect "glue cells" run.cells (← nat)
  keyword "blocks"
  expect "blocks" run.blocks (← nat)
  keyword "publics"
  let publics ← many (← nat) nat
  expect "statement words" (run.publics.map ZMod.val) publics
  keyword "rows"
  let count ← nat
  let rows := (run.calls.flatMap Call.computedRows).toArray
  expect "glue rows" rows.size count
  for h : r in [0:rows.size] do
    for s in List.finRange 7 do
      expect s!"row {r} side {s}" ((rows[r]'h.2.1).side s |>.map fun t => (t.1, t.2.val)) (← traceSide)
  pure s!"match: {instrs.size} instructions, {rows.size} rows, {run.cells} glue cells, {run.blocks} blocks, \
    {run.publics.length} statement words{mirrored}"

/-- The export's path and Lean's program for it, from the arguments. -/
def arguments : List String → IO (Except String (String × Option Mirror))
  | [path] => pure (.ok (path, none))
  | [path, "shared"] => pure (.ok (path, some ⟨shared.program, true⟩))
  | [path, "early"] => pure (.ok (path, some ⟨early.program, true⟩))
  | [path, "layer1", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := Layer1Input)).map fun input =>
        (path, some ⟨(layer1 input).program, false⟩)
  | [path, "linear", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(linear input).program, true⟩)
  | [path, "honest", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(honest input).program, true⟩)
  | [path, "late", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := StageInput)).map fun input =>
        (path, some ⟨(late input).program, true⟩)
  | [path, "lane_split"] => pure (.ok (path, some ⟨laneSplit.program, true⟩))
  | [path, "openings", input] => do
      let json ← IO.FS.readFile input
      pure <| (Lean.Json.parse json >>= Lean.fromJson? (α := OpeningsJson)).map fun input =>
        (path, some ⟨(openings input).program, true⟩)
  | _ => pure (.error "usage: lowerCheck <export.txt> [shared | early | layer1 <layer1.json> | \
      linear <linear.json> | honest <honest.json> | late <late.json> | lane_split | openings <openings.json>]")

end CompressionLean.LowerCheck

open CompressionLean.LowerCheck in
def main (args : List String) : IO UInt32 := do
  let (path, mirror) ← match ← arguments args with
    | .ok parsed => pure parsed
    | .error message => IO.eprintln message; return 2
  match ((check mirror).run 0).run (← IO.FS.readBinFile path) with
  | .ok (message, _) => IO.println message; return 0
  | .error message => IO.println s!"mismatch: {message}"; return 1
