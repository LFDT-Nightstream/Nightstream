import CompressionLean.Ports.Circuit.Linear
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Ports.Circuit.Late

/-!
Owns the Lean side of the programs that `neo-spartan`'s
`tests/internal/lowering_stages.rs` records, each written with the Lean
builder in Rust's call order. `lowerCheck` compares each with Rust's IR.
-/

namespace CompressionLeanExe.Fixtures

open CompressionLean CompressionLean.Circuit NightstreamFPrime.Spec
open CompressionLean.Replay (Word)

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
def layer1 (input : Layer1Key) : Build Unit := do
  let seed ← (Array.range 4).mapM fun _ => Build.pub
  let statement ← statementInput input.relation
  let proof ← ProofView.read input.relation input.p1Key.rows input.p0Key.rows
  verify input.relation input.p1Key input.p0Key (fun lane => seed[lane]!) statement proof

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
  let classes := #[⟨Replay.D, 3⟩, ⟨30, Word.ofNat (goldilocksModulus - 2)⟩]
  let layout : Replay.Layout := { matrices := 1, rowVariables := 1, runVariables := 1, slotVariables := 1,
                                  runBounds := #[0, 1], classes, segments := #[], slots := 1 }
  let lanes ← laneTables layout tau
  let rest : Transcript Unit := do
    for (head, tail) in lanes do
      (head ++ tail).forM Duplex.observeExt
    let _ ← Duplex.sample
  let _ ← rest.run (← Duplex.new)

/-- What `openings` reads (`lowering.rs`, `openings.json`): the `layer1`
input, and the duplex inputs pending and the output lanes unread after the
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

end CompressionLeanExe.Fixtures
