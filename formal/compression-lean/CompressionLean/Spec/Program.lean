import CompressionLean.Spec.Parent
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Ports.Terminal.Program
import CompressionLean.Lowering.Sound

/-!
Owns the production final program (`FinalProgram::run`,
`lifecycle/terminal/mod.rs:54-80`, with `verify_with`,
`neo-spartan/src/lib.rs:692-703`) for a layer-1 key, and what layer 1 reads
from a witness of it.

- `productionProgram`: the program. `shrinkRun`: its lowering at a
  statement, the run that the shrink verifier checks. `counts`: the counts of
  that run, the same at every statement.
- `wordsOf`: the words of the executor's run of the program on a statement
  and a witness, by handle.
- `statementWith`, `seedWith`, `proofWith`: the statement, the seed and the
  proof that layer 1 reads from those words, through the handles that the
  program writes from the empty builder (`ClaimView.decode`,
  `handoffWords`, `ProofView.decode`).
-/

namespace CompressionLean.Spec

open CompressionLean.Circuit CompressionLean.Terminal.Package CompressionLean.Terminal.Program
open CompressionLean.Lowering
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime

/-! ## Views read through words -/

/-- The extension value of three handles. -/
def decodeExt (w : Var → Word) (e : ExtVar) : Ext := ⟨w (e 0), w (e 1), w (e 2)⟩

/-- The four words of a digest. -/
def decodeDigest (w : Var → Word) (d : DigestVar) : Array Word := #[w (d 0), w (d 1), w (d 2), w (d 3)]

/-- A pair of extension values. -/
def decodePair (w : Var → Word) (p : ExtVar × ExtVar) : Ext × Ext := (decodeExt w p.1, decodeExt w p.2)

/-- One GKR layer. -/
def _root_.CompressionLean.Circuit.GkrLayerView.decode (w : Var → Word) (l : GkrLayerView) : Replay.GkrLayer :=
  { rounds := l.rounds.map (·.map (decodeExt w)), children := l.children.map (·.map (decodeExt w)) }

/-- A GKR proof. -/
def _root_.CompressionLean.Circuit.GkrView.decode (w : Var → Word) (v : GkrView) : Replay.GkrProof :=
  { roots := v.roots.map (decodePair w), layers := v.layers.map (·.decode w) }

/-- A queried setup leaf. -/
def _root_.CompressionLean.Circuit.LeafView.decode (w : Var → Word) (v : LeafView) : Replay.Leaf :=
  { values := v.values.map w, path := v.path.map (decodeDigest w) }

/-- One WHIR round, without its query rows. -/
def _root_.CompressionLean.Circuit.RoundView.decode (w : Var → Word) (r : RoundView) : Replay.WhirRound :=
  { root := decodeDigest w r.root, ood := r.ood.map (decodeExt w), sumcheck := r.sumcheck.map (decodePair w) }

/-- One WHIR opening, without its query rows. -/
def _root_.CompressionLean.Circuit.OpeningView.decode (w : Var → Word) (v : OpeningView) : Replay.Opening :=
  { initialOod := v.initialOod.map (decodeExt w), evals := v.evals.map (·.map (decodeExt w)),
    initialSumcheck := v.initialSumcheck.map (decodePair w), rounds := v.rounds.map (·.decode w),
    finalPoly := v.finalPoly.map (decodeExt w), finalSumcheck := v.finalSumcheck.map (decodePair w) }

/-- A layer-1 proof. The run values are `e_re`, `e_im`, `row`, the order of
`observe_early` and of the replay's `earlyStage`. -/
def _root_.CompressionLean.Circuit.ProofView.decode (w : Var → Word) (v : ProofView) : Replay.Proof :=
  { p0 := decodeDigest w v.p0, histogram := v.histogram.map w, early := v.early.decode w,
    earlyValues :=
      { eRe := decodeExt w v.earlyValues.runs[0]!, eIm := decodeExt w v.earlyValues.runs[1]!,
        row := decodeExt w v.earlyValues.runs[2]!, pairs := v.earlyValues.pairs.map (decodePair w),
        block := decodeExt w v.earlyValues.block },
    quotient := v.quotient.map (decodeExt w), linear := v.linear.map (·.map (decodeExt w)),
    finals := v.finals.map (decodeExt w), partials := v.partials.map (·.map (decodeExt w)),
    p1 := decodeDigest w v.p1, leaves := v.leaves.map (·.decode w), late := v.late.decode w,
    lateValues := v.lateValues.map (decodePair w), p1Opening := v.p1Opening.decode w,
    p0Opening := v.p0Opening.decode w }

/-- The terminal claim as the verifier's statement. -/
def _root_.CompressionLean.Terminal.Program.ClaimVars.view (claim : ClaimVars) : ClaimView :=
  { commitment := claim.commitment, publicInput := claim.publicInput, point := claim.point,
    evalK := claim.evalK, evalA := claim.evalA }

/-- The statement of a claim view: each handle reads as its word under `w`. -/
def _root_.CompressionLean.Circuit.ClaimView.decode (w : Var → Word) (v : ClaimView) : Replay.Statement :=
  { commitment := v.commitment.map (·.map w), publicInput := v.publicInput.map (·.map w),
    point := v.point.map (Prod.map w w), evalK := v.evalK.map (Prod.map w w),
    evalA := v.evalA.map (·.map (Prod.map w w)) }

section FoldTranscript

open CompressionLean.Circuit.FoldTranscript

/-- One chunk added into the rate lanes, then a permutation (`fold.rs:39-46`). -/
def _root_.CompressionLean.Circuit.FoldTranscript.absorbChunk (state chunk : Array Word) : Array Word :=
  Replay.permute ((List.range chunk.size).foldl (fun s i => s.set! i (s[i]! + chunk[i]!)) state)

/-- `absorb` on words. -/
def _root_.CompressionLean.Circuit.FoldTranscript.absorbWords (state words : Array Word) : Array Word :=
  (List.range ((words.size + rate - 1) / rate)).foldl
    (fun s k => absorbChunk s (words.extract (rate * k) (rate * k + rate))) state

/-- The seed that `handoff` returns: the first four lanes after the layer-1
domain chunk (`fold.rs:68-72`). -/
def _root_.CompressionLean.Circuit.FoldTranscript.handoffWords (state : Array Word) : Array Word :=
  (absorbWords state (domainChunk compressDomain)).extract 0 4

end FoldTranscript

/-! ## The program -/

/-- The nine `public` words of the statement (`terminal/mod.rs:55-59`). -/
def statementHandles : Build (Array Var) := (Array.range statementWords).mapM fun _ => Build.pub

/-- The iteration check, then `Package::apply` (`terminal/mod.rs:60-72`). -/
def checkedApply (package : Loaded) (context : Array Word) (statement : Array Var) :
    Build (ClaimVars × (Fin 16 → Var)) := do
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  apply package statement context

/-- `FinalProgram::run` up to `verify_with` (`terminal/mod.rs:54-72`). -/
def terminalPrefix (package : Loaded) (context : Array Word) : Build (ClaimVars × (Fin 16 → Var)) :=
  statementHandles >>= checkedApply package context

section Runs

variable (key : Layer1Key) (loaded : Loaded) (context : Array Replay.Word)

/-- `FinalProgram::run` with `verify_with` for `key` (`lifecycle/terminal/mod.rs:54-80`).
Irreducible, so that no check evaluates the program. -/
@[irreducible] def productionProgram : Build Unit :=
  finalProgram loaded context fun claim state => verifyWith key state claim.view

/-- The shrink run of the final program for the statement `x`: its lowering. -/
noncomputable def shrinkRun (x : StateWords) : Shrink.Run Spec.F :=
  lower (productionProgram key loaded context).program (List.ofFn x)

/-- The counts of the final program's shrink runs, which do not depend on
the statement (`lower_counts`). -/
noncomputable def counts : Shrink.Counts :=
  (lower (productionProgram key loaded context).program ([] : List Spec.F)).counts

/-- The words of the executor's run of the final program on `x` and the
witness `w`, by handle. -/
noncomputable def wordsOf (x : StateWords) (w : ℕ → Spec.F) : Var → Replay.Word :=
  let L := (productionProgram key loaded context).program
  let stmt := List.ofFn x
  fun h => Replay.Word.ofF
    ((Exec.run Spec.Poseidon2.permute stmt (tapeOf Block.poseidonTemplate L stmt w) L).value h)

/-- The builder after the terminal part (`lifecycle/terminal/mod.rs:54-72`). -/
def prefixRun : (ClaimVars × (Fin 16 → Var)) × Builder := (terminalPrefix loaded context).run {}

/-- The builder after the handoff (`neo-spartan/src/lib.rs:697`). -/
def seedRun : DigestVar × Builder :=
  (FoldTranscript.fromState (prefixRun loaded context).1.2).handoff.run (prefixRun loaded context).2

/-- The builder after the proof view (`neo-spartan/src/lib.rs:698-702`). -/
def viewRun : Circuit.ProofView × Builder :=
  (Circuit.ProofView.read key.relation key.p1Key.rows key.p0Key.rows).run (seedRun loaded context).2

/-- The statement that layer 1 reads, through `w`. -/
def statementWith (w : Var → Replay.Word) : Replay.Statement := (prefixRun loaded context).1.1.view.decode w

/-- The seed that layer 1 reads, through `w`. -/
def seedWith (w : Var → Replay.Word) : Array Replay.Word :=
  FoldTranscript.handoffWords (Array.ofFn fun lane => w ((prefixRun loaded context).1.2 lane))

/-- The proof that layer 1 reads, through `w`. -/
def proofWith (w : Var → Replay.Word) : Replay.Proof := (viewRun key loaded context).1.decode w

end Runs

/-- The empty replay transcript, where `verify` starts (`verifier.rs:136`). -/
def start : Replay.Transcript := ⟨Replay.Duplex.new, #[]⟩

end CompressionLean.Spec
