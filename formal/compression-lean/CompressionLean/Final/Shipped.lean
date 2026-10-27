import CompressionLean.Final.ParentClaim
import CompressionLean.Bridge.Main
import CompressionLean.Bridge.Merkle
import CompressionLean.Circuit.Final
import CompressionLean.Circuit.Sound
import CompressionLean.Circuit.Valid
import CompressionLean.Circuit.Counts
import CompressionLean.Final.Wiring

/-!
Owns the production bound of the shipped final program. `ProgramEncodes`
and `FinalRestoreBreak` take a program and a reading relation as
parameters; this file fixes both.

- `Shipped`: the loaded terminal package, the layer-1 key and the verifier
  context's authority. Its program is the lowered `finalViews` (`program`,
  `prog`, `counts`). Its game is the bridge's `layer1Game` (`game`).
- `decode`: the seed, the statement and the proof that the program's
  handles hold under a witness: the executor's values on the witness's tape.
- `Reads`: a reading is the bridge's path of the replay's run, from `start`,
  on what the witness decodes to.
- `TerminalWiring` (the decoded statement is the terminal claim) is proved
  (`terminalWiring`, from `wiring_of_exec` in `Wiring.lean`). One open
  premise: `CircuitAgrees` (the replay accepts what the circuit accepts).
- `shipped_restore_budget`: the production bound for `ShippedBreak`. Under
  the honest key root and the relation facts, its read condition fails only
  when WHIR lists are missing, Poseidon2 collides at explicit inputs on a
  setup leaf, or the layer-1 log did not walk the path (`reads_exist`).
-/

namespace CompressionLean.Final

open Polynomial NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-! ## The bridge's parameters over the concrete fields -/

/-- `Cubic` is the bridge's challenge field: its root `x` has `x³ = x + 1`. -/
noncomputable instance cubic_challengeField : Bridge.ChallengeField Cubic where
  root := AdjoinRoot.root cubic
  cubic := by
    have root : AdjoinRoot.mk cubic (X ^ 3 - X - 1) = 0 := AdjoinRoot.mk_self
    rw [map_sub, map_sub, map_pow, AdjoinRoot.mk_X, map_one] at root
    linear_combination root
  card := card_cubic

/-- The bridge's `Φ81` is F′'s modulus. -/
theorem phi_eq_modulus : Bridge.phi = modulus := by
  unfold Bridge.phi modulus
  rw [show ringDegree = 54 from rfl, show ringMiddleDegree = 27 from rfl, add_assoc]

/-- The bridge's `bar` is F′'s `Φ81` bar transform. -/
theorem barOf_eq (p l : Fin ringDegree) : Bridge.barOf p l = gl (nativeBarEntry p l) := by
  rw [Bridge.barOf, Replay.bar, dif_pos ⟨p.2, l.2⟩, Replay.Gl.toF_ofNat]
  exact ZMod.natCast_zmod_val (n := goldilocksModulus) _

/-! ## The shipped program -/

/-- A shipped final verifier: the terminal package that `Package.load` gives,
the layer-1 key and the authority of the verifier context. The CI digest
tests pin the key. The context words of a `terminalCheck` export must be the
authority's digest (`context`). -/
structure Shipped where
  package : Circuit.Package.Loaded
  load : Circuit.Package.load (Terminal.circuitPackage ()) Terminal.outputs = .ok package
  key : Circuit.Layer1Input
  authority : VerifierContext.Authority

/-- What layer 1 reads from a witness: the seed, the statement and the proof. -/
structure Decoded where
  seed : Array Replay.Gl
  statement : Replay.Statement
  proof : Replay.Proof

/-! ## Decoding a witness

A view's handles as replay data, under the value `v` of each handle. -/

namespace Decode

variable (v : Circuit.Var → Replay.Gl)

def ext (a : Circuit.ExtVar) : Replay.Ext := ⟨v (a 0), v (a 1), v (a 2)⟩

def exts (a : Array Circuit.ExtVar) : Array Replay.Ext := a.map (ext v)

def pairs (a : Array (Circuit.ExtVar × Circuit.ExtVar)) : Array (Replay.Ext × Replay.Ext) :=
  a.map fun p => (ext v p.1, ext v p.2)

def kValues (a : Array Circuit.KVar) : Array (Replay.Gl × Replay.Gl) := a.map fun p => (v p.1, v p.2)

def digest (d : Circuit.DigestVar) : Array Replay.Gl := Array.ofFn fun lane => v (d lane)

def gkr (g : Circuit.GkrView) : Replay.GkrProof :=
  ⟨pairs v g.roots, g.layers.map fun layer => ⟨layer.rounds.map (exts v), layer.children.map (exts v)⟩⟩

/-- An opening without its WHIR query rows, which the replay does not read. -/
def opening (o : Circuit.OpeningView) : Replay.Opening :=
  ⟨exts v o.initialOod, o.evals.map (exts v), pairs v o.initialSumcheck,
    o.rounds.map fun r => ⟨digest v r.root, exts v r.ood, pairs v r.sumcheck⟩, exts v o.finalPoly,
    pairs v o.finalSumcheck⟩

def statement (c : Circuit.ClaimView) : Replay.Statement :=
  ⟨c.commitment.map (·.map v), c.publicInput.map (·.map v), kValues v c.point, kValues v c.evalK,
    c.evalA.map (kValues v)⟩

/-- A proof, field by field. The early runs are `e_re`, `e_im` and the row
value, in the order that both verifiers observe them. -/
def proof (p : Circuit.ProofView) : Replay.Proof where
  p0 := digest v p.p0
  histogram := p.histogram.map v
  early := gkr v p.early
  earlyValues := ⟨ext v p.earlyValues.runs[0]!, ext v p.earlyValues.runs[1]!, ext v p.earlyValues.runs[2]!,
    pairs v p.earlyValues.pairs, ext v p.earlyValues.block⟩
  quotient := exts v p.quotient
  linear := p.linear.map (exts v)
  finals := exts v p.finals
  partials := p.partials.map (exts v)
  p1 := digest v p.p1
  leaves := p.leaves.map fun leaf => ⟨leaf.values.map v, leaf.path.map (digest v)⟩
  late := gkr v p.late
  lateValues := pairs v p.lateValues
  p1Opening := opening v p.p1Opening
  p0Opening := opening v p.p0Opening

end Decode

namespace Shipped

/-- The Poseidon2 block template over `Gl` (`Spec.F` has no global `Field`
instance). -/
abbrev template : Shrink.Template Gl := Circuit.Block.poseidonTemplate

/-- The replay's start: a new duplex and no draws (`Replay/Main.lean`). -/
def start : Replay.Transcript := ⟨Replay.Duplex.new, #[]⟩

variable (sp : Shipped)

/-- The context words: the authority's digest, which the terminal package
reads (`SelectedVerifierContext`). -/
def context : Array Replay.Gl := ((VerifierContext.digest sp.authority).map Circuit.Package.word).toArray

/-- The instructions of the final program. -/
def program : Array Circuit.Instr := (Circuit.finalViews sp.package sp.context sp.key).program

/-- The handles that layer 1 reads. -/
def views : Circuit.FinalViews := ((Circuit.finalViews sp.package sp.context sp.key).run {}).1

/-- The final program lowered on the statement `x`. -/
def prog (x : StateWords) : Shrink.Run Gl := Circuit.lower sp.program (List.ofFn x)

/-- The counts that the digest pins: the lowering on no statement words. -/
noncomputable def counts : Shrink.Counts := (Circuit.lower (F := Gl) sp.program []).counts

/-- The layer-1 game of the key's relation over `Gl` and `Cubic`, as the
bridge plays it. -/
noncomputable def game (S : Layer1.Setup Gl) (L0 L1 : ℕ) :=
  Layer1.layer1Game (Bridge.shapeOf sp.key.relation) Bridge.phi Bridge.barOf S L0 L1
    (2 ^ (sp.key.relation.setupVars - 2)) Layer1.goldilocks (Bridge.queryAt (K := Cubic) sp.key.relation)

/-- The executor's run of the final program on the statement `x` and the
tape of the witness `w`, as in `lower_sound`. -/
noncomputable def exec (x : StateWords) (w : ℕ → Gl) : Circuit.Exec.State Gl :=
  Circuit.Exec.run (F := Gl) Spec.Poseidon2.permute (List.ofFn x)
    (Circuit.tapeOf template sp.program (List.ofFn x) w) sp.program

/-- What layer 1 reads from the witness `w` for the statement `x`: the
executor's value of each handle, as a word. -/
noncomputable def decode (x : StateWords) (w : ℕ → Gl) : Decoded :=
  let v (h : Circuit.Var) : Replay.Gl := Circuit.Package.word ((sp.exec x w).value h)
  ⟨Decode.digest v sp.views.seed, Decode.statement v sp.views.claim, Decode.proof v sp.views.proof⟩

/-- The replay of what `w` decodes to. -/
noncomputable def replay (x : StateWords) (w : ℕ → Gl) : Replay.ReplayM Replay.OpeningPoints :=
  Replay.verify sp.key.relation (sp.decode x w).seed (sp.decode x w).statement (sp.decode x w).proof

/-- **A reading.** The path `p` is the bridge's path of the replay's run
from `start` on what `w` decodes to: it draws the replay's draws before
WHIR's, and it ends in a transcript of the decoded statement. -/
def Reads (S : Layer1.Setup Gl) (L0 L1 : ℕ) (x : StateWords) (w : ℕ → Gl) (p : (sp.game S L0 L1).Pos) :
    Prop :=
  ∃ points t' ds rest tr, Replay.Runs (sp.replay x w) start points t' ∧ t'.draws.toList = ds ++ rest ∧
    Layer1.Game.Pos.draws _ p = ds.map (Bridge.drawn (K := Cubic) sp.key.relation.queries) ∧
    Layer1.Game.Pos.node _ p = .pure (some tr) ∧ tr.statement = Bridge.statementOf (sp.decode x w).statement

/-- **The circuit-replay agreement.** Open: the executor-simulation units
will prove it. A witness of the final program's CCS for `x` gives an
accepting replay run, from `start`, on what it decodes to.

It also contains static facts that the replay checks and the circuit does
not: the statement fits the relation (`st.fits rel`), the proof has the
shapes that `readProof` checks, and the WHIR batches have their sizes. They
hold for a consistent key, because the views have the key's shapes. -/
def CircuitAgrees : Prop :=
  ∀ x w, ((sp.prog x).ccs template sp.counts).Relation w → ∃ points t', Replay.Runs (sp.replay x w) start points t'

/-- **The terminal wiring.** Proved by `terminalWiring`. A witness of the
final program's CCS for `x` holds a terminal assignment for `x` whose claim
(`claimOf`) is the decoded statement. -/
def TerminalWiring : Prop :=
  ∀ x w, ((sp.prog x).ccs template sp.counts).Relation w →
    ∃ env, TerminalFor sp.authority x env ∧ Bridge.statementOf (sp.decode x w).statement = claimOf env

/-- **The shipped break**, under Fiat–Shamir: the final verifier accepts the
statement `x = tr.inst`, the F′ claim for `x` is false, and every program
witness among the shrink candidates whose decoding the replay accepts from
`start` has a reading whose walk under the layer-1 log accepts. A witness
that the replay rejects meets the condition, so the event keeps it;
`CircuitAgrees` says that no such witness exists (`shippedBreak_sub`). -/
def ShippedBreak (S : Layer1.Setup Gl) (L0 L1 : ℕ)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (r : RestoreOutcome Gl Cubic (sp.game S L0 L1)) : Prop :=
  ∃ tr, r.1 = some (some tr) ∧ ¬ ParentClaim relation ajtai sp.authority tr.inst ∧
    ∀ w ∈ tr.cands, ((sp.prog tr.inst).ccs template sp.counts).Relation w →
      ∀ points t', Replay.Runs (sp.replay tr.inst w) start points t' →
        ∃ p, sp.Reads S L0 L1 tr.inst w p ∧ ∃ b, r.2 p = some b

/-! ## The links -/

/-- **The terminal wiring holds.** The counts do not depend on the statement
(`lower_counts`), so `lower_sound` gives an accepting executor run, and
`wiring_of_exec` reads the terminal assignment from it. -/
theorem terminalWiring : sp.TerminalWiring := by
  intro x w holds
  rw [Shipped.counts, ← Circuit.lower_counts (F := Gl) sp.program (List.ofFn x), Shipped.prog] at holds
  -- A variable for the program, so that no step evaluates it.
  generalize program : sp.program = L at holds
  obtain ⟨env, terminal, claim⟩ := wiring_of_exec sp.package sp.load sp.authority sp.key x _ _ L program
    (Circuit.lower_sound L (List.ofFn x) w holds) (sp.exec x w) (by subst program; rfl)
  -- The decoded statement, one definition at a time.
  have decoded : (sp.decode x w).statement =
      Decode.statement (fun h => Circuit.Package.word ((sp.exec x w).value h)) sp.views.claim := rfl
  have views : sp.views = ((Circuit.finalViews sp.package (contextWords sp.authority) sp.key).run {}).1 := rfl
  rw [decoded, views]
  exact ⟨env, terminal, claim⟩

/-- The terminal wiring gives the encoding premise for the shipped program
and its readings. -/
theorem programEncodes_of_wiring (wiring : sp.TerminalWiring) (S : Layer1.Setup Gl) (L0 L1 : ℕ) :
    ProgramEncodes template sp.counts sp.prog sp.authority (sp.game S L0 L1) (sp.Reads S L0 L1) := by
  intro x w holds
  obtain ⟨env, terminal, statement⟩ := wiring x w holds
  exact ⟨env, terminal, fun _ ⟨_, _, _, _, tr, _, _, _, leaf, same⟩ => ⟨tr, leaf, same.trans statement⟩⟩

/-- Under the agreement, the shipped break is a `FinalRestoreBreak` of the
shipped program with its readings. -/
theorem shippedBreak_sub (agrees : sp.CircuitAgrees) (S : Layer1.Setup Gl) (L0 L1 : ℕ)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (r : RestoreOutcome Gl Cubic (sp.game S L0 L1)) (break_ : sp.ShippedBreak S L0 L1 relation ajtai r) :
    FinalRestoreBreak (ParentClaim relation ajtai sp.authority) template sp.counts sp.prog (sp.game S L0 L1)
      (sp.Reads S L0 L1) r := by
  obtain ⟨tr, accepted, notClaim, reads⟩ := break_
  refine ⟨tr, accepted, fun w member holds => ?_, notClaim⟩
  obtain ⟨points, t', run⟩ := agrees _ w holds
  exact reads w member holds points t' run

/-- **Non-vacuity of the read condition.** A replay run from `start` on what
`w` decodes to has a reading, unless WHIR lists are missing or Poseidon2
collides at explicit inputs on a setup leaf. The proof is the bridge
(`replay_accepts_implies_path`) with `MerkleBinds` from the honest root. -/
theorem reads_exist (S : Layer1.Setup Gl) (L0 L1 : ℕ) (facts : Bridge.RelationFacts sp.key.relation S)
    (root : sp.key.relation.key.root = Bridge.honestRoot sp.key.relation S) (x : StateWords) (w : ℕ → Gl)
    {points : Replay.OpeningPoints} {t' : Replay.Transcript} (run : Replay.Runs (sp.replay x w) start points t')
    (lists : ∃ cands p1, cands.length ≤ L0 ∧ p1.length ≤ L1 ∧
      Bridge.WhirOpens (K := Cubic) (Bridge.shapeOf sp.key.relation) (sp.decode x w).proof points cands p1)
    (noCollision : ¬ ∃ leaf ∈ (sp.decode x w).proof.leaves.toList, ∃ i < 2 ^ (sp.key.relation.setupVars - 2),
      Replay.pathOk sp.key.relation.key.root i leaf = true ∧
        Bridge.PathCollision (Bridge.honestValues sp.key.relation S) (sp.key.relation.setupVars - 2) leaf i) :
    ∃ p, sp.Reads S L0 L1 x w p := by
  obtain ⟨cands, p1, short0, short1, whir⟩ := lists
  have shape : Bridge.ProofShape sp.key.relation (sp.decode x w).proof := by
    have verify := run
    unfold replay Replay.verify at verify
    obtain ⟨_, _, _, rest⟩ := Replay.run_bind verify
    obtain ⟨_, _, read, _⟩ := Replay.run_bind rest
    exact Bridge.readProof_ok (Replay.run_liftExcept read).1
  have merkle : Bridge.MerkleBinds sp.key.relation S (sp.decode x w).proof.leaves := by
    refine (Bridge.merkleBinds_of_root _ S _ root fun leaf member => ?_).resolve_right noCollision
    rw [(shape.leafShapes leaf member).2]
    simp only [Replay.foldVars]
    omega
  obtain ⟨tr, ds, rest, draws, statement, -, p, node, pathDraws⟩ :=
    Bridge.replay_accepts_implies_path sp.key.relation S facts (sp.decode x w).seed (sp.decode x w).statement
      (sp.decode x w).proof merkle L0 L1 cands short0 p1 short1 run whir
  exact ⟨p, points, t', ds, rest, tr, run, by simpa [start] using draws, pathDraws, node, statement⟩

/-! ## The production bound -/

/-- **The production bound of the shipped program.** Under Fiat–Shamir,
over `Gl` and `Cubic`, with the charged terms of each commitment within
Rust's split, a prover with `calls` queries causes `ShippedBreak` with
probability at most `calls / 2^t`.

The named premises: `agrees` (`CircuitAgrees`), `facts` (the relation's
sizes and structure) and `honest` (the honest setup). -/
theorem shipped_restore_budget (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (L0 L1 : ℕ) (agrees : sp.CircuitAgrees)
    (facts : Bridge.RelationFacts sp.key.relation S)
    (shape : ParentShape (Bridge.shapeOf sp.key.relation) Data.logicalWidth)
    (honest : HonestSetup (Bridge.shapeOf sp.key.relation) S relation ajtai)
    (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees (Bridge.shapeOf sp.key.relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ (sp.key.relation.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars template sp.counts)
      (Shrink.Layout.cellVars template sp.counts) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (N : ℕ) (sizes : (Bridge.shapeOf sp.key.relation).Sizes N) (small : N ≤ goldilocksModulus)
    (calls : ℕ) (prover : Layer1.Game.Strategy (RestoreOutcome Gl Cubic (sp.game S L0 L1))) :
    Layer1.Game.prob prover
        (restorePair (Shrink.shrinkGame template sp.counts sp.prog Ls) (sp.game S L0 L1) calls [] [])
        (sp.ShippedBreak S L0 L1 relation ajtai) ≤
      calls / 2 ^ t := by
  have toParent (x : StateWords)
      (claim : TerminalClaim (Bridge.shapeOf sp.key.relation) Bridge.phi Bridge.barOf S sp.authority x) :
      ParentClaim relation ajtai sp.authority x := by
    rw [phi_eq_modulus] at claim
    exact parentClaim_of_terminalClaim _ _ S relation ajtai shape sizes.blocks_le barOf_eq honest sp.authority x
      claim
  refine (prob_mono prover _ fun r break_ => finalRestoreBreak_mono toParent _ _ _ _ _ _
      (sp.shippedBreak_sub agrees S L0 L1 relation ajtai r break_)).trans ?_
  exact final_restore_budget (Bridge.shapeOf sp.key.relation) Bridge.phi Bridge.barOf S L0 L1 h0 h1 cubic_big
    (show sp.key.relation.setupVars ≤ 35 by have := facts.high; omega) t template sp.counts sp.prog Ls p0 p1 rust
    (Bridge.queryAt sp.key.relation) N sizes (natCast_injective N small)
    (Bridge.queryAt_distinct sp.key.relation facts.high) facts.fits Bridge.phi_monic Bridge.phi_natDegree
    (by rw [shape.D]; decide) Circuit.Block.poseidonTemplate_wellFormed
    (fun x => Circuit.lower_valid template le_rfl sp.program (List.ofFn x))
    (natCast_injective 9 (by norm_num [goldilocksModulus])) sp.authority (sp.Reads S L0 L1)
    (sp.programEncodes_of_wiring sp.terminalWiring S L0 L1) calls prover

end Shipped

end CompressionLean.Final
