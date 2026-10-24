import CompressionLean.Sim.Claim
import CompressionLean.Sim.Fold
import CompressionLean.Sim.Verify
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Lowering.Sound
import CompressionLean.Lowering.Record
import CompressionLean.Lowering.Valid
import CompressionLean.Bridge.Main
import CompressionLean.Final.Bridge
import CompressionLean.Checks.KeyFacts
import CompressionLean.Spec.Bound

/-!
Owns `ProgramEncodes` (`Claim.lean`) for the production final program
(`FinalProgram::run`, `lifecycle/terminal/mod.rs:54-80`, with `verify_with`,
`neo-spartan/src/lib.rs:692-703`), for any layer-1 key. The program, what
layer 1 reads from its witness (`wordsOf`, `statementWith`, `seedWith`,
`proofWith`) and the reading (`Reads`) are in `Spec/`.

- `ctxOf`: the executor's run on a witness, whose words are `wordsOf`.
- `programEncodes`: `ProgramEncodes` from the `verify` composition
  (`verify_ensures`) and premises on the key. The game uses F′'s modulus,
  which is the bridge's `phi` (`phi_eq_modulus`).

Adversary: the prover picks the whole witness. Nothing here assumes that a
word is in range or that a hint is honest; the circuit's checks and the
stage lemmas carry every such fact.
-/

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Final
open CompressionLean.Checks
open CompressionLean.Lowering
open CompressionLean.Circuit CompressionLean.Terminal.Package CompressionLean.Terminal.Program
open NightstreamFPrime.Spec (goldilocksModulus)
open CompressionLean.Bridge CompressionLean.Layer1
open NightstreamFPrime
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

section Runs

variable (key : Layer1Key) (loaded : Loaded) (context : Array Replay.Word)

/-- Each shrink run of the program names only its own cells (`lower_valid`). -/
theorem shrinkRun_valid (x : StateWords) : (shrinkRun key loaded context x).Valid Block.poseidonTemplate :=
  lower_valid Block.poseidonTemplate (Nat.le_of_eq rfl) _ _

/-- The executor's run of the final program on `x` and the witness `w`. -/
noncomputable def ctxOf (x : StateWords) (w : ℕ → Spec.F) : Ctx Spec.F :=
  let L := (productionProgram key loaded context).program
  let stmt := List.ofFn x
  let tape := tapeOf Block.poseidonTemplate L stmt w
  { perm := Spec.Poseidon2.permute, stmt, tape, val := (Exec.run Spec.Poseidon2.permute stmt tape L).value }

end Runs

section Gives

variable {key : Layer1Key} {loaded : Loaded} {context : Array Replay.Word}
  {authority : Lifecycle.VerifierContext.Authority} {c : Ctx Spec.F} {w : Var → Replay.Word}

/-- What an accepting run gives: a terminal assignment for the run's
statement whose claim is the decoded statement, and an accepting replay of
the decoded seed, statement and proof. -/
def Gives (key : Layer1Key) (loaded : Loaded) (context : Array Replay.Word)
    (authority : Lifecycle.VerifierContext.Authority) (c : Ctx Spec.F) (w : Var → Replay.Word) : Prop :=
  ∃ env, TerminalFor authority (Sim.statementOf c) env ∧
    Spec.statementOf (statementWith loaded context w) = claimOf env ∧
    ∃ points t', Spec.Runs (Replay.verify key.relation (seedWith loaded context w) (statementWith loaded context w)
      (proofWith key loaded context w)) start points t'

instance verifyWith_appends (key : Layer1Key) (state : Fin 16 → Var) (claim : ClaimView) :
    Appends (Circuit.verifyWith key state claim) := by
  unfold Circuit.verifyWith; appends

/-- `verify_with` (`neo-spartan/src/lib.rs:692-703`) from the handles that
the terminal part hands it. -/
private theorem verifyWith_at (hp : Permutes c.perm) (hw : Decodes c w) (facts : KeyFacts key)
    (named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < Replay.D)
    (h : Handoff c authority (prefixRun loaded context).1 (prefixRun loaded context).2.values) :
    EnsuresAt c (prefixRun loaded context).2
      (verifyWith key (prefixRun loaded context).1.2 (prefixRun loaded context).1.1.view)
      (fun _ _ => Gives key loaded context authority c w) := by
  obtain ⟨env, terminal, holds, hstate, shaped⟩ := h
  unfold verifyWith
  refine EnsuresAt.bind (FoldTranscript.handoff_ensures
    (gs := Array.ofFn fun lane => w ((prefixRun loaded context).1.2 lane)) hp ?_ (by simp) _ rfl) fun le₁ seed => ?_
  · have hl := List.forall₂_iff_get.mp hstate
    refine IsWords.ofFn (by simp) fun lane => ⟨?_, ?_⟩
    · have := (hl.2 lane (by simp) (by rw [← hl.1]; simp)).1
      simpa only [List.get_eq_getElem, List.getElem_ofFn] using this
    · rw [getElem!_pos _ _ (by simp)]
      simpa using hw ((prefixRun loaded context).1.2 lane)
  refine EnsuresAt.bind (ProofView.read_ensures key.relation _ _ facts.circuit.run facts.circuit.row facts.circuit.slot
    hw _ rfl) fun le₂ view => ?_
  obtain ⟨proof, read⟩ := view
  refine (verify_ensures facts.circuit facts.relation named (w := w) (seedWords := seedWith loaded context w)
    (st := statementWith loaded context w) (g := proofWith key loaded context w) hp
    ⟨?_, fun lane => (seed lane).mono⟩ (isClaim_decode hw (claimHolds_words_lt holds)).mono
    (fits_decode w holds shaped facts.constants.kappa facts.constants.blocks facts.constants.point
      facts.constants.matrices) proof read _ rfl).post fun runs => ⟨env, terminal, statementOf_decode hw holds shaped, runs⟩
  rw [seedWith, FoldTranscript.handoffWords, Array.size_extract, FoldTranscript.absorbWords_size _ (by simp)]
  rfl

/-- A witness of the program's CCS gives what `Gives` states. -/
private theorem program_gives
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Replay.Word.cast (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (facts : KeyFacts key)
    (named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < Replay.D)
    (x : StateWords) (wit : ℕ → Spec.F)
    (holds : ((shrinkRun key loaded context x).ccs Block.poseidonTemplate
      (shrinkRun key loaded context x).counts).Relation wit) :
    Gives key loaded context authority (ctxOf key loaded context x wit) (wordsOf key loaded context x wit) := by
  have hw : Decodes (ctxOf key loaded context x wit) (wordsOf key loaded context x wit) :=
    fun _ => (Replay.Word.cast_ofF _).symm
  have accepts : Exec (ctxOf key loaded context x wit).perm (productionProgram key loaded context).program
      (ctxOf key loaded context x wit).stmt (ctxOf key loaded context x wit).tape :=
    lower_sound (productionProgram key loaded context).program (List.ofFn x) wit holds
  exact finalProgram_exec (layer1 := fun claim state => verifyWith key state claim.view)
    (R := fun _ _ => Gives key loaded context authority (ctxOf key loaded context x wit) (wordsOf key loaded context x wit))
    (by rw [productionProgram]) loads valid hctx (verifyWith_at permutes_poseidon2 hw facts named)
    (fun _ => inferInstance) rfl accepts

end Gives

/-- The bridge's `Φ81` is F′'s modulus. -/
theorem phi_eq_modulus : phi = modulus := by
  unfold phi modulus
  rw [add_assoc]
  rfl

/-- The bridge's bar transform is F′'s (`nativeBarEntry`). -/
theorem barOf_eq (p l : Fin Spec.ringDegree) :
    barOf p l = Spec.gl (Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel.nativeBarEntry p l) := by
  unfold barOf Replay.bar
  rw [dif_pos ⟨p.isLt, l.isLt⟩, Replay.Word.toF_ofNat]
  exact ZMod.natCast_zmod_val (show Gl from _)

private theorem statementOf_ctxOf (key : Layer1Key) (loaded : Loaded) (context : Array Replay.Word) (x : StateWords)
    (wit : ℕ → Spec.F) : Sim.statementOf (ctxOf key loaded context x wit) = x := by
  funext k
  fin_cases k <;> rfl

/-- **`ProgramEncodes` for the production final program**
(`lifecycle/terminal/mod.rs:54-80`), for a key that meets the premises. -/
theorem programEncodes {key : Layer1Key} {loaded : Loaded} {context : Array Replay.Word}
    {authority : Lifecycle.VerifierContext.Authority} (S : Setup Gl) (facts : KeyFacts key)
    (fits : S.Fits (shapeOf key.relation))
    (describes : (structureOf key.relation.key.layout).Describes (shapeOf key.relation) S) (L0 L1 : ℕ)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Replay.Word.cast (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0) :
    ProgramEncodes (F := Gl) Block.poseidonTemplate (counts key loaded context) (shrinkRun key loaded context) authority
      (game key S L0 L1) (Reads key loaded context S L0 L1) := by
  intro x wit holds
  rw [counts, lower_counts _ [] (List.ofFn x)] at holds
  have named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < Replay.D := fun s member => by
    obtain ⟨i, inside, rfl⟩ := Array.mem_iff_getElem.mp member
    have lane := Bridge.segment_lane key.relation S fits facts.relation.checked describes inside
    have named := facts.relation.checked.class_named i inside
    simp only [structureOf] at named
    rw [getElem!_pos key.relation.key.layout.segments i inside] at lane named
    exact ⟨named, lane⟩
  obtain ⟨env, terminal, claim, points, t', run⟩ := program_gives loads valid hctx facts named x wit holds
  rw [statementOf_ctxOf] at terminal
  refine ⟨env, terminal, fun p reads => ?_⟩
  obtain ⟨cands, short0, p1, short1, msgs, draws, whir, merkle⟩ := reads
  obtain ⟨tr, ds, rest, -, ⟨t₄, run₄, draws₄⟩, statement, -, reach⟩ :=
    replay_accepts_implies_path key.relation S facts.relation fits describes _ _ _ merkle L0 L1 cands short0 p1
      short1 run (whir _ _ run)
  rw [phi_eq_modulus] at reach
  obtain ⟨p₀, node, pathDraws, pathMsgs⟩ := reach
  have same : p = p₀ := by
    refine Game.Pos.eq_of_msgs_draws (msgs.trans pathMsgs.symm) (draws.trans ?_)
    rw [show Spec.eval _ start = _ from run₄, draws₄]
    exact pathDraws.symm
  subst same
  exact ⟨tr, node, statement.trans claim⟩

end CompressionLean.Sim
