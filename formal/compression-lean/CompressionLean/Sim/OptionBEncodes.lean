import CompressionLean.Sim.OptionBSlots
import CompressionLean.Final.OptionB

/-! Option B's final program encodes its claim (CW-6.5), the premise `Final.OptionB.Encodes` of its
bound. A witness of the shrink relation gives an
accepted run (`Lowering.lower_sound`): its prefix gives the finish columns' environment, in which
the program's checks hold (`finishPrefix_ensures`), and each slot's call runs the replay of its
layer-1 verifier (`slotLayer1_at`) on that slot's claim words (`statementWith_eq`). The replay
reaches the leaf of the layer-1 game that the position reads (`Bridge.replay_accepts_implies_path`). -/

namespace CompressionLean.Sim.OptionB

open CompressionLean.Circuit CompressionLean.OptionB CompressionLean.Lowering
open CompressionLean.Replay (Word)
open CompressionLean.Spec.OptionB
open CompressionLean.Layer1
open NightstreamFPrime.Circuit (Expr Env)
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)

section Runs

variable (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key)

noncomputable def ctxOf (x : Spec.StateWords) (w : ℕ → F) : Ctx F :=
  let L := (program finish stateKeys keys).program
  let stmt := List.ofFn x
  let tape := tapeOf Circuit.Block.poseidonTemplate L stmt w
  { perm := NightstreamFPrime.Spec.Poseidon2.permute, stmt, tape,
    val := (Exec.run NightstreamFPrime.Spec.Poseidon2.permute stmt tape L).value }

theorem shrinkRun_valid (x : Spec.StateWords) :
    (shrinkRun finish stateKeys keys x).Valid Circuit.Block.poseidonTemplate :=
  lower_valid Circuit.Block.poseidonTemplate (Nat.le_of_eq rfl) _ _

theorem slotBuilder_zero : slotBuilder finish stateKeys keys 0 = (prefixRun finish stateKeys).2 := by
  unfold slotBuilder; exact ite_eq_left rfl

theorem slotBuilder_one : slotBuilder finish stateKeys keys 1 =
    ((slotLayer1 finish (keys 0) (prefixRun finish stateKeys).1.2 0).run (prefixRun finish stateKeys).2).2 := by
  unfold slotBuilder; exact ite_eq_right (by decide)

theorem prefixRun_eq : (finishPrefix finish stateKeys).run {} = prefixRun finish stateKeys := by
  unfold prefixRun; rfl

end Runs

/-- A slot's claim words, decoded through the column handles, are its claim words in the columns.
The claim reads the `slotWords - 1` words from `start`. -/
theorem claimAt_decode {column : ℕ → Var} {start : ℕ} {w : Var → Word} {g : ℕ → Word}
    (agree : ∀ k, start ≤ k → k < start + 2054 → w (column k) = g k) :
    (claimAt column start).decode w = (claimAt id start).decode g := by
  simp only [ClaimView.decode, claimAt, Array.map_map, Function.comp_def, id]
  congr 1
  · exact Array.map_congr_left fun row inside => Array.map_congr_left fun lane lane_inside => by
      simp only [Array.mem_range] at inside lane_inside
      exact agree _ (by omega) (by omega)
  · exact Array.map_congr_left fun block inside => Array.map_congr_left fun lane lane_inside => by
      simp only [Array.mem_range] at inside lane_inside
      exact agree _ (by omega) (by omega)
  · exact Array.map_congr_left fun k inside => by
      simp only [Array.mem_range] at inside
      simp only [Prod.map, agree (start + 1728 + 2 * k) (by omega) (by omega),
        agree (start + 1728 + 2 * k + 1) (by omega) (by omega)]
  · exact Array.map_congr_left fun k inside => by
      simp only [Array.mem_range] at inside
      simp only [Prod.map, agree (start + 1188 + 2 * k) (by omega) (by omega),
        agree (start + 1188 + 2 * k + 1) (by omega) (by omega)]
  · exact Array.map_congr_left fun family inside => Array.map_congr_left fun k k_inside => by
      simp only [Array.mem_range] at inside k_inside
      simp only [Prod.map, agree (start + 1188 + (family + 1) * 108 + 2 * k) (by omega) (by omega),
        agree (start + 1188 + (family + 1) * 108 + 2 * k + 1) (by omega) (by omega)]

/-- What an accepted run gives: an environment of the finish columns in which the program's checks
hold, and for each slot its claim words and the replay of its layer-1 verifier. -/
def Gives (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) (c : Ctx F) (w : Var → Word) :
    Prop :=
  ∃ env, FinishTerminal finish stateKeys (statementOf c) env ∧
    ∀ slot : Fin 2, Spec.statementOf (statementWith finish stateKeys slot w) = slotClaim finish slot env ∧
      ∃ points t', Spec.Runs (Replay.verify (keys slot).relation (seedWith finish stateKeys keys slot w)
        (statementWith finish stateKeys slot w) (proofWith finish stateKeys keys slot w)) Spec.start points t'

/-- A program that starts with the statement handles reads the statement. -/
theorem readsStatement_of {c : Ctx F} {k : Array Var → Build Unit} (hk : ∀ s, Appends (k s))
    (hval : c.val = (Exec.run c.perm c.stmt c.tape (Spec.statementHandles >>= k).program).value) :
    ReadsStatement c := by
  obtain ⟨new, hi, -⟩ := (hk (List.range' 0 9).toArray).out
    { instrs := #[] ++ Array.replicate 9 .pub, values := 0 + 9 }
  have pre : (Spec.statementHandles >>= k).program.toList = List.replicate 9 .pub ++ new := by
    rw [Build.program, StateT.run_bind, statementHandles_run]
    show ((k (List.range' 0 9).toArray).run { instrs := #[] ++ Array.replicate 9 .pub, values := 0 + 9 }).2.instrs.toList = _
    rw [hi]
    simp
  exact fun j => by
    rw [hval]
    exact c.run_pubs pre j.isLt

/-- A slot's statement is its claim words in the environment `env` that the column handles hold. -/
theorem statementWith_eq {finish : Package} {stateKeys : List Word} {c : Ctx F} {w : Var → Word} {env : Env}
    {m : ℕ} (fit : fits finish = true) (words : ∀ h, w h = Word.ofF (c.val h))
    (columns : ∀ k < finish.sboxStart, ColumnIs c (prefixRun finish stateKeys).1.2 env m k) (slot : Fin 2) :
    Spec.statementOf (statementWith finish stateKeys slot w) = slotClaim finish slot env := by
  unfold statementWith slotClaim
  rw [claimAt_decode fun k _ inside => ?_]
  have room := slot_room fit slot.isLt
  rw [words, (columns k (by omega)).2]

/-- Stated for `program`, so that a caller can keep it irreducible. -/
theorem finalProgram_gives {finish : Package} {stateKeys : List Word} {keys : Fin 2 → Layer1Key}
    {c : Ctx F} {w : Var → Word} {program : Build Unit} (eq : program = finalProgram finish stateKeys keys)
    (fit : fits finish = true) (keysLength : stateKeys.length ≤ 8) (facts : ∀ slot, Spec.KeyFacts (keys slot))
    (named : ∀ slot, ∀ s ∈ (keys slot).relation.key.layout.segments,
      s.classIndex < (keys slot).relation.key.layout.classes.size ∧ s.lane < Replay.D)
    (hperm : c.perm = NightstreamFPrime.Spec.Poseidon2.permute) (words : ∀ h, w h = Word.ofF (c.val h))
    (hval : c.val = (Exec.run c.perm c.stmt c.tape program.program).value)
    (accepts : Exec c.perm program.program c.stmt c.tape) :
    Gives finish stateKeys keys c w := by
  have hw : Decodes c w := fun h => by rw [words]; exact (cast_ofF _).symm
  have hp : Permutes c.perm := hperm ▸ permutes_poseidon2
  obtain ⟨k, eqk, hk⟩ : ∃ k : Array Var → Build Unit,
      finalProgram finish stateKeys keys = Spec.statementHandles >>= k ∧ ∀ s, Appends (k s) := by
    refine ⟨_, by unfold finalProgram finishPrefix; rw [bind_assoc]; rfl, fun s => ?_⟩
    appends
  have reads : ReadsStatement c := readsStatement_of hk (eqk ▸ eq ▸ hval)
  have step : EnsuresAt c {} (finalProgram finish stateKeys keys) fun _ _ => Gives finish stateKeys keys c w := by
    unfold finalProgram
    refine EnsuresAt.bind_run (prefixRun_eq finish stateKeys)
      (finishPrefix_ensures hperm fit keysLength reads {} rfl) fun _ handoff => ?_
    obtain ⟨terminal, columns⟩ := handoff
    have slot0 := slotLayer1_at (keys := keys) (slot := 0) hp hw (facts 0) (named 0) fit fun k inside => by
      rw [slotBuilder_zero]; exact (columns k inside).1
    rw [slotBuilder_zero] at slot0
    simp only [Fin.val_zero] at slot0
    refine EnsuresAt.bind slot0 fun le runs0 => ?_
    have grow : (prefixRun finish stateKeys).2.values ≤ (slotBuilder finish stateKeys keys 1).values := by
      rw [slotBuilder_one]
      exact (slotLayer1_appends _ _ _ _).le _
    have bound1 : ∀ k < finish.sboxStart,
        (prefixRun finish stateKeys).1.2 k < (slotBuilder finish stateKeys keys 1).values := fun k inside =>
      Nat.lt_of_lt_of_le (columns k inside).1 grow
    have slot1 := slotLayer1_at (keys := keys) (slot := 1) hp hw (facts 1) (named 1) fit bound1
    rw [slotBuilder_one] at slot1
    simp only [Fin.val_one] at slot1
    refine slot1.post fun runs1 => ⟨_, terminal, fun slot => ⟨statementWith_eq fit words columns slot, ?_⟩⟩
    fin_cases slot
    · exact runs0
    · exact runs1
  exact EnsuresAt.exec (eq ▸ step) hval accepts

theorem program_gives {finish : Package} {stateKeys : List Word} {keys : Fin 2 → Layer1Key}
    (fit : fits finish = true) (keysLength : stateKeys.length ≤ 8) (facts : ∀ slot, Spec.KeyFacts (keys slot))
    (named : ∀ slot, ∀ s ∈ (keys slot).relation.key.layout.segments,
      s.classIndex < (keys slot).relation.key.layout.classes.size ∧ s.lane < Replay.D)
    (x : Spec.StateWords) (wit : ℕ → F)
    (holds : ((shrinkRun finish stateKeys keys x).ccs Circuit.Block.poseidonTemplate
      (shrinkRun finish stateKeys keys x).counts).Relation wit) :
    Gives finish stateKeys keys (ctxOf finish stateKeys keys x wit) (wordsOf finish stateKeys keys x wit) :=
  finalProgram_gives (by rw [program]) fit keysLength facts named rfl (fun _ => rfl) rfl
    (lower_sound (program finish stateKeys keys).program (List.ofFn x) wit holds)

theorem statementOf_ctxOf (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key)
    (x : Spec.StateWords) (wit : ℕ → F) : statementOf (ctxOf finish stateKeys keys x wit) = x := by
  funext k
  fin_cases k <;> rfl

/-- The segment facts that `verify_ensures` reads, from the key's facts and its setup. -/
theorem named_of {key : Layer1Key} {S : Setup Gl} (facts : Spec.KeyFacts key) (fits : S.Fits (Spec.shapeOf key.relation))
    (describes : (Spec.structureOf key.relation.key.layout).Describes (Spec.shapeOf key.relation) S) :
    ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < Replay.D := fun s member => by
  obtain ⟨i, inside, rfl⟩ := Array.mem_iff_getElem.mp member
  have lane := Bridge.segment_lane key.relation S fits facts.relation.checked describes inside
  have named := facts.relation.checked.class_named i inside
  simp only [Spec.structureOf] at named
  rw [getElem!_pos key.relation.key.layout.segments i inside] at lane named
  exact ⟨named, lane⟩

/-- A position that reads a replay which runs reaches the leaf of that replay's statement. -/
theorem slot_leaf {key : Layer1Key} {S : Setup Gl} (facts : Spec.KeyFacts key)
    (fits : S.Fits (Spec.shapeOf key.relation))
    (describes : (Spec.structureOf key.relation.key.layout).Describes (Spec.shapeOf key.relation) S) (L0 L1 : ℕ)
    {seed : Array Word} {st : Replay.Statement} {proof : Replay.Proof} {points : Replay.OpeningPoints}
    {t' : Replay.Transcript} (run : Spec.Runs (Replay.verify key.relation seed st proof) Spec.start points t')
    {p : (Spec.game key S L0 L1).Pos} (reads : ReadsWith key S L0 L1 st seed proof p) :
    ∃ tr, Game.Pos.node (Spec.game key S L0 L1) p = .pure (some tr) ∧ tr.statement = Spec.statementOf st := by
  obtain ⟨cands, short0, p1, short1, msgs, draws, whir, merkle⟩ := reads
  obtain ⟨tr, ds, rest, -, ⟨t₄, run₄, draws₄⟩, statement, -, reach⟩ :=
    Bridge.replay_accepts_implies_path key.relation S facts.relation fits describes _ _ _ merkle L0 L1 cands
      short0 p1 short1 run (whir _ _ run)
  rw [phi_eq_modulus] at reach
  obtain ⟨p₀, node, pathDraws, pathMsgs⟩ := reach
  have same : p = p₀ := by
    refine Game.Pos.eq_of_msgs_draws (msgs.trans pathMsgs.symm) (draws.trans ?_)
    rw [show Spec.eval _ Spec.start = _ from run₄, draws₄]
    exact pathDraws.symm
  subst same
  exact ⟨tr, node, statement⟩

/-- Option B's final program encodes its claim: every accepted witness gives finish columns in
which the program's checks hold, whose slot claims are the leaves of the positions it reads. -/
theorem optionBEncodes {finish : Package} {stateKeys : List Word} {keys : Fin 2 → Layer1Key}
    (S : Fin 2 → Setup Gl) (facts : ∀ slot, Spec.KeyFacts (keys slot))
    (fits : ∀ slot, (S slot).Fits (Spec.shapeOf (keys slot).relation))
    (describes : ∀ slot,
      (Spec.structureOf (keys slot).relation.key.layout).Describes (Spec.shapeOf (keys slot).relation) (S slot))
    (L0 L1 : ℕ) (fit : Spec.OptionB.fits finish = true) (keysLength : stateKeys.length ≤ 8) :
    Final.OptionB.Encodes (F := Gl) Circuit.Block.poseidonTemplate (counts finish stateKeys keys)
      (shrinkRun finish stateKeys keys) (FinishTerminal finish stateKeys) (slotClaim finish 0) (slotClaim finish 1)
      (Spec.game (keys 0) (S 0) L0 L1) (Spec.game (keys 1) (S 1) L0 L1)
      (Reads finish stateKeys keys 0 (S 0) L0 L1) (Reads finish stateKeys keys 1 (S 1) L0 L1) := by
  intro x wit holds
  rw [counts, lower_counts _ [] (List.ofFn x)] at holds
  obtain ⟨env, terminal, slots⟩ := program_gives fit keysLength facts
    (fun slot => named_of (facts slot) (fits slot) (describes slot)) x wit holds
  have terminal : FinishTerminal finish stateKeys x env :=
    Eq.mp (congrArg (fun s => FinishTerminal finish stateKeys s env) (statementOf_ctxOf finish stateKeys keys x wit))
      terminal
  refine ⟨env, terminal, fun p reads => ?_, fun p reads => ?_⟩
  · obtain ⟨claim, points, t', run⟩ := slots 0
    obtain ⟨tr, node, statement⟩ := slot_leaf (facts 0) (fits 0) (describes 0) L0 L1 run reads
    exact ⟨tr, node, statement.trans claim⟩
  · obtain ⟨claim, points, t', run⟩ := slots 1
    obtain ⟨tr, node, statement⟩ := slot_leaf (facts 1) (fits 1) (describes 1) L0 L1 run reads
    exact ⟨tr, node, statement.trans claim⟩

end CompressionLean.Sim.OptionB
