import CompressionLean.Sim.OptionBPrefix
import CompressionLean.Sim.Verify
import CompressionLean.Sim.Claim
import CompressionLean.Sim.Encodes
import CompressionLean.Checks.KeyFacts

/-! Each slot's layer-1 call in option B's final program (CW-6.5): from the prefix's handoff, the
replay of the slot's layer-1 verifier runs on the slot's statement, seed and proof
(`slotLayer1_at`), as `Sim.Encodes.verifyWith_at` gives today's one call. -/

namespace CompressionLean.Sim.OptionB

open CompressionLean.Circuit CompressionLean.OptionB CompressionLean.Lowering
open CompressionLean.Replay (Word)
open CompressionLean.Spec.OptionB
open CompressionLean.Checks
open NightstreamFPrime.Circuit (Expr Env)
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)

variable {c : Ctx F}

instance seedState_appends (column : ℕ → Var) (bits slot : ℕ) : Appends (seedState column bits slot) := by
  unfold seedState; appends

instance seed_appends (column : ℕ → Var) (bits slot : ℕ) : Appends (seed column bits slot) := by
  unfold seed; appends

instance slotLayer1_appends (finish : Package) (key : Layer1Key) (first slot : ℕ) :
    Appends (slotLayer1 finish key first slot) := by
  unfold slotLayer1; appends

/-- Every builder ensures `True`. -/
theorem trivial_ensures {α : Type} {n : ℕ} (b : Build α) : Ensures c n b fun _ _ => True :=
  fun _ _ _ _ _ _ _ _ => True.intro

theorem seed_ensures {column : ℕ → Var} {bits slot n : ℕ} :
    Ensures c n (seed column bits slot) fun state m => ∀ lane, state lane < m := by
  unfold seed
  refine (trivial_ensures _).bind fun state m _ _ => ?_
  exact Ensures.map (Ensures.emit (Q := fun h m' => m' = h + 16) _ fun _ _ _ _ => rfl)
    fun h m' hm lane => by rw [hm]; exact Nat.add_lt_add_left lane.isLt h

/-- A slot's claim handles are finish columns below the S-box columns. -/
theorem claimAt_words_lt {column : ℕ → Var} {start bound m : ℕ} (room : start + 2055 ≤ bound)
    (columns : ∀ k < bound, column k < m) : ∀ h ∈ (claimAt column start).words, h < m := by
  intro h member
  simp only [ClaimView.words, claimAt, Array.mem_append, Array.mem_flatten, Array.mem_map, Array.mem_range,
    Array.mem_flatMap, exists_exists_and_eq_and] at member
  rcases member with (⟨a, ha, b, hb, rfl⟩ | ⟨a, ha, b, hb, rfl⟩) |
      ⟨_, ((⟨a, ha, rfl⟩ | ⟨a, ha, rfl⟩) | ⟨a, ha, b, hb, rfl⟩), hmem⟩
  · exact columns _ (by omega)
  · exact columns _ (by omega)
  all_goals
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hmem
    rcases hmem with rfl | rfl <;> exact columns _ (by omega)

/-- `EnsuresAt.bind` through a named run of the first builder. -/
theorem EnsuresAt.bind_run {α β : Type} {p : Builder} {b : Build α} {f : α → Build β} {Q : α → ℕ → Prop}
    {R : β → ℕ → Prop} {r : α × Builder} (hrun : b.run p = r) (hb : EnsuresAt c p b Q)
    (hf : p.values ≤ r.2.values → Q r.1 r.2.values → EnsuresAt c r.2 (f r.1) R)
    (hb' : Appends b := by appends) (hf' : ∀ a, Appends (f a) := by appends) : EnsuresAt c p (b >>= f) R := by
  subst hrun
  exact EnsuresAt.bind hb hf hb' hf'

theorem seedRun_eq (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) (slot : Fin 2) :
    (seed (fun k => (prefixRun finish stateKeys).1.2 + k) (bitsStart finish.application.1) slot).run
      (slotBuilder finish stateKeys keys slot) = seedRun finish stateKeys keys slot := by
  unfold seedRun; rfl

theorem handoffRun_eq (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) (slot : Fin 2) :
    (FoldTranscript.fromState (seedRun finish stateKeys keys slot).1).handoff.run
      (seedRun finish stateKeys keys slot).2 = handoffRun finish stateKeys keys slot := by
  unfold handoffRun; rfl

theorem viewRun_eq (finish : Package) (stateKeys : List Word) (keys : Fin 2 → Layer1Key) (slot : Fin 2) :
    (ProofView.read (keys slot).relation (keys slot).p1Key.rows (keys slot).p0Key.rows).run
      (handoffRun finish stateKeys keys slot).2 = viewRun finish stateKeys keys slot := by
  unfold viewRun; rfl

/-- A slot's decoded claim has the production shape. -/
theorem claimAt_fits {key : Layer1Key} (constants : Spec.ProductionConstants key.relation) (column : ℕ → Var)
    (start : ℕ) (w : Var → Word) : ((claimAt column start).decode w).fits key.relation = true := by
  have kappa : key.relation.kappa = 22 := constants.kappa.trans Production.kappa_eq
  have blocks : key.relation.publicBlocks = 5 := constants.blocks.trans Production.publicBlocks_eq
  have point : key.relation.pointVariables = 28 := constants.point.trans Production.pointVariables_eq
  have matrices : key.relation.matrices = 4 := constants.matrices.trans Production.matrices_eq
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq, Array.all_eq_true]
  simp [ClaimView.decode, claimAt, kappa, blocks, point, matrices, Replay.D]

/-- A next slot's words lie below the fresh bits, so below the S-box columns. -/
theorem slot_room {finish : Package} (fit : fits finish = true) {slot : ℕ} (inside : slot < 2) :
    slotStart finish.application.1 (5 + slot) + 2055 ≤ finish.sboxStart := by
  have words := (fits_spec fit).1
  unfold bitsStart slotStart witnessWord currentWord z0Word keysWord publicWords slotWords at words
  unfold slotStart witnessWord currentWord z0Word keysWord publicWords slotWords
  omega

/-- Slot `slot`'s layer-1 call, when every finish column's handle comes before it: the replay of
its verifier runs on its statement, seed and proof. -/
theorem slotLayer1_at {finish : Package} {stateKeys : List Word} {keys : Fin 2 → Layer1Key} {slot : Fin 2}
    {w : Var → Word} (hp : Permutes c.perm) (hw : Decodes c w) (facts : Spec.KeyFacts (keys slot))
    (named : ∀ s ∈ (keys slot).relation.key.layout.segments,
      s.classIndex < (keys slot).relation.key.layout.classes.size ∧ s.lane < Replay.D)
    (fit : fits finish = true)
    (columns : ∀ k < finish.sboxStart,
      (prefixRun finish stateKeys).1.2 + k < (slotBuilder finish stateKeys keys slot).values) :
    EnsuresAt c (slotBuilder finish stateKeys keys slot)
      (slotLayer1 finish (keys slot) (prefixRun finish stateKeys).1.2 slot)
      fun _ _ => ∃ points t', Spec.Runs (Replay.verify (keys slot).relation (seedWith finish stateKeys keys slot w)
        (statementWith finish stateKeys slot w) (proofWith finish stateKeys keys slot w)) Spec.start points t' := by
  unfold slotLayer1
  refine EnsuresAt.bind_run (seedRun_eq finish stateKeys keys slot) (seed_ensures _ rfl) fun le₁ hstate => ?_
  unfold verifyWith
  refine EnsuresAt.bind_run (handoffRun_eq finish stateKeys keys slot) (FoldTranscript.handoff_ensures
    (gs := Array.ofFn fun lane => w ((seedRun finish stateKeys keys slot).1 lane)) hp ?_ Array.size_ofFn _ rfl)
      fun le₂ hseed => ?_
  · exact IsWords.ofFn Array.size_ofFn fun lane => ⟨hstate lane, by
      rw [getElem!_pos _ _ (by simp)]
      simpa using hw ((seedRun finish stateKeys keys slot).1 lane)⟩
  refine EnsuresAt.bind_run (viewRun_eq finish stateKeys keys slot) (ProofView.read_ensures (keys slot).relation
    _ _ facts.circuit.run facts.circuit.row facts.circuit.slot hw _ rfl) fun le₃ view => ?_
  obtain ⟨proof, read⟩ := view
  refine (verify_ensures facts.circuit facts.relation named (w := w)
    (seedWords := seedWith finish stateKeys keys slot w) (st := statementWith finish stateKeys slot w)
    (g := proofWith finish stateKeys keys slot w) hp ⟨?_, fun lane => (hseed lane).mono⟩
    (isClaim_decode hw (claimAt_words_lt (slot_room fit slot.isLt) fun k inside =>
      lt_of_lt_of_le (columns k inside) ((le₁.trans le₂).trans le₃))).mono
    (claimAt_fits facts.constants _ _ w) proof read _ rfl).post fun runs => runs
  rw [seedWith, FoldTranscript.handoffWords, Array.size_extract, FoldTranscript.absorbWords_size _ (by simp)]
  rfl

end CompressionLean.Sim.OptionB
