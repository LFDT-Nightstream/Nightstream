import CompressionLean.Sim.OptionBEncodes

/-! The claim of option B's final program stays complete when only the live columns have a cell
(`final_program.rs::Finish::new`). The final program gives the value 0 to each column below the
S-box columns that it does not read. Finish columns in which its checks hold keep them, and keep
both slot claims, when these columns become 0 (`finishTerminal_live`, `slotClaim_live`): each check
reads only columns of `Spec.OptionB.readColumns`, and a read column below the S-box columns is live
(`live_iff`). So a statement whose claim holds has finish columns of the program's form in which
the claim holds (`holds_live`). -/

namespace CompressionLean.Sim.OptionB

open CompressionLean.Circuit CompressionLean.OptionB CompressionLean.Lowering
open CompressionLean.Replay (Word)
open CompressionLean.Spec.OptionB
open NightstreamFPrime.Circuit (Expr Env)
open NightstreamFPrime.Export.OptionB.StepPackage (Package PackagePiece)
open NightstreamFPrime.Gadgets.Poseidon2

/-- `liveEnv` keeps each column that the program reads. -/
theorem liveEnv_read {finish : Package} (env : Env) {k : ℕ} (read : k ∈ readColumns finish) :
    liveEnv finish env k = env k := by
  unfold liveEnv
  split
  · rename_i dead
    rw [(live_iff finish k).mpr ⟨dead.1, read⟩] at dead
    exact absurd dead.2 (by decide)
  · rfl

/-- `liveEnv` is 0 on each column below the S-box columns that is not live. -/
theorem liveEnv_dead {finish : Package} (env : Env) {k : ℕ} (inside : k < finish.sboxStart)
    (dead : (live finish).getD k false = false) : liveEnv finish env k = 0 :=
  ite_eq_left ⟨inside, dead⟩

theorem row_read {finish : Package} {row : Shrink.R1csRow Word} (member : row ∈ finishRows finish)
    {t : ℕ × Word} (term : t ∈ row.a ++ row.b ++ row.c) : t.1 ∈ readColumns finish :=
  List.mem_append_left _ (List.mem_append_left _
    (List.mem_flatMap.mpr ⟨row, member, List.mem_map_of_mem term⟩))

theorem lane_read {finish : Package} {block : NightstreamFPrime.Export.Generic.Frame.Block}
    (member : block ∈ finish.blocks) {lane : Expr} (inside : lane ∈ block.lanes) {form : F × List (ℕ × F)}
    (linear : linearForm lane = some form) {t : ℕ × F} (term : t ∈ form.2) : t.1 ∈ readColumns finish :=
  List.mem_append_left _ (List.mem_append_right _ (List.mem_flatMap.mpr ⟨block, member,
    List.mem_append_left _ (List.mem_flatMap.mpr ⟨lane, inside, by
      rw [linear, Option.getD_some]
      exact List.mem_map_of_mem term⟩)⟩))

theorem output_read {finish : Package} {block : NightstreamFPrime.Export.Generic.Frame.Block}
    (member : block ∈ finish.blocks) (lane : Fin 16) : block.start + 1080 + lane ∈ readColumns finish :=
  List.mem_append_left _ (List.mem_append_right _ (List.mem_flatMap.mpr ⟨block, member,
    List.mem_append_right _ (List.mem_range'_1.mpr ⟨Nat.le_add_right _ _, Nat.add_lt_add_left lane.isLt _⟩)⟩))

/-- The columns of `programReads` by their ranges. -/
theorem mem_programReads {witnessWords k : ℕ} : k ∈ programReads witnessWords ↔
    k = 0 ∨ k = iterationWord ∨ (keysWord ≤ k ∧ k < keysWord + 8) ∨ (z0Word ≤ k ∧ k < z0Word + 4) ∨
      (currentWord ≤ k ∧ k < currentWord + 4) ∨
      (slotStart witnessWords 5 ≤ k ∧ k < slotStart witnessWords 5 + 2054) ∨
      (slotStart witnessWords 6 ≤ k ∧ k < slotStart witnessWords 6 + 2054) ∨
      (bitsStart witnessWords ≤ k ∧ k < bitsStart witnessWords + 256) := by
  simp only [programReads, List.flatMap_cons, List.flatMap_nil, List.mem_append, List.mem_range'_1,
    List.not_mem_nil, or_false, slotWords]
  constructor <;> intro h <;> omega

theorem program_read {finish : Package} {k : ℕ} (read : k ∈ programReads finish.application.1) :
    k ∈ readColumns finish :=
  List.mem_append_right _ read

/-- A row holds in each environment that agrees on its columns. -/
theorem holds_congr {env env' : ℕ → F} {row : Shrink.R1csRow Word}
    (same : ∀ t ∈ row.a ++ row.b ++ row.c, env' t.1 = env t.1) :
    R1csRow.Holds env' row ↔ R1csRow.Holds env row := by
  have side : ∀ terms : List (ℕ × Word), (∀ t ∈ terms, t ∈ row.a ++ row.b ++ row.c) →
      (terms.map fun t => env' t.1 * Word.cast t.2).sum = (terms.map fun t => env t.1 * Word.cast t.2).sum :=
    fun terms sub => congrArg List.sum (List.map_congr_left fun t member => by rw [same t (sub t member)])
  simp only [R1csRow.Holds]
  rw [side row.a fun t member => List.mem_append_left _ (List.mem_append_left _ member),
    side row.b fun t member => List.mem_append_left _ (List.mem_append_right _ member),
    side row.c fun t member => List.mem_append_right _ member]

/-- The final program's checks hold on the live columns of finish columns on which they hold. -/
theorem finishTerminal_live {finish : Package} {stateKeys : List Word} {x : Spec.StateWords} {env : Env}
    (fit : fits finish = true) (keys : stateKeys.length ≤ 8) (terminal : FinishTerminal finish stateKeys x env) :
    FinishTerminal finish stateKeys x (liveEnv finish env) := by
  obtain ⟨positive, one, iteration, states, keyWords, rows, blocks⟩ := terminal
  obtain ⟨-, blockFits⟩ := fits_spec fit
  have same : ∀ k, k ∈ programReads finish.application.1 → liveEnv finish env k = env k :=
    fun k read => liveEnv_read env (program_read read)
  have one' : liveEnv finish env 0 = 1 := (same 0 (mem_programReads.mpr (Or.inl rfl))).trans one
  refine ⟨positive, one', (same _ (mem_programReads.mpr (Or.inr (Or.inl rfl)))).trans iteration,
    fun lane => ⟨(same _ (mem_programReads.mpr ?_)).trans (states lane).1,
      (same _ (mem_programReads.mpr ?_)).trans (states lane).2⟩,
    fun k inside => (same _ (mem_programReads.mpr ?_)).trans (keyWords k inside), fun r member => ?_,
    fun block member => ?_⟩
  · have := lane.isLt
    omega
  · have := lane.isLt
    omega
  · omega
  · have inRows : Terminal.Package.row 0 r ∈ finishRows finish := List.mem_map_of_mem member
    exact (Lowering.Package.row_holds _ 0 one' r).mp ((holds_congr fun t term =>
      liveEnv_read env (row_read inRows term)).mpr ((Lowering.Package.row_holds env 0 one r).mpr (rows r member)))
  · obtain ⟨lanes, -, forms⟩ := blockFits block member
    have outputs : (fun lane : Fin 16 => liveEnv finish env (block.start + 1080 + lane)) =
        fun lane : Fin 16 => env (block.start + 1080 + lane) :=
      funext fun lane => liveEnv_read env (output_read member lane)
    have inputs : (fun lane : Fin 16 => (block.lanes.getD lane 0).eval (liveEnv finish env)) =
        fun lane : Fin 16 => (block.lanes.getD lane 0).eval env := funext fun lane => by
      have inside : lane.val < block.lanes.length := by rw [lanes]; exact lane.isLt
      rw [List.getD_eq_getElem _ _ inside]
      obtain ⟨form, linear, -⟩ := forms _ (List.getElem_mem inside)
      rw [linearForm_eval _ _ form linear, linearForm_eval _ _ form linear]
      exact congrArg (form.1 + ·) (congrArg List.sum (List.map_congr_left fun t term => by
        rw [liveEnv_read env (lane_read member (List.getElem_mem inside) linear term)]))
    rw [outputs, inputs]
    exact blocks block member

/-- A slot's claim words are live columns. -/
theorem slotClaim_live {finish : Package} (env : Env) {slot : ℕ} (inside : slot < 2) :
    slotClaim finish slot (liveEnv finish env) = slotClaim finish slot env := by
  unfold slotClaim
  refine congrArg Spec.statementOf (claimAt_decode (column := id) fun k low high => ?_)
  rw [id, liveEnv_read env (program_read (mem_programReads.mpr ?_))]
  have slots : slot = 0 ∨ slot = 1 := by omega
  rcases slots with rfl | rfl
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨low, high⟩)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨low, high⟩))))))

/-- Completeness of the live columns: a statement whose claim holds has finish columns that are 0
on each column without a cell, in which the program's checks hold and both slot claims are in their
layer-1 relations. -/
theorem holds_live {finish : Package} {stateKeys : List Word} (fit : fits finish = true)
    (keys : stateKeys.length ≤ 8) {s0 s1 : Layer1.Shape} {Φ : Polynomial F} {bar : ℕ → ℕ → F}
    {S0 S1 : Layer1.Setup F} {x : Spec.StateWords}
    (holds : Spec.OptionB.Holds s0 s1 Φ bar S0 S1 (FinishTerminal finish stateKeys) (slotClaim finish 0)
      (slotClaim finish 1) x) :
    ∃ env, (∀ k < finish.sboxStart, (live finish).getD k false = false → env k = 0) ∧
      FinishTerminal finish stateKeys x env ∧ (∃ z, Layer1.Relation s0 Φ bar S0 (slotClaim finish 0 env) z) ∧
      ∃ z, Layer1.Relation s1 Φ bar S1 (slotClaim finish 1 env) z := by
  obtain ⟨env, terminal, slot0, slot1⟩ := holds
  refine ⟨liveEnv finish env, fun k inside dead => liveEnv_dead env inside dead,
    finishTerminal_live fit keys terminal, ?_, ?_⟩
  · rw [slotClaim_live env (by decide)]
    exact slot0
  · rw [slotClaim_live env (by decide)]
    exact slot1

end CompressionLean.Sim.OptionB
