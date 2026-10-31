import NightstreamFPrime.Lifecycle.Nebula.HonestRows
import NightstreamFPrime.Lifecycle.Nebula.WitnessEncoding

/-! Owns completeness of the relation of a memory application: every step that the
model's `invoke` accepts from a reachable carry, with one application step on
its ports, has witness words that satisfy the memory program's validity
predicate, and the program's step function gives the state digest of the
output state and the output carry. It owns the record chains, the challenge
transcript, and the input state of the honest witness; the application's
rows come from `PortApp.complete`. -/

namespace NightstreamFPrime.Lifecycle.Nebula.Honest

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open Fin.CommRing

variable {p : Geometry} {c : MemoryCarry} {σ : Type} {inv : Invocation σ Digest}
variable {a : AppShape} (appIn appOut : Fin a.width → F) (aux : Fin a.aux → F)

/-! ### Chains and the challenge transcript -/

theorem ofFn_getD {α : Type} {n : ℕ} (l : List α) (d : α) (length : l.length = n) :
    List.ofFn (fun j : Fin n => l.getD j d) = l := by
  apply List.ext_getElem (by simp [length])
  intro k h₁ h₂
  simp only [List.getElem_ofFn, List.getD_eq_getElem _ _ h₂]

theorem opsLaneBits_eq (accepted : Accepted p c inv) :
    (witness p c inv appIn appOut aux).opsLaneBits = (opsLane p inv.records.ops).map bitWord := by
  show (List.ofFn fun j : Fin p.bOps => (encodeOp p (opSlot inv j) (diff c inv j)).lane).flatten = _
  conv_rhs => rw [← ofFn_getD inv.records.ops OpSlot.padSlot accepted.rows.shaped.opsLength]
  simp only [lane_encodeOp, opsLane, List.flatMap, List.map_flatten, List.map_ofFn,
    Function.comp_def, opSlot]

theorem scanLaneBits_eq (l : List ScanSlot) (length : l.length = p.bScan) :
    StepWitness.scanLaneBits (fun j : Fin p.bScan => encodeScan p (l.getD j ⟨0, 0⟩)) =
      (scanLane p l).map bitWord := by
  conv_rhs => rw [← ofFn_getD l ⟨0, 0⟩ length]
  simp only [StepWitness.scanLaneBits, lane_encodeScan, scanLane, List.flatMap, List.map_flatten,
    List.map_ofFn, Function.comp_def]

theorem idxEff_val (valid : p.Valid) (reach : Reach p c) :
    (witness p c inv appIn appOut aux).idxEff.val = (start p c inv).idx :=
  natWord_val (lt_trans (start_idx_lt valid reach) (Geometry.n_lt valid))

theorem previous_ops : (witness p c inv appIn appOut aux).previousDigest 0 = (start p c inv).seen.ops := by
  funext i
  fin_cases i <;> rfl

theorem previous_initial : (witness p c inv appIn appOut aux).previousDigest 4 = (start p c inv).seen.initial := by
  funext i
  fin_cases i <;> rfl

theorem previous_final : (witness p c inv appIn appOut aux).previousDigest 8 = (start p c inv).seen.final := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenOps :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 23 = (finished p c inv).seen.ops := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenInitial :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 27 = (finished p c inv).seen.initial := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenFinal :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 31 = (finished p c inv).seen.final := by
  funext i
  fin_cases i <;> rfl

theorem chainOps_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 23 =
      chainLink .ops (witness p c inv appIn appOut aux).idxEff ((witness p c inv appIn appOut aux).previousDigest 0)
        (packWords (witness p c inv appIn appOut aux).opsLaneBits) := by
  rw [chainLink_eq _ _ _ (by rw [opsLaneBits_eq appIn appOut aux accepted]; exact bitWord_bits),
    opsLaneBits_eq appIn appOut aux accepted, map_toBool_bitWord, idxEff_val appIn appOut aux valid reach, previous_ops,
    digest_seenOps, finished_seen]
  rfl

theorem chainInitial_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 27 =
      chainLink .mem (witness p c inv appIn appOut aux).idxEff ((witness p c inv appIn appOut aux).previousDigest 4)
        (packWords (StepWitness.scanLaneBits (witness p c inv appIn appOut aux).initial)) := by
  have lanes : StepWitness.scanLaneBits (witness p c inv appIn appOut aux).initial =
      (scanLane p inv.records.initialScan).map bitWord :=
    scanLaneBits_eq _ accepted.rows.shaped.initialLength
  rw [chainLink_eq _ _ _ (by rw [lanes]; exact bitWord_bits), lanes, map_toBool_bitWord,
    idxEff_val appIn appOut aux valid reach, previous_initial, digest_seenInitial, finished_seen]
  rfl

theorem chainFinal_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv appIn appOut aux).carryOut 31 =
      chainLink .mem (witness p c inv appIn appOut aux).idxEff ((witness p c inv appIn appOut aux).previousDigest 8)
        (packWords (StepWitness.scanLaneBits (witness p c inv appIn appOut aux).final)) := by
  have lanes : StepWitness.scanLaneBits (witness p c inv appIn appOut aux).final =
      (scanLane p inv.records.finalScan).map bitWord :=
    scanLaneBits_eq _ accepted.rows.shaped.finalLength
  rw [chainLink_eq _ _ _ (by rw [lanes]; exact bitWord_bits), lanes, map_toBool_bitWord,
    idxEff_val appIn appOut aux valid reach, previous_final, digest_seenFinal, finished_seen]
  rfl

theorem freshEta_row (valid : p.Valid) (reach : Reach p c) :
    etaChallenges ⟨planDigest p, ((witness p c inv appIn appOut aux).cIn 2).val,
      (witness p c inv appIn appOut aux).proposalDigest 0,
      StepWitness.carryDigest (witness p c inv appIn appOut aux).carryIn 35,
      (witness p c inv appIn appOut aux).proposalDigest 4⟩ = (witness p c inv appIn appOut aux).freshEta := by
  have ts : ((witness p c inv appIn appOut aux).cIn 2).val = c.ts := by
    rw [cIn_val appIn appOut aux (by decide), carryVector_ts, natWord_val (canonical valid reach).2.2]
  have first : (witness p c inv appIn appOut aux).proposalDigest 0 = inv.proposal.1 := by
    funext i
    fin_cases i <;> rfl
  have second : (witness p c inv appIn appOut aux).proposalDigest 4 = inv.proposal.2 := by
    funext i
    fin_cases i <;> rfl
  have root : StepWitness.carryDigest (witness p c inv appIn appOut aux).carryIn 35 = c.memRoot := by
    funext i
    fin_cases i <;> rfl
  rw [ts, first, second, root]
  rfl

/-- The memory rows of SPEC.md section 7 and the input state hold on the honest
witness. -/
theorem rowsHold (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    (witness p c inv appIn appOut aux).RowsHold (stateWords (List.ofFn appIn) (carryWords c)) :=
  { polyRows appIn appOut aux valid reach accepted with
    chainOps := chainOps_row appIn appOut aux valid reach accepted
    chainInitial := chainInitial_row appIn appOut aux valid reach accepted
    chainFinal := chainFinal_row appIn appOut aux valid reach accepted
    freshEta := freshEta_row appIn appOut aux valid reach
    stateIn := rfl }


end NightstreamFPrime.Lifecycle.Nebula.Honest

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Geometry} {app : PortApp p}

/-- The honest slots of an accepted invocation are well formed and decode to
its ports. -/
theorem honest_slots (valid : p.Valid) {c : MemoryCarry} (reach : Reach p c)
    {inv : Invocation app.State Digest} (accepted : Honest.Accepted p c inv) :
    (∀ j : Fin p.bOps, (encodeOp p (Honest.opSlot inv j) (Honest.diff c inv j)).WellFormed) ∧
      (List.ofFn fun j : Fin p.bOps =>
        (encodeOp p (Honest.opSlot inv j) (Honest.diff c inv j)).decode.port) = inv.ports := by
  have length := accepted.rows.shaped.opsLength
  have rows := (Honest.rowsHold (a := app.toAppShape) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    valid reach accepted).toPolyRows
  refine ⟨fun j => StepWitness.PolyRows.wellFormed _ rows j, ?_⟩
  have decoded (j : Fin p.bOps) :
      (encodeOp p (Honest.opSlot inv j) (Honest.diff c inv j)).decode = Honest.opSlot inv j := by
    have member : Honest.opSlot inv j ∈ inv.records.ops := by
      unfold Honest.opSlot
      rw [List.getD_eq_getElem _ _ (by rw [length]; exact j.isLt)]
      exact List.getElem_mem _
    exact decode_encodeOp (accepted.rows.shaped.opsFit _ member) _
  simp only [decoded]
  rw [Invocation.ports, ← Honest.ofFn_getD inv.records.ops OpSlot.padSlot length, List.map_ofFn]
  rfl

/-- Completeness of the relation: a step that the model accepts from a reachable carry,
with one application step on its ports, has witness words that satisfy the
memory program's validity predicate, and the program's step function gives
the state digest of the output state and the output carry. -/
theorem complete (valid : p.Valid) {c c' : MemoryCarry} (reach : Reach p c)
    {s : app.State} {inv : Invocation app.State Digest}
    (accepts : invoke (setup p) c inv = some c') (step : app.model.Step s inv.ports inv.next) :
    ∃ witness : List F, witness.length = Words.count p app.toAppShape ∧
      MemoryApp.valid p app (stateWords (List.ofFn s) (carryWords c)) witness ∧
      MemoryApp.step p app (stateWords (List.ofFn s) (carryWords c)) witness =
        stateWords (List.ofFn inv.next) (carryWords c') := by
  obtain ⟨accepted, rfl⟩ := Honest.accepted p c inv accepts
  obtain ⟨wf, decoded⟩ := honest_slots valid reach accepted
  obtain ⟨priv, appRows⟩ := app.complete valid _ step wf decoded
  have count := Words.memoryCount_eq (p := p)
  refine ⟨(Honest.witness p c inv s inv.next priv).words,
    StepWitness.length_words _, ?_, ?_⟩
  · unfold MemoryApp.valid
    rw [StepWitness.decode_words]
    exact ⟨Honest.rowsHold _ _ _ valid reach accepted, appRows⟩
  · unfold MemoryApp.step
    congr 1
    · congr 1
      funext k
      have := k.isLt
      rw [StepWitness.wordOf_words _ (by simp only [Words.appOut, Words.count]; omega),
        StepWitness.wordAt_appOut]
      rfl
    · show _ = List.ofFn (carryVector (Honest.finished p c inv))
      congr 1
      funext k
      have := k.isLt
      rw [StepWitness.wordOf_words _ (by simp only [Words.carryOut, Words.count]; omega),
        StepWitness.wordAt_carryOut]
      rfl

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
