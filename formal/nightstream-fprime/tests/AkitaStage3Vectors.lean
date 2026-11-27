import tests.AkitaVectors
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3
import NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix

/-! The gate of Stage 3 of `key30.aks` (`Replay/Stage3.lean`, `Replay/Stage3Weight.lean`). For each of the two positive
`key30.aks` vectors the replay of `Replay/Transcript.lean` (the one replay of
`tests/AkitaAllVectors.lean`) runs from the public inputs and the proof bytes; it evaluates
`stage3.end`, which the Rust test of the vectors cannot evaluate (the
field `unchecked` of the vector): `stage3Run` with the weight `stage3Weight` of the plan of the
root (`SetupPlan.ofBatch` of `Key30.root` in the order `groupOrder`, the plan of the layout) must
accept in the replay (`Derived.stages`), and a changed prefix evaluation, a changed point of Stage 3, a changed
relation point, a changed `α`, a changed `τ_1` and the groups in their own order must each fail
it. The events are read only for the comparisons.

- The natural length of the setup prefix in the slot bytes is the length of the plan (the check
  of `setup_prefix_coverage_eval_len`), and the rounds of Stage 3 are the rounds of the plan.
- The slot and the commitment rows of the setup prefix are trusted inputs of the code: the
  registry of the verifier setup (`AkitaVerifierSetup::from_parts`), which Nightstream reads from
  its verification key. Each is a byte string of `crates/ns-compress/artifacts/key30.vk`.
- Level 1 opens the group of the setup prefix (T8): its absorbs `point.0`, `point.1` and
  `openings` and its two drawn claim coefficients equal the events.
- Pages 0 and 511 of the seed (the first and the last page of the `2^21` elements that the
  verifier expands) satisfy the hypothesis of `page_expand`. -/

namespace AkitaStage3Vectors

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay
open AkitaVectors (Replayed find? eventExts)

def failWith {α : Type} (message : String) : IO α := throw <| IO.userError message

def at0 (site : String) : Key := ⟨0, site, none⟩

/-- `bytes` holds `part` from some offset. -/
def holds (bytes part : ByteArray) : Bool :=
  (List.range (bytes.size + 1 - part.size)).any fun i =>
    (List.range part.size).all fun j => bytes.get! (i + j) == part.get! j

def rowBytes (rows : Array Fq) : ByteArray :=
  rows.foldl (fun out a => out ++ Setup.littleEndian 8 a.word.toNat) ByteArray.empty

def bump (xs : List Ext) (i : ℕ) : List Ext := xs.modify i (· + Ext.one)

/-- The gate on the replay of one positive `key30.aks` file. -/
def gate (r : Replayed) (vk : ByteArray) : IO Unit := do
  let start ← IO.monoMsNow
  let v := r.v
  let o := r.o
  let name := v.name
  let layout := r.layout
  let pub := r.pub
  let d := r.d
  let recorded (site : String) : Array Ext := (d.recorded.get? (at0 site)).getD #[]
  let first (site : String) : Ext := (recorded site).getD 0 Ext.zero
  let some F := layout.folds.head? | failWith s!"{name}: no root"
  let some rounds := F.stage3 | failWith s!"{name}: the root has no Stage 3"
  let row := if v.shape.2.1 = 27 then Key30.w27 else Key30.w26
  let plan := SetupPlan.ofBatch (Key30.root row) (groupOrder 3) F.low F.logBasis
  let α := first "alpha"
  let τ := (recorded "tau1").toList
  let r2 := (recorded "fused.point").toList
  let σ := first "stage3.setup_claim"
  let value := first "stage3.prefix_evaluation"
  let ρ := (recorded "stage3.point").toList
  let stageRounds := ((d.rounds.find? (·.1 == at0 "stage3")).map (·.2)).getD []
  -- the shapes of `stage3Run_runs`
  unless stageRounds.length = rounds ∧ stageRounds.all (·.1.length = 2) ∧ ρ.length = rounds do
    failWith s!"{name}: Stage 3 has {stageRounds.length} rounds, the layout {rounds}"
  unless (plainRun σ stageRounds).point == ρ ∧
      (plainRun σ stageRounds).claim == first "stage3.output" do
    failWith s!"{name}: the rounds of Stage 3 differ from the replay"
  -- the plan against the slot: natural length and rounds
  let G := plan.groups
  let base := plan.base
  let footprint := G.foldl (fun m g => max m (max (g.innerRank * g.zCols * (g.innerDegree / base))
    (g.outerRank * g.physWidth * (g.outerDegree / base)))) (plan.openRank * plan.dColsAll *
      (plan.openDegree / base))
  let natural := footprint * base
  unless Wire.word pub.slot 4 == UInt64.ofNat natural ∧ pub.slot.extract 0 4 == "SPF4".toUTF8 do
    failWith s!"{name}: the slot has another natural length than the plan ({natural})"
  unless Nat.log 2 base + Nat.log 2 (footprint.nextPowerOfTwo) = rounds do
    failWith s!"{name}: the plan has another count of rounds"
  -- stage3.end of the replay
  unless F.setup == some plan do failWith s!"{name}: the layout has another plan of Stage 3"
  let weight := stage3Weight plan α τ r2
  let some (_, .stage3 (some (point, _))) := d.stages.find? (·.1 == at0 "stage3.end") |
    failWith s!"{name}: stage3.end of the replay fails: output {repr (first "stage3.output")}, \
      value·weight {repr (value * weight ρ)}"
  unless point == ρ do failWith s!"{name}: the point of Stage 3"
  -- each change fails; one task for each
  let output := first "stage3.output"
  let changed : List (String × Task Bool) := [
    ("value + 1", Task.spawn fun _ => (stage3Run σ stageRounds (value + Ext.one) weight).isSome),
    ("ρ_0 + 1", Task.spawn fun _ => (value * weight (bump ρ 0)).same output),
    ("ρ_last + 1", Task.spawn fun _ => (value * weight (bump ρ (rounds - 1))).same output),
    ("r_2[κ] + 1", Task.spawn fun _ =>
      (stage3Run σ stageRounds value (stage3Weight plan α τ (bump r2 F.low))).isSome),
    ("α + 1", Task.spawn fun _ =>
      (stage3Run σ stageRounds value (stage3Weight plan (α + Ext.one) τ r2)).isSome),
    ("τ_1[0] + 1", Task.spawn fun _ =>
      (stage3Run σ stageRounds value (stage3Weight plan α (bump τ 0) r2)).isSome),
    ("own group order", Task.spawn fun _ => (stage3Run σ stageRounds value
      (stage3Weight (SetupPlan.ofBatch (Key30.root row) (Equiv.refl _) F.low F.logBasis) α τ
        r2)).isSome)]
  for (label, holds) in changed do
    if holds.get then failWith s!"{name}: stage3.end holds for the change {label}"
  -- the trusted inputs of the registry
  unless holds vk pub.slot ∧ holds vk (rowBytes pub.setupRows) ∧ pub.setupRows.size = 32 do
    failWith s!"{name}: the slot or the rows of the setup prefix are not in key30.vk"
  -- level 1 opens the setup prefix (T8)
  for site in ["point.0", "point.1", "openings"] do
    let some e := find? o "absorb" 1 site | failWith s!"{name}: no event {site}"
    let some (_, bytes) := d.absorbs.find? (·.1 == ⟨1, site, none⟩) | failWith s!"{name}: {site}"
    unless extBytes (← IO.ofExcept (eventExts e)) == bytes do failWith s!"{name}: level 1 {site}"
  let some e := find? o "challenge" 1 "row_coefficients" | failWith s!"{name}: no coefficients"
  let some (_, drawn) := d.challenges.find? (·.1 == ⟨1, "row_coefficients", none⟩) |
    failWith s!"{name}: the replay draws no claim coefficients at level 1"
  unless drawn == (← IO.ofExcept (eventExts e)) ∧ drawn.size = 2 do
    failWith s!"{name}: the claim coefficients of level 1"
  unless (← IO.ofExcept (eventExts (← (find? o "absorb" 1 "point.0").getDM
      (failWith "point.0")))).toList == ρ do
    failWith s!"{name}: level 1 does not open the setup prefix at the point of Stage 3"
  let done ← IO.monoMsNow
  IO.println s!"stage3 {name}: {rounds} rounds, plan of {G.length} groups (base {base}, natural \
    length {natural}), stage3.end holds; 7 changes fail; slot and rows in key30.vk; level 1 \
    point.0 = stage3.point, openings, 2 claim coefficients = events ({done - start} ms)"

/-- The hypothesis of `page_expand` holds on page `j` of the seed of the code: the loop keeps
`4096` words. -/
def pageGate (j : ℕ) : IO Unit := do
  let some kept := Setup.page Setup.sampleSeed j Setup.pageSize | failWith s!"page {j}: none"
  unless kept.size = Setup.pageSize do failWith s!"page {j}: {kept.size} elements"
  IO.println s!"page {j} of the seed: Setup.page keeps {kept.size} words (the hypothesis of \
    page_expand)"

end AkitaStage3Vectors
