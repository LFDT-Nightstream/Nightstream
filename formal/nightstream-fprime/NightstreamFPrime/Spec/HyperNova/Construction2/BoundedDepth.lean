import NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit
import NightstreamFPrime.Spec.Folding.Schedule

/-! Option B on per-circuit Construction 2: slot 0 is the light circuit and slot 1 the refresh
circuit. A running claim carries its depth, so the state hash binds it. The light verifier
rejects a claim at the depth cap, and the refresh verifier resets the depth to 0.

Soundness: the accepted folds on any slot form an admissible schedule, so the bounds of
`Folding.Schedule` apply to them.

Completeness: the wrapper accepts every fold the underlying verifier accepts below the cap. The
control function sees only the application state, so the application must schedule refreshes.
Each slot is refreshed only when the refresh circuit runs right after that slot's circuit, so
slot 1 needs two refresh steps in a row. `refreshSchedule` never reaches the cap and refreshes
one step in `cap + 1`. `light_count_le` shows that no schedule refreshes less often. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Spec.Folding.Schedule (Step nextDepth Admissible)

universe uKey uClaim uFresh uProof

def circuitStep (circuit : Fin 2) : Step :=
  if circuit.val = 0 then .light else .refresh

/-- A running claim and the number of light folds since its last refresh. -/
abbrev Running (Claim : Type uClaim) := Claim × Nat

/-- The underlying fold verifier of each step kind. -/
abbrev Folds (Key : Type uKey) (Claim : Type uClaim) (Fresh : Type uFresh)
    (Proof : Type uProof) :=
  Step → Key → Claim → Fresh → Proof → Option Claim

variable {Key : Type uKey} {Claim : Type uClaim} {Fresh : Type uFresh} {Proof : Type uProof}

def verifier (cap : Nat) (folds : Folds Key Claim Fresh Proof) (circuit : Fin 2) :
    Verifier Key (Running Claim) Fresh Proof where
  verify := fun key running fresh proof =>
    if circuitStep circuit = .light ∧ cap ≤ running.2 then none
    else (folds (circuitStep circuit) key running.1 fresh proof).map fun claim =>
      (claim, nextDepth running.2 (circuitStep circuit))

def setup (cap : Nat) (folds : Folds Key Claim Fresh Proof) (verifierKeys : Fin 2 → Key)
    (defaultClaim : Claim) : PerCircuit.Setup Key (Running Claim) Fresh Proof 2 where
  verifierKeys := verifierKeys
  nifs := verifier cap folds
  defaultRunning := (defaultClaim, 0)

/-- The wrapper accepts exactly the folds the underlying verifier accepts, except a light fold
at the cap, and moves the depth as the schedule does. -/
theorem accepts_iff (cap : Nat) (folds : Folds Key Claim Fresh Proof) (circuit : Fin 2)
    (key : Key) (claim : Claim) (depth : Nat) (fresh : Fresh) (proof : Proof)
    (output : Running Claim) :
    Accepts (verifier cap folds circuit) key (claim, depth) fresh proof output ↔
      (circuitStep circuit = .light → depth < cap) ∧
        folds (circuitStep circuit) key claim fresh proof = some output.1 ∧
        output.2 = nextDepth depth (circuitStep circuit) := by
  unfold Accepts verifier
  by_cases blocked : circuitStep circuit = .light ∧ cap ≤ depth
  · simp only [eq_true blocked, ↓reduceIte, reduceCtorEq, false_iff]
    rintro ⟨room, _⟩
    exact absurd (room blocked.1) (Nat.not_lt.mpr blocked.2)
  · simp only [blocked, ↓reduceIte]
    have room : circuitStep circuit = .light → depth < cap := fun light =>
      Nat.lt_of_not_le fun atCap => blocked ⟨light, atCap⟩
    cases folded : folds (circuitStep circuit) key claim fresh proof with
    | none => simp
    | some folded =>
        obtain ⟨outClaim, outDepth⟩ := output
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
        constructor
        · rintro ⟨rfl, rfl⟩
          exact ⟨room, rfl, rfl⟩
        · rintro ⟨_, rfl, rfl⟩
          exact ⟨rfl, rfl⟩

structure Fold (Key : Type uKey) (Claim : Type uClaim) (Fresh : Type uFresh)
    (Proof : Type uProof) where
  circuit : Fin 2
  key : Key
  input : Claim
  fresh : Fresh
  proof : Proof
  output : Running Claim

/-- The folds a slot receives, in order, each accepted at the depth the previous one left. -/
def AcceptedFrom (cap : Nat) (folds : Folds Key Claim Fresh Proof) :
    Nat → List (Fold Key Claim Fresh Proof) → Prop
  | _, [] => True
  | depth, fold :: rest =>
      Accepts (verifier cap folds fold.circuit) fold.key (fold.input, depth) fold.fresh
          fold.proof fold.output ∧
        AcceptedFrom cap folds fold.output.2 rest

def steps (history : List (Fold Key Claim Fresh Proof)) : List Step :=
  history.map fun fold => circuitStep fold.circuit

theorem admissible_of_accepted (cap : Nat) (folds : Folds Key Claim Fresh Proof) :
    ∀ (history : List (Fold Key Claim Fresh Proof)) (depth : Nat),
      AcceptedFrom cap folds depth history → Admissible cap depth (steps history)
  | [], _, _ => trivial
  | fold :: rest, depth, ⟨accepted, later⟩ => by
      obtain ⟨room, _, moved⟩ := (accepts_iff cap folds fold.circuit fold.key fold.input depth
        fold.fresh fold.proof fold.output).mp accepted
      have tail := admissible_of_accepted cap folds rest fold.output.2 later
      simp only [steps, List.map_cons]
      cases kind : circuitStep fold.circuit with
      | light =>
          simp only [kind, nextDepth] at moved
          rw [moved] at tail
          exact ⟨room kind, tail⟩
      | refresh =>
          simp only [kind, nextDepth] at moved
          rw [moved] at tail
          exact tail

/-- Every accumulator of an accepted history stays below `B` when honest, and extraction
recovers each below `B + cap · T · (b − 1)`. -/
theorem accepted_bounds (p : GlobalParams) {cap : Nat} (capFits : 1 + cap ≤ p.maxFresh)
    (folds : Folds Key Claim Fresh Proof) (history : List (Fold Key Claim Fresh Proof))
    (depth : Nat) (start : depth ≤ cap) (accepted : AcceptedFrom cap folds depth history)
    {final : Nat} (finalAtMostB : final ≤ p.bigB) :
    (∀ reached ∈ Folding.Schedule.depths depth (steps history),
        p.accumulatorBound reached < p.bigB) ∧
      ∀ bound ∈ Folding.Schedule.recoveredBounds p final (steps history),
        bound ≤ p.bigB + cap * (p.expansionT * (p.b - 1)) := by
  have admissible := admissible_of_accepted cap folds history depth accepted
  exact ⟨Folding.Schedule.honest_bounds p capFits (steps history) depth start admissible,
    Folding.Schedule.recoveredBounds_le p finalAtMostB (steps history) depth admissible⟩

/-- The slot depths after a recursive step that runs `circuit` after `prior`, or `none` when the
light verifier rejects. -/
def depthStep (cap : Nat) (depth : Fin 2 → Nat) (prior circuit : Fin 2) :
    Option (Fin 2 → Nat) :=
  if circuitStep circuit = .light ∧ cap ≤ depth prior then none
  else some fun slot =>
    if slot = prior then nextDepth (depth prior) (circuitStep circuit) else depth slot

/-- A recursive step of the two-slot machine moves the slot depths by `depthStep`. -/
theorem transition_depthStep
    {Digest State Witness Encoded : Type}
    (cap : Nat) (folds : Folds Key Claim Fresh Proof) (verifierKeys : Fin 2 → Key)
    (defaultClaim : Claim)
    (machine : Paper.Machine Key Digest State Witness (Running Claim) Fresh Encoded 2)
    (functionIndex : Fin 2)
    (input : Paper.Input Key State Witness (Running Claim) Fresh Proof 2)
    (output : Paper.Output Digest State (Running Claim) 2)
    (holds : PerCircuit.Transition (setup cap folds verifierKeys defaultClaim) machine
      functionIndex input output)
    (recursive : 0 < input.iteration) :
    ∃ priorPcValid : Paper.InRange 2 input.priorPc,
      depthStep cap (fun slot => (input.running slot).2) (Paper.selectedIndex priorPcValid)
          functionIndex =
        some fun slot => (output.runningNext slot).2 := by
  obtain ⟨priorPcValid, accepted, unchanged⟩ := PerCircuit.recursive_selected
    (setup cap folds verifierKeys defaultClaim) machine functionIndex input output holds recursive
  refine ⟨priorPcValid, ?_⟩
  obtain ⟨room, _, moved⟩ := (accepts_iff cap folds functionIndex _
    (input.running (Paper.selectedIndex priorPcValid)).1
    (input.running (Paper.selectedIndex priorPcValid)).2 _ _ _).mp accepted
  have open_ : ¬(circuitStep functionIndex = .light ∧
      cap ≤ (input.running (Paper.selectedIndex priorPcValid)).2) := fun ⟨light, atCap⟩ =>
    absurd (room light) (Nat.not_lt.mpr atCap)
  simp only [depthStep, eq_false open_, ↓reduceIte, Option.some.injEq]
  funext slot
  by_cases same : slot = Paper.selectedIndex priorPcValid
  · subst same
    simp only [↓reduceIte]
    exact moved.symm
  · simp only [same, ↓reduceIte, unchanged slot same]

/-- The slot depths after running `circuits` once the circuit `prior` has run, or `none` when a
light fold reaches the cap. -/
def runDepths (cap : Nat) : (Fin 2 → Nat) → Fin 2 → List (Fin 2) → Option (Fin 2 → Nat)
  | depth, _, [] => some depth
  | depth, prior, circuit :: rest =>
      (depthStep cap depth prior circuit).bind fun next => runDepths cap next circuit rest

theorem runDepths_lights (cap : Nat) (rest : List (Fin 2)) :
    ∀ (count : Nat) (depth : Fin 2 → Nat), depth 0 + count ≤ cap →
      runDepths cap depth 0 (List.replicate count 0 ++ rest) =
        runDepths cap (fun slot => if slot = 0 then depth 0 + count else depth slot) 0 rest
  | 0, depth, _ => by
      congr
      funext slot
      by_cases zero : slot = 0
      · subst zero
        rfl
      · simp [zero]
  | count + 1, depth, room => by
      have open_ : ¬(circuitStep 0 = .light ∧ cap ≤ depth 0) := fun ⟨_, atCap⟩ => by omega
      simp only [List.replicate_succ, List.cons_append, runDepths, depthStep, open_,
        ↓reduceIte, Option.bind_some]
      rw [runDepths_lights cap rest count _ (by simp [circuitStep, nextDepth]; omega)]
      congr
      funext slot
      by_cases zero : slot = 0
      · subst zero
        simp [circuitStep, nextDepth]
        omega
      · simp [zero]

theorem slot_cases (slot : Fin 2) : slot = 0 ∨ slot = 1 :=
  match slot with
  | ⟨0, _⟩ => .inl rfl
  | ⟨1, _⟩ => .inr rfl

theorem depthStep_light {cap : Nat} {depth : Fin 2 → Nat} {prior : Fin 2}
    (room : depth prior < cap) :
    depthStep cap depth prior 0 =
      some fun slot => if slot = prior then depth prior + 1 else depth slot := by
  have open_ : ¬(circuitStep 0 = .light ∧ cap ≤ depth prior) := fun ⟨_, atCap⟩ => by omega
  simp only [depthStep, eq_false open_, ↓reduceIte]
  rfl

theorem depthStep_refresh (cap : Nat) (depth : Fin 2 → Nat) (prior : Fin 2) :
    depthStep cap depth prior 1 = some fun slot => if slot = prior then 0 else depth slot := by
  simp [depthStep, circuitStep, nextDepth]

/-- One block: `cap + 1` light steps and a refresh step. The first light step folds into the
refresh slot, the next `cap` fill the light slot, and the refresh step resets the light slot. -/
def refreshBlock (cap : Nat) : List (Fin 2) :=
  List.replicate (cap + 1) 0 ++ [1]

/-- One period: `cap` blocks, then a refresh step that resets the refresh slot. -/
def refreshPeriod (cap : Nat) : List (Fin 2) :=
  (List.replicate cap (refreshBlock cap)).flatten ++ [1]

/-- After a refresh step, any number of periods runs without reaching the cap. -/
def refreshSchedule (cap periods : Nat) : List (Fin 2) :=
  (List.replicate periods (refreshPeriod cap)).flatten

/-- A block takes one light fold into the refresh slot and leaves the light slot at 0. -/
theorem runDepths_refreshBlock {cap used : Nat} (room : used < cap) (depth : Fin 2 → Nat)
    (lightSlot : depth 0 = 0) (refreshSlot : depth 1 = used) (rest : List (Fin 2)) :
    runDepths cap depth 1 (refreshBlock cap ++ rest) =
      runDepths cap (fun slot => if slot = 0 then 0 else used + 1) 1 rest := by
  rw [refreshBlock, List.append_assoc, List.replicate_succ, List.cons_append, runDepths,
    depthStep_light (by omega), Option.bind_some,
    runDepths_lights cap _ cap _ (by simp [lightSlot]), List.singleton_append, runDepths,
    depthStep_refresh, Option.bind_some]
  congr 1
  funext slot
  rcases slot_cases slot with rfl | rfl <;> simp [refreshSlot]

theorem runDepths_refreshBlocks {cap : Nat} (rest : List (Fin 2)) :
    ∀ (blocks used : Nat) (depth : Fin 2 → Nat), used + blocks ≤ cap → depth 0 = 0 →
      depth 1 = used →
      runDepths cap depth 1 ((List.replicate blocks (refreshBlock cap)).flatten ++ rest) =
        runDepths cap (fun slot => if slot = 0 then 0 else used + blocks) 1 rest
  | 0, used, depth, _, lightSlot, refreshSlot => by
      simp only [List.replicate_zero, List.flatten_nil, List.nil_append, Nat.add_zero]
      congr 1
      funext slot
      rcases slot_cases slot with rfl | rfl <;> simp [lightSlot, refreshSlot]
  | blocks + 1, used, depth, room, lightSlot, refreshSlot => by
      rw [List.replicate_succ, List.flatten_cons, List.append_assoc,
        runDepths_refreshBlock (by omega) depth lightSlot refreshSlot,
        runDepths_refreshBlocks rest blocks (used + 1) _ (by omega) (by simp) (by simp),
        show used + 1 + blocks = used + (blocks + 1) by omega]

theorem runDepths_refreshPeriod (cap : Nat) (rest : List (Fin 2)) :
    runDepths cap (fun _ => 0) 1 (refreshPeriod cap ++ rest) =
      runDepths cap (fun _ => 0) 1 rest := by
  rw [refreshPeriod, List.append_assoc, runDepths_refreshBlocks _ cap 0 _ (by omega) rfl rfl,
    List.singleton_append, runDepths, depthStep_refresh, Option.bind_some]
  congr 1
  funext slot
  rcases slot_cases slot with rfl | rfl <;> simp

/-- An honest prover that follows `refreshSchedule` never meets the depth cap. -/
theorem runDepths_refreshSchedule (cap : Nat) :
    ∀ periods, runDepths cap (fun _ => 0) 1 (refreshSchedule cap periods) = some fun _ => 0
  | 0 => rfl
  | periods + 1 => by
      simp only [refreshSchedule, List.replicate_succ, List.flatten_cons]
      rw [runDepths_refreshPeriod]
      exact runDepths_refreshSchedule cap periods

theorem count_flatten_replicate (slot : Fin 2) (block : List (Fin 2)) :
    ∀ copies, ((List.replicate copies block).flatten).count slot = copies * block.count slot
  | 0 => by simp
  | copies + 1 => by
      rw [List.replicate_succ, List.flatten_cons, List.count_append,
        count_flatten_replicate slot block copies, Nat.succ_mul]
      exact Nat.add_comm _ _

/-- A period takes `cap` light steps per refresh step: one step in `cap + 1` is a refresh. -/
theorem refreshPeriod_count (cap : Nat) :
    (refreshPeriod cap).count 0 = cap * (refreshPeriod cap).count 1 := by
  simp only [refreshPeriod, List.count_append, count_flatten_replicate]
  simp [refreshBlock, List.count_replicate]

/-- Each light step adds 1 to a depth and each refresh step removes at most `cap`, so a run that
never meets the cap takes at most `cap` light steps per refresh step, plus what the final depths
hold. -/
theorem runDepths_light_count {cap : Nat} :
    ∀ (circuits : List (Fin 2)) (depth : Fin 2 → Nat) (prior : Fin 2) (final : Fin 2 → Nat),
      (∀ slot, depth slot ≤ cap) → runDepths cap depth prior circuits = some final →
      (∀ slot, final slot ≤ cap) ∧
        circuits.count 0 + (depth 0 + depth 1) ≤ cap * circuits.count 1 + (final 0 + final 1)
  | [], depth, _, final, bounded, ran => by
      simp only [runDepths, Option.some.injEq] at ran
      subst ran
      exact ⟨bounded, by simp⟩
  | circuit :: rest, depth, prior, final, bounded, ran => by
      cases step : depthStep cap depth prior circuit with
      | none => simp [runDepths, step] at ran
      | some next =>
          simp only [runDepths, step, Option.bind_some] at ran
          have priorBound := bounded prior
          rcases slot_cases circuit with rfl | rfl
          · have room : depth prior < cap := Nat.lt_of_not_le fun atCap => by
              simp [depthStep, circuitStep, atCap] at step
            rw [depthStep_light room, Option.some.injEq] at step
            subst step
            obtain ⟨finalBound, count⟩ := runDepths_light_count rest _ 0 final
              (fun slot => by by_cases same : slot = prior <;> simp [same, bounded] <;> omega)
              ran
            refine ⟨finalBound, ?_⟩
            rcases slot_cases prior with rfl | rfl <;> simp at count ⊢ <;> omega
          · rw [depthStep_refresh, Option.some.injEq] at step
            subst step
            obtain ⟨finalBound, count⟩ := runDepths_light_count rest _ 1 final
              (fun slot => by by_cases same : slot = prior <;> simp [same, bounded])
              ran
            refine ⟨finalBound, ?_⟩
            rcases slot_cases prior with rfl | rfl <;> simp [Nat.mul_add] at count ⊢ <;> omega

/-- A run that starts with every depth at 0 and never meets the cap takes at most `cap` light
steps per refresh step, plus `2 · cap`. `refreshPeriod_count` meets this rate exactly. -/
theorem light_count_le {cap : Nat} (circuits : List (Fin 2)) (prior : Fin 2)
    (final : Fin 2 → Nat) (ran : runDepths cap (fun _ => 0) prior circuits = some final) :
    circuits.count 0 ≤ cap * circuits.count 1 + 2 * cap := by
  obtain ⟨finalBound, count⟩ := runDepths_light_count circuits _ prior final
    (fun _ => Nat.zero_le cap) ran
  have := finalBound 0
  have := finalBound 1
  omega

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
