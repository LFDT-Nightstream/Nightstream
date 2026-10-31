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
slot 1 needs two refresh steps in a row. `refreshSchedule` never reaches the cap. -/

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

/-- One period: a light step folds into the refresh slot, `cap` light steps fill the light slot,
and two refresh steps reset the light slot and then the refresh slot. -/
def refreshPeriod (cap : Nat) : List (Fin 2) :=
  0 :: (List.replicate cap 0 ++ [1, 1])

/-- After a refresh step, any number of periods runs without reaching the cap. -/
def refreshSchedule (cap periods : Nat) : List (Fin 2) :=
  (List.replicate periods (refreshPeriod cap)).flatten

theorem runDepths_refreshPeriod {cap : Nat} (capPositive : 0 < cap) (rest : List (Fin 2)) :
    runDepths cap (fun _ => 0) 1 (refreshPeriod cap ++ rest) =
      runDepths cap (fun _ => 0) 1 rest := by
  have intoRefreshSlot : ¬(circuitStep 0 = .light ∧ cap ≤ 0) := fun ⟨_, atCap⟩ => by omega
  simp only [refreshPeriod, List.cons_append, runDepths, depthStep, intoRefreshSlot,
    ↓reduceIte, Option.bind_some]
  rw [List.append_assoc, runDepths_lights cap _ cap _ (by simp)]
  have refreshedLight : ∀ depth : Nat, ¬(circuitStep 1 = .light ∧ cap ≤ depth) :=
    fun _ ⟨light, _⟩ => by simp [circuitStep] at light
  simp only [List.cons_append, List.nil_append, runDepths, depthStep, refreshedLight,
    ↓reduceIte, Option.bind_some]
  congr
  funext slot
  match slot with
  | ⟨0, _⟩ => simp [circuitStep, nextDepth]
  | ⟨1, _⟩ => simp [circuitStep, nextDepth]

/-- An honest prover that follows `refreshSchedule` never meets the depth cap. -/
theorem runDepths_refreshSchedule {cap : Nat} (capPositive : 0 < cap) :
    ∀ periods, runDepths cap (fun _ => 0) 1 (refreshSchedule cap periods) = some fun _ => 0
  | 0 => rfl
  | periods + 1 => by
      simp only [refreshSchedule, List.replicate_succ, List.flatten_cons]
      rw [runDepths_refreshPeriod capPositive]
      exact runDepths_refreshSchedule capPositive periods

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
