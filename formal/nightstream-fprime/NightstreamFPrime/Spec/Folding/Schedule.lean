import NightstreamFPrime.Spec.Folding.Light.Bound

/-! The bounded-depth schedule of option B (Cyclo, ePrint 2026/359, Section 2.2). An accumulator
takes light folds up to a cap, and a refresh fold resets its depth to 0.

Honest runs: an accumulator at depth `d` has norm at most `accumulatorBound d`, below `B` for
every depth the cap allows. Extraction runs backward: a light fold recovers its input at its
output bound plus `T · (b − 1)`, and a refresh recovers its input below `B`, so every recovered
accumulator stays below `B + cap · T · (b − 1)`. Relaxed binding at twice that bound gives the
MSIS norm `8 · T · (B + cap · T · (b − 1))` (SuperNeo v1.2 Section 8 with `B` replaced). -/

namespace NightstreamFPrime.Spec.Folding.Schedule

open NightstreamFPrime.Spec

inductive Step where
  | light
  | refresh
deriving DecidableEq, Repr

def nextDepth : Nat → Step → Nat
  | depth, .light => depth + 1
  | _, .refresh => 0

/-- No light fold runs at or past the cap; a refresh may run at any depth. -/
def Admissible (cap : Nat) : Nat → List Step → Prop
  | _, [] => True
  | depth, .light :: rest => depth < cap ∧ Admissible cap (depth + 1) rest
  | _, .refresh :: rest => Admissible cap 0 rest

/-- The accumulator's depth before each step and after the last. -/
def depths : Nat → List Step → List Nat
  | depth, [] => [depth]
  | depth, step :: rest => depth :: depths (nextDepth depth step) rest

theorem depths_le_cap {cap : Nat} :
    ∀ (steps : List Step) (depth : Nat), depth ≤ cap → Admissible cap depth steps →
      ∀ reached ∈ depths depth steps, reached ≤ cap
  | [], depth, start, _, reached, member => by
      simp only [depths, List.mem_singleton] at member
      omega
  | .light :: rest, depth, start, admissible, reached, member => by
      simp only [depths, List.mem_cons] at member
      rcases member with same | later
      · omega
      · exact depths_le_cap rest (depth + 1) admissible.1 admissible.2 reached later
  | .refresh :: rest, depth, start, admissible, reached, member => by
      simp only [depths, List.mem_cons] at member
      rcases member with same | later
      · omega
      · exact depths_le_cap rest 0 (by omega) admissible reached later

theorem accumulatorBound_succ (p : GlobalParams) (depth : Nat) :
    p.accumulatorBound (depth + 1) = p.accumulatorBound depth + p.expansionT * (p.b - 1) := by
  unfold GlobalParams.accumulatorBound
  rw [show 1 + (depth + 1) + p.k = (1 + depth + p.k) + 1 by omega, Nat.add_mul, Nat.add_mul,
    Nat.one_mul, Nat.mul_assoc]

/-- A light fold takes an accumulator within `accumulatorBound d` to one within
`accumulatorBound (d + 1)`; this is the bound `Light.complete` gives. -/
theorem light_bound (p : GlobalParams) (depth : Nat) :
    p.accumulatorBound depth + 1 + p.expansionT * (p.b - 1) =
      p.accumulatorBound (depth + 1) + 1 := by
  rw [accumulatorBound_succ]
  omega

/-- Along an admissible run, the bound `accumulatorBound` of each reached depth stays below `B`.
This is arithmetic on the bound, not a statement about an accumulator's norm. -/
theorem honest_bounds (p : GlobalParams) {cap : Nat} (capFits : 1 + cap ≤ p.maxFresh)
    (steps : List Step) (depth : Nat) (start : depth ≤ cap)
    (admissible : Admissible cap depth steps) :
    ∀ reached ∈ depths depth steps, p.accumulatorBound reached < p.bigB := by
  intro reached member
  have atMost := depths_le_cap steps depth start admissible reached member
  exact p.accumulatorBound_lt_bigB (by omega)

/-- The bound at which extraction recovers the accumulator before a step, from the bound of the
accumulator after it. -/
def extractedBefore (p : GlobalParams) : Step → Nat → Nat
  | .light, after => after + p.expansionT * (p.b - 1)
  | .refresh, _ => p.bigB

/-- The recovered bound before the first step, from the bound checked after the last. -/
def recoveredBound (p : GlobalParams) (final : Nat) : List Step → Nat
  | [] => final
  | step :: rest => extractedBefore p step (recoveredBound p final rest)

/-- Recovered bounds of the whole run, first step first, the checked final bound last. -/
def recoveredBounds (p : GlobalParams) (final : Nat) : List Step → List Nat
  | [] => [final]
  | step :: rest => recoveredBound p final (step :: rest) :: recoveredBounds p final rest

theorem recoveredBound_le (p : GlobalParams) {cap final : Nat} (finalAtMostB : final ≤ p.bigB) :
    ∀ (steps : List Step) (depth : Nat), Admissible cap depth steps →
      recoveredBound p final steps ≤ p.bigB + (cap - depth) * (p.expansionT * (p.b - 1))
  | [], _, _ => by
      simp only [recoveredBound]
      omega
  | .light :: rest, depth, admissible => by
      have later := recoveredBound_le p finalAtMostB rest (depth + 1) admissible.2
      have below := admissible.1
      simp only [recoveredBound, extractedBefore]
      have steps : cap - depth = (cap - (depth + 1)) + 1 := by omega
      rw [steps, Nat.add_mul, Nat.one_mul]
      omega
  | .refresh :: _, _, _ => by
      simp only [recoveredBound, extractedBefore]
      omega

/-- Every accumulator that extraction recovers along an admissible run stays below
`B + cap · T · (b − 1)`. -/
theorem recoveredBounds_le (p : GlobalParams) {cap final : Nat} (finalAtMostB : final ≤ p.bigB) :
    ∀ (steps : List Step) (depth : Nat), Admissible cap depth steps →
      ∀ bound ∈ recoveredBounds p final steps,
        bound ≤ p.bigB + cap * (p.expansionT * (p.b - 1))
  | [], _, _, bound, member => by
      simp only [recoveredBounds, List.mem_singleton] at member
      have : 0 ≤ cap * (p.expansionT * (p.b - 1)) := Nat.zero_le _
      omega
  | step :: rest, depth, admissible, bound, member => by
      simp only [recoveredBounds, List.mem_cons] at member
      rcases member with first | later
      · subst first
        have atMost := recoveredBound_le p finalAtMostB (step :: rest) depth admissible
        have shrink : (cap - depth) * (p.expansionT * (p.b - 1)) ≤
            cap * (p.expansionT * (p.b - 1)) :=
          Nat.mul_le_mul_right _ (Nat.sub_le cap depth)
        omega
      · cases step with
        | light =>
            exact recoveredBounds_le p finalAtMostB rest (depth + 1) admissible.2 bound later
        | refresh =>
            exact recoveredBounds_le p finalAtMostB rest 0 admissible bound later

/-- SuperNeo v1.2 Section 8's MSIS norm `8 · T · B` with `B` replaced by the largest recovered
bound of a run with this cap. -/
def msisChainBound (p : GlobalParams) (cap : Nat) : Nat :=
  8 * p.expansionT * (p.bigB + cap * (p.expansionT * (p.b - 1)))

theorem msisChainBound_zero (p : GlobalParams) : msisChainBound p 0 = p.msisNormBound := by
  simp [msisChainBound, GlobalParams.msisNormBound, Nat.mul_assoc]

/-- The production cap: 64 light folds per refresh, as in Cyclo's Table 2. -/
def productionLightFoldCap : Nat := 64

theorem production_cap_fits : 1 + productionLightFoldCap ≤ productionGlobalParams.maxFresh := by
  decide

theorem production_msisChainBound :
    msisChainBound productionGlobalParams productionLightFoldCap = 137134080 := by
  decide

end NightstreamFPrime.Spec.Folding.Schedule
