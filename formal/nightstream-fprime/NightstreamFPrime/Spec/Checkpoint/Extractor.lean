import NightstreamFPrime.Spec.Checkpoint.Retry
import NightstreamFPrime.Spec.Checkpoint.Structure

/-! The extractor of Akita Theorem 10.17 (ePrint 2026/1983, Appendix E.2, "Procedures"), with a
limit on the attempts of each retry loop, and its invariants (Appendix E.2, "Invariants").

One node `E_r` is a function of the oracle table and of a tape of fresh samples. At an indexed
stage the node draws one slot of the answer again and keeps the other slots: this is the model of
one squeeze run, in which coordinate `u` of the challenge is a fixed slot of one long output. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

variable {Addr Point Coord Slot Proof Cert Break Cand Key Witness : Type}

/-- The fresh samples of the retry loops of one node: a fresh table and a tape of the child for
each attempt. The loop of coordinate `u` of an indexed stage reads `some u`; the loop of a
checkpoint reads `none`. -/
abbrev Loops (Addr Coord Slot : Type) (budget : ℕ) (Inner : Type) : Type :=
  Option Coord → Fin budget → (Addr → Coord → Slot) × Inner

/-- The tape of the node with `depth` stages below it. -/
def Tape (Addr Coord Slot : Type) (budget : ℕ) : ℕ → Type
  | 0 => Unit
  | depth + 1 => Tape Addr Coord Slot budget depth ×
      Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth)

namespace Tape

variable {budget depth : ℕ}

def own (tape : Tape Addr Coord Slot budget (depth + 1)) : Tape Addr Coord Slot budget depth :=
  (show Tape Addr Coord Slot budget depth ×
    Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth) from tape).1

def loops (tape : Tape Addr Coord Slot budget (depth + 1)) :
    Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth) :=
  (show Tape Addr Coord Slot budget depth ×
    Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth) from tape).2

instance fintype [Fintype Addr] [DecidableEq Addr] [Fintype Coord] [DecidableEq Coord]
    [Fintype Slot] (budget : ℕ) : (depth : ℕ) → Fintype (Tape Addr Coord Slot budget depth)
  | 0 => inferInstanceAs (Fintype Unit)
  | depth + 1 =>
      letI := fintype budget depth
      inferInstanceAs (Fintype (Tape Addr Coord Slot budget depth ×
        Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth)))

instance nonempty [Nonempty Slot] (budget : ℕ) :
    (depth : ℕ) → Nonempty (Tape Addr Coord Slot budget depth)
  | 0 => ⟨()⟩
  | depth + 1 =>
      have := nonempty budget depth
      inferInstanceAs (Nonempty (Tape Addr Coord Slot budget depth ×
        Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth)))

theorem expect_succ [Fintype Addr] [DecidableEq Addr] [Fintype Coord] [DecidableEq Coord]
    [Fintype Slot]
    (value : Tape Addr Coord Slot budget depth →
      Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth) → ℝ) :
    𝔼 tape : Tape Addr Coord Slot budget (depth + 1), value tape.own tape.loops =
      𝔼 own, 𝔼 loops, value own loops := by
  have split := Finset.expect_product'
    (Finset.univ : Finset (Tape Addr Coord Slot budget depth))
    (Finset.univ : Finset (Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth)))
    value
  rw [Finset.univ_product_univ] at split
  exact split

end Tape

def breakOf : Option (Cert ⊕ Break) → Option Break
  | some (.inr broken) => some broken
  | _ => none

def certOr (default : Cert) : Option (Cert ⊕ Break) → Cert
  | some (.inl certificate) => certificate
  | _ => default

section Nodes

variable [DecidableEq Addr] [DecidableEq Point] [DecidableEq Coord]
  (structure_ : Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
  (prover : OracleComp Addr (Coord → Slot) Proof) {Inner : Type}

/-- The child node on a table, stopped after its first run of `A` when the run selects another
stage input. -/
noncomputable def guarded (level : ℕ) (anchor : Point)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (tape : Inner) : Option (Cert ⊕ Break) :=
  if structure_.protocol.input prover table level = anchor then child table tape else none

/-- One attempt at coordinate `coordinate` of the indexed stage `level`: the slot of the
coordinate at the key of `anchor` is drawn again. -/
noncomputable def indexedTry (level : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point) (coordinate : Coord)
    (sample : (Addr → Coord → Slot) × Inner) : Option (Cert ⊕ Break) :=
  guarded structure_ prover level anchor child
    (setSlot table (structure_.protocol.address anchor) coordinate
      (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2

/-- The loop of one coordinate finds no successful attempt, or finds one with the decoded value
of the reference run. -/
noncomputable def indexedFails (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point) (coordinate : Coord)
    (attempts : Fin budget → (Addr → Coord → Slot) × Inner) : Prop :=
  (firstPass (fun sample =>
      (indexedTry structure_ prover level child table anchor coordinate sample).isSome)
    budget attempts).elim True fun sample =>
      structure_.key level coordinate (sample.1 (structure_.protocol.address anchor) coordinate) =
        structure_.key level coordinate (table (structure_.protocol.address anchor) coordinate)

noncomputable def indexedChild (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point) (coordinate : Coord)
    (attempts : Fin budget → (Addr → Coord → Slot) × Inner) : Option (Cert ⊕ Break) :=
  (firstPass (fun sample =>
      (indexedTry structure_ prover level child table anchor coordinate sample).isSome)
    budget attempts).bind (indexedTry structure_ prover level child table anchor coordinate)

noncomputable def indexedCombine (level : ℕ) (coordinates : List Coord) (budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (reference : Cert)
    (loops : Loops Addr Coord Slot budget Inner) : Option (Cert ⊕ Break) :=
  if ∃ coordinate ∈ coordinates, indexedFails structure_ prover level budget child table
      (structure_.protocol.input prover table level) coordinate (loops (some coordinate)) then none
  else
    match coordinates.findSome? fun coordinate => breakOf (indexedChild structure_ prover level
        budget child table (structure_.protocol.input prover table level) coordinate
        (loops (some coordinate))) with
    | some broken => some (.inr broken)
    | none => some (structure_.combine level reference fun coordinate =>
        certOr reference (indexedChild structure_ prover level budget child table
          (structure_.protocol.input prover table level) coordinate (loops (some coordinate))))

/-- The node `E_r` of an indexed extraction stage: the reference continuation, then for each
coordinate one continuation whose answer differs from the reference in that coordinate only. -/
noncomputable def indexedNode (level : ℕ) (coordinates : List Coord) (budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (own : Inner) (loops : Loops Addr Coord Slot budget Inner) :
    Option (Cert ⊕ Break) :=
  match child table own with
  | some (.inl reference) =>
      indexedCombine structure_ prover level coordinates budget child table reference loops
  | other => other

/-- One attempt of step 3 of a checkpoint: a fresh table on `Sub(⟨anchor⟩)`. -/
noncomputable def checkpointTry (level : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point) (sample : (Addr → Coord → Slot) × Inner) :
    Option (Cert ⊕ Break) :=
  guarded structure_ prover level anchor child
    (structure_.protocol.regrow level (structure_.protocol.address anchor) table sample.1) sample.2

noncomputable def checkpointChild (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point)
    (attempts : Fin budget → (Addr → Coord → Slot) × Inner) : Option (Cert ⊕ Break) :=
  (firstPass (fun sample => (checkpointTry structure_ prover level child table anchor sample).isSome)
    budget attempts).bind (checkpointTry structure_ prover level child table anchor)

/-- Steps 2 to 4 of a checkpoint. -/
noncomputable def checkpointRetry (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (found : Cert) (loops : Loops Addr Coord Slot budget Inner) :
    Option (Cert ⊕ Break) :=
  if structure_.good level found then some (.inl found)
  else
    match checkpointChild structure_ prover level budget child table
        (structure_.protocol.input prover table level) (loops none) with
    | some (.inl other) =>
        if structure_.candidate level other = structure_.candidate level found then none
        else some (.inr (structure_.collide level found other))
    | some (.inr broken) => some (.inr broken)
    | none => none

/-- The node `E_ℓ` of a checkpoint (Appendix E.2, steps 1 to 4). -/
noncomputable def checkpointNode (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (own : Inner) (loops : Loops Addr Coord Slot budget Inner) :
    Option (Cert ⊕ Break) :=
  match child table own with
  | some (.inl found) => checkpointRetry structure_ prover level budget child table found loops
  | other => other

/-- The node `E_level` with `depth` stages below it. The node of depth 0 runs `A^J` and returns
the transcript when the verifier accepts. -/
noncomputable def node (budget : ℕ) :
    (depth : ℕ) → ℕ → (Addr → Coord → Slot) → Tape Addr Coord Slot budget depth →
      Option (Cert ⊕ Break)
  | 0, _, table, _ =>
      if structure_.protocol.accept (structure_.protocol.final prover table) then
        some (.inl (structure_.leaf (structure_.protocol.final prover table)))
      else none
  | depth + 1, level, table, tape =>
      match structure_.kind level with
      | .indexed coordinates =>
          indexedNode structure_ prover level coordinates budget (node budget depth (level + 1))
            table tape.own tape.loops
      | .checkpoint _ =>
          checkpointNode structure_ prover level budget (node budget depth (level + 1))
            table tape.own tape.loops
      | .plain => node budget depth (level + 1) table tape.own

theorem node_indexed (budget : ℕ) {depth level : ℕ} {coordinates : List Coord}
    (indexed : structure_.kind level = .indexed coordinates) (table : Addr → Coord → Slot)
    (tape : Tape Addr Coord Slot budget (depth + 1)) :
    node structure_ prover budget (depth + 1) level table tape =
      indexedNode structure_ prover level coordinates budget
        (node structure_ prover budget depth (level + 1)) table tape.own tape.loops := by
  simp only [node, indexed]

theorem node_checkpoint (budget : ℕ) {depth level span : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (table : Addr → Coord → Slot)
    (tape : Tape Addr Coord Slot budget (depth + 1)) :
    node structure_ prover budget (depth + 1) level table tape =
      checkpointNode structure_ prover level budget
        (node structure_ prover budget depth (level + 1)) table tape.own tape.loops := by
  simp only [node, checkpoint]

theorem node_plain (budget : ℕ) {depth level : ℕ} (plain : structure_.kind level = .plain)
    (table : Addr → Coord → Slot) (tape : Tape Addr Coord Slot budget (depth + 1)) :
    node structure_ prover budget (depth + 1) level table tape =
      node structure_ prover budget depth (level + 1) table tape.own := by
  simp only [node, plain]

/-- The extractor `E_1` with at most `budget` attempts in each retry loop. -/
noncomputable def extract (budget : ℕ) (table : Addr → Coord → Slot)
    (tape : Tape Addr Coord Slot budget structure_.protocol.stages) : Option (Cert ⊕ Break) :=
  node structure_ prover budget structure_.protocol.stages 0 table tape

/-- The last stage through which the output of a node records the transcript: the next indexed
stage, or the complete transcript. -/
def Structure.frontier : ℕ → ℕ → ℕ
  | 0, level => level
  | depth + 1, level =>
      match structure_.kind level with
      | .indexed _ => level
      | _ => Structure.frontier depth (level + 1)

omit [DecidableEq Addr] [DecidableEq Point] [DecidableEq Coord] in
theorem Structure.le_frontier (depth level : ℕ) : level ≤ structure_.frontier depth level := by
  induction depth generalizing level with
  | zero => exact le_rfl
  | succ depth induction =>
      unfold Structure.frontier
      split
      · exact le_rfl
      · exact (Nat.le_succ level).trans (induction (level + 1))

omit [DecidableEq Addr] [DecidableEq Point] [DecidableEq Coord] in
theorem Structure.plain_le_frontier (count : ℕ) :
    ∀ depth level, count ≤ depth →
      (∀ later, level ≤ later → later < level + count → structure_.kind later = .plain) →
      level + count ≤ structure_.frontier depth level := by
  induction count with
  | zero => exact fun depth level _ _ => structure_.le_frontier depth level
  | succ count induction =>
      intro depth level enough plain
      obtain ⟨depth, rfl⟩ : ∃ smaller, depth = smaller + 1 := ⟨depth - 1, by omega⟩
      unfold Structure.frontier
      rw [plain level le_rfl (by omega)]
      refine le_trans (by omega) (induction depth (level + 1) (by omega) fun later low high => ?_)
      exact plain later (by omega) (by omega)

/-- Invariant (2) of Appendix E.2 for an output: a certificate is in `Y_level` and its transcript
agrees with that of `A^J` through `frontier`; a tagged output is in `T`. -/
def Certified (level frontier : ℕ) (table : Addr → Coord → Slot) : Cert ⊕ Break → Prop
  | .inl certificate => structure_.holds level certificate ∧ ∀ earlier ≤ frontier,
      structure_.protocol.cut earlier (structure_.point certificate) =
        some (structure_.protocol.input prover table earlier)
  | .inr broken => structure_.breaks broken

/-- Invariants (2) and (3) of Appendix E.2 for a node. -/
def Sound (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (level frontier : ℕ) : Prop :=
  ∀ table tape out, child table tape = some out →
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Certified structure_ prover level frontier table out

omit [DecidableEq Addr] [DecidableEq Coord] in
theorem guarded_eq_some {level : ℕ} {anchor : Point}
    {child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    {table : Addr → Coord → Slot} {tape : Inner} {out : Cert ⊕ Break} :
    guarded structure_ prover level anchor child table tape = some out ↔
      structure_.protocol.input prover table level = anchor ∧ child table tape = some out := by
  unfold guarded
  split <;> simp [*]

theorem indexedChild_eq_some {level budget : ℕ}
    {child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    {table : Addr → Coord → Slot} {anchor : Point} {coordinate : Coord}
    {attempts : Fin budget → (Addr → Coord → Slot) × Inner} {out : Cert ⊕ Break}
    (found : indexedChild structure_ prover level budget child table anchor coordinate attempts =
      some out) :
    ∃ sample, firstPass (fun sample =>
        (indexedTry structure_ prover level child table anchor coordinate sample).isSome)
        budget attempts = some sample ∧
      structure_.protocol.input prover (setSlot table (structure_.protocol.address anchor)
        coordinate (sample.1 (structure_.protocol.address anchor) coordinate)) level = anchor ∧
      child (setSlot table (structure_.protocol.address anchor) coordinate
        (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2 = some out := by
  obtain ⟨sample, first, tried⟩ := Option.bind_eq_some_iff.mp found
  exact ⟨sample, first, (guarded_eq_some structure_ prover).mp tried⟩

omit [DecidableEq Coord] in
theorem checkpointChild_eq_some {level budget : ℕ}
    {child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    {table : Addr → Coord → Slot} {anchor : Point}
    {attempts : Fin budget → (Addr → Coord → Slot) × Inner} {out : Cert ⊕ Break}
    (found : checkpointChild structure_ prover level budget child table anchor attempts =
      some out) :
    ∃ sample, firstPass (fun sample =>
        (checkpointTry structure_ prover level child table anchor sample).isSome)
        budget attempts = some sample ∧
      structure_.protocol.input prover (structure_.protocol.regrow level
        (structure_.protocol.address anchor) table sample.1) level = anchor ∧
      child (structure_.protocol.regrow level (structure_.protocol.address anchor) table sample.1)
        sample.2 = some out := by
  obtain ⟨sample, first, tried⟩ := Option.bind_eq_some_iff.mp found
  exact ⟨sample, first, (guarded_eq_some structure_ prover).mp tried⟩

variable [Fintype Coord] [Fintype Slot]

theorem indexedNode_sound (lawful : structure_.Lawful) {level frontier : ℕ}
    {coordinates : List Coord} (indexed : structure_.kind level = .indexed coordinates)
    (within : level < structure_.protocol.stages) (deep : level + 1 ≤ frontier)
    {child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (sound : Sound structure_ prover child (level + 1) frontier) (budget : ℕ)
    (table : Addr → Coord → Slot) (own : Inner) (loops : Loops Addr Coord Slot budget Inner)
    (out : Cert ⊕ Break)
    (result : indexedNode structure_ prover level coordinates budget child table own loops =
      some out) :
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Certified structure_ prover level level table out := by
  unfold indexedNode at result
  rcases first : child table own with _ | (reference | broken)
  · rw [first] at result
    exact absurd result (by simp)
  · simp only [first] at result
    obtain ⟨accepts, held, agrees⟩ := sound table own _ first
    refine ⟨accepts, ?_⟩
    unfold indexedCombine at result
    split at result
    · exact absurd result (by simp)
    · rename_i passes
      split at result
      · rename_i broken found
        obtain rfl := Option.some.inj result
        obtain ⟨coordinate, _, isBreak⟩ := List.exists_of_findSome?_eq_some found
        rcases tried : indexedChild structure_ prover level budget child table
            (structure_.protocol.input prover table level) coordinate (loops (some coordinate))
          with _ | (certificate | other)
        · simp [tried, breakOf] at isBreak
        · simp [tried, breakOf] at isBreak
        · rw [tried] at isBreak
          obtain rfl : other = broken := Option.some.inj isBreak
          obtain ⟨sample, _, _, ran⟩ := indexedChild_eq_some structure_ prover tried
          exact (sound _ _ _ ran).2
      · rename_i unbroken
        obtain rfl := Option.some.inj result
        have each : ∀ coordinate ∈ coordinates,
            structure_.holds (level + 1) (certOr reference (indexedChild structure_ prover level
              budget child table (structure_.protocol.input prover table level) coordinate
              (loops (some coordinate)))) ∧
            structure_.protocol.cut level (structure_.point (certOr reference
              (indexedChild structure_ prover level budget child table
                (structure_.protocol.input prover table level) coordinate
                (loops (some coordinate))))) =
              some (structure_.protocol.input prover table level) ∧
            ∃ other slot, structure_.protocol.cut (level + 1) (structure_.point (certOr reference
                (indexedChild structure_ prover level budget child table
                  (structure_.protocol.input prover table level) coordinate
                  (loops (some coordinate))))) = some other ∧
              structure_.protocol.last other = Function.update (structure_.protocol.last
                (structure_.protocol.input prover table (level + 1))) coordinate slot ∧
              structure_.key level coordinate slot ≠ structure_.key level coordinate
                (structure_.protocol.last (structure_.protocol.input prover table (level + 1))
                  coordinate) := by
          intro coordinate inside
          have works : ¬ indexedFails structure_ prover level budget child table
              (structure_.protocol.input prover table level) coordinate (loops (some coordinate)) :=
            fun fails => passes ⟨coordinate, inside, fails⟩
          unfold indexedFails at works
          rcases first : firstPass (fun sample => (indexedTry structure_ prover level child table
              (structure_.protocol.input prover table level) coordinate sample).isSome) budget
              (loops (some coordinate)) with _ | sample
          · simp [first] at works
          · rw [first] at works
            have passed : (indexedTry structure_ prover level child table
                (structure_.protocol.input prover table level) coordinate sample).isSome :=
              firstPass_pass (pass := fun sample => (indexedTry structure_ prover level child table
                (structure_.protocol.input prover table level) coordinate sample).isSome = true)
                first
            obtain ⟨tried, ran⟩ := Option.isSome_iff_exists.mp passed
            have chosen : indexedChild structure_ prover level budget child table
                (structure_.protocol.input prover table level) coordinate (loops (some coordinate)) =
                  some tried := by
              unfold indexedChild
              rw [first]
              exact ran
            have clean := (List.findSome?_eq_none_iff.mp unbroken) coordinate inside
            rw [chosen] at clean ⊢
            rcases tried with certificate | other
            · obtain ⟨same, childRan⟩ := (guarded_eq_some structure_ prover).mp ran
              obtain ⟨_, heldChild, agreesChild⟩ := sound _ _ _ childRan
              refine ⟨heldChild, ?_, _, sample.1 (structure_.protocol.address
                (structure_.protocol.input prover table level)) coordinate,
                agreesChild (level + 1) deep, ?_, ?_⟩
              · rw [← same]
                exact agreesChild level (Nat.le_of_succ_le deep)
              · rw [structure_.protocol.last_input prover _ within,
                  structure_.protocol.last_input prover _ within, same, setSlot_point]
              · rw [structure_.protocol.last_input prover _ within]
                exact works
            · simp [breakOf] at clean
        have combined := lawful.combine_holds level coordinates reference _
          (structure_.protocol.input prover table level)
          (structure_.protocol.input prover table (level + 1)) indexed held
          (agrees level (Nat.le_of_succ_le deep)) (agrees (level + 1) deep) each
        rcases shape : structure_.combine level reference fun coordinate =>
            certOr reference (indexedChild structure_ prover level budget child table
              (structure_.protocol.input prover table level) coordinate (loops (some coordinate)))
          with certificate | broken
        · rw [shape] at combined
          exact ⟨combined.1, fun earlier below => (combined.2 earlier below).trans
            (agrees earlier (below.trans (Nat.le_of_succ_le deep)))⟩
        · rw [shape] at combined
          exact combined
  · simp only [first] at result
    obtain rfl := Option.some.inj result
    exact sound table own _ first

theorem checkpointNode_sound (lawful : structure_.Lawful) {level span frontier : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (deep : level + 1 ≤ frontier)
    {child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (sound : Sound structure_ prover child (level + 1) frontier) (budget : ℕ)
    (table : Addr → Coord → Slot) (own : Inner) (loops : Loops Addr Coord Slot budget Inner)
    (out : Cert ⊕ Break)
    (result : checkpointNode structure_ prover level budget child table own loops = some out) :
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Certified structure_ prover level frontier table out := by
  unfold checkpointNode at result
  rcases first : child table own with _ | (found | broken)
  · rw [first] at result
    exact absurd result (by simp)
  · simp only [first] at result
    obtain ⟨accepts, held, agrees⟩ := sound table own _ first
    refine ⟨accepts, ?_⟩
    unfold checkpointRetry at result
    split at result
    · rename_i isGood
      obtain rfl := Option.some.inj result
      exact ⟨lawful.good_holds level span found checkpoint held isGood, agrees⟩
    · split at result
      · rename_i other tried
        split at result
        · exact absurd result (by simp)
        · rename_i different
          obtain rfl := Option.some.inj result
          obtain ⟨sample, _, same, ran⟩ := checkpointChild_eq_some structure_ prover tried
          obtain ⟨_, heldOther, agreesOther⟩ := sound _ _ _ ran
          refine lawful.collide_breaks level span found other
            (structure_.protocol.input prover table level) checkpoint held heldOther
            (agrees level (Nat.le_of_succ_le deep)) ?_ (Ne.symm different)
          rw [← same]
          exact agreesOther level (Nat.le_of_succ_le deep)
      · rename_i broken tried
        obtain rfl := Option.some.inj result
        obtain ⟨sample, _, _, ran⟩ := checkpointChild_eq_some structure_ prover tried
        exact (sound _ _ _ ran).2
      · exact absurd result (by simp)
  · simp only [first] at result
    obtain rfl := Option.some.inj result
    exact sound table own _ first

/-- Invariants (2) and (3) of Appendix E.2: a node that returns an output runs on a table that
the verifier accepts, and the output is a certificate of `Y_level` whose transcript agrees with
that of `A^J`, or an element of `T`. -/
theorem node_sound (lawful : structure_.Lawful) (budget : ℕ) :
    ∀ depth level, level + depth = structure_.protocol.stages →
      Sound structure_ prover (node structure_ prover budget depth level) level
        (structure_.frontier depth level) := by
  intro depth
  induction depth with
  | zero =>
      intro level full table tape out result
      obtain rfl : level = structure_.protocol.stages := by omega
      unfold node at result
      split at result
      · rename_i accepts
        obtain rfl := Option.some.inj result
        refine ⟨accepts, lawful.leaf_holds _ accepts, fun earlier below => ?_⟩
        rw [lawful.leaf_point]
        exact structure_.protocol.cut_input prover table below le_rfl
      · exact absurd result (by simp)
  | succ depth induction =>
      intro level full table tape out result
      have below := induction (level + 1) (by omega)
      unfold node at result
      unfold Structure.frontier
      split at result
      · rename_i coordinates indexed
        rw [indexed]
        exact indexedNode_sound structure_ prover lawful indexed (by omega)
          (structure_.le_frontier depth (level + 1)) below budget table _ _ out result
      · rename_i span checkpoint
        rw [checkpoint]
        exact checkpointNode_sound structure_ prover lawful checkpoint
          (structure_.le_frontier depth (level + 1)) below budget table _ _ out result
      · rename_i plain
        rw [plain]
        obtain ⟨accepts, certified⟩ := below table _ out result
        refine ⟨accepts, ?_⟩
        rcases out with certificate | broken
        · exact ⟨lawful.plain_holds level certificate plain certified.1, certified.2⟩
        · exact certified

/-- The output of the extractor (Definition 10.15): a witness of the relation for the statement of
the prover (item (viii) of Definition 10.16), or a witness of `R_bind`. -/
theorem extract_sound (lawful : structure_.Lawful) (budget : ℕ) (table : Addr → Coord → Slot)
    (tape : Tape Addr Coord Slot budget structure_.protocol.stages) (out : Cert ⊕ Break)
    (result : extract structure_ prover budget table tape = some out) :
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Sum.elim (fun certificate => structure_.relation (structure_.protocol.input prover table 0)
        (structure_.witness certificate)) structure_.breaks out := by
  obtain ⟨accepts, certified⟩ := node_sound structure_ prover lawful budget
    structure_.protocol.stages 0 (Nat.zero_add _) table tape out result
  refine ⟨accepts, ?_⟩
  rcases out with certificate | broken
  · exact lawful.witness_holds certificate _ certified.1 (certified.2 0 (Nat.zero_le _))
  · exact certified

end Nodes

end NightstreamFPrime.Spec.Checkpoint
