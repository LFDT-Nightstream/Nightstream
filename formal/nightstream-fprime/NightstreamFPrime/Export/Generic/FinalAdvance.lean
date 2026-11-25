import NightstreamFPrime.Export.Generic.RelationCompleteness
import NightstreamFPrime.Export.Generic.MarkedBlocks

namespace NightstreamFPrime.Export.Generic.FinalAdvance

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.Pieces

def target (ops : List Op) : Option Block :=
  ops.findSome? fun
    | .subcircuit child =>
      if child.name = "light.pirlc.sampler" ∨ child.name = "pirlc.v1_2.sampler_chain" then
        child.marks.getLast?.map fun mark => ⟨mark.start, mark.lanes⟩
      else none
    | _ => none

private def matchesBlock (block : Block) : Piece → Bool
  | .permutation start lanes => decide (start = block.start ∧ lanes = block.lanes)
  | _ => false

private def independent (block : Block) : Piece → Bool
  | piece@(.permutation start lanes) =>
    matchesBlock block piece ||
      (decide (start + 1096 ≤ block.start ∨ block.start + 1096 ≤ start) &&
        lanes.all (avoids block.covers))
  | piece => CircuitRelation.avoidsDead block.covers piece

def witnessAvoids (block : Block) (witness : List WitnessBatch) : Bool :=
  witness.all fun batch => batch.recipes.all (avoids block.covers) &&
    batch.hints.all (fun hint => avoids block.covers hint.source)

def checked (protectedEnd : Nat) (block : Block) (pieces : List Piece)
    (witness : List WitnessBatch) : Bool :=
  decide (protectedEnd ≤ block.start) && checkFrame [block] &&
    pieces.all (independent block) && witnessAvoids block witness &&
    pieces.any (matchesBlock block)

def plan (protectedEnd : Nat) (candidate : Option Block) (pieces : List Piece)
    (witness : List WitnessBatch) : Option Block :=
  candidate.filter fun block => checked protectedEnd block pieces witness

def kept (choice : Option Block) (pieces : List Piece) : List Piece :=
  match choice with
  | none => pieces
  | some block => pieces.filter fun piece => !matchesBlock block piece

def restore (choice : Option Block) (env : Env) : Env :=
  match choice with
  | none => env
  | some block => Pilot.completePermutationInvocationEnv block.invocation env

theorem kept_sublist (choice : Option Block) (pieces : List Piece) :
    (kept choice pieces).Sublist pieces := by
  cases choice with
  | none => exact List.Sublist.refl _
  | some _ => exact List.filter_sublist

private theorem blocks_sublist {left right : List Piece} (sub : left.Sublist right) :
    (CircuitRelation.blocksOf left).Sublist (CircuitRelation.blocksOf right) := by
  induction sub with
  | slnil => exact .slnil
  | cons piece _ ih => cases piece <;> first | exact ih | exact .cons _ ih
  | cons_cons piece _ ih => cases piece <;> first | exact ih | exact .cons_cons _ ih

private theorem avoids_mono {small large : Nat → Bool}
    (subset : ∀ column, large column = false → small column = false) (expression : Expr)
    (original : avoids large expression = true) : avoids small expression = true :=
  (MarkedBlocks.avoids_iff _ _).mpr
    (((MarkedBlocks.avoids_iff _ _).mp original).mono expression subset)

theorem kept_packable {publicColumns : Nat} (choice : Option Block) {pieces : List Piece}
    (packed : CircuitRelation.packable publicColumns pieces = true) :
    CircuitRelation.packable publicColumns (kept choice pieces) = true := by
  have sub := kept_sublist choice pieces
  have blocks := blocks_sublist sub
  have live (column : Nat) (old : deadIn (CircuitRelation.blocksOf pieces) column = false) :
      deadIn (CircuitRelation.blocksOf (kept choice pieces)) column = false := by
    simp only [deadIn, List.any_eq_false] at old ⊢
    exact fun block member => old _ (blocks.subset member)
  simp only [CircuitRelation.packable, checkFrame, CircuitRelation.startsFrom, Bool.and_eq_true,
    List.all_eq_true] at packed ⊢
  refine ⟨⟨⟨MarkedBlocks.ordered_of_pairwise _
    (List.Pairwise.sublist blocks (ordered_pairwise _ packed.1.1.1)), ?_⟩, ?_⟩, ?_⟩
  · intro block member lane inLanes
    obtain ⟨linear, avoidsOld⟩ := packed.1.1.2 block (blocks.subset member) lane inLanes
    refine ⟨linear, avoids_mono ?_ lane avoidsOld⟩
    intro column outside
    simp only [Bool.or_eq_false_iff] at outside ⊢
    exact ⟨outside.1, live _ outside.2⟩
  · intro block member
    exact packed.1.2 _ (blocks.subset member)
  · intro piece member
    have old := packed.2 piece (sub.subset member)
    cases piece with
    | permutation _ _ => rfl
    | constraint expression => exact avoids_mono live expression old
    | ringProduct cells start lanes =>
      simp only [CircuitRelation.avoidsDead, Bool.and_eq_true, List.all_eq_true,
        Bool.not_eq_true'] at old ⊢
      exact ⟨fun lane inLanes => avoids_mono live lane (old.1 _ inLanes),
        fun index inside => live _ (old.2 _ inside)⟩

private theorem checked_of_plan {protectedEnd : Nat} {candidate : Option Block}
    {pieces : List Piece} {witness : List WitnessBatch} {block : Block}
    (selected : plan protectedEnd candidate pieces witness = some block) :
    checked protectedEnd block pieces witness = true := by
  cases candidate with
  | none => simp [plan] at selected
  | some _ =>
    simp [plan] at selected
    exact selected.2

private theorem other_holds (block : Block) (before after : Env)
    (agree : ∀ column, block.covers column = false → after column = before column)
    (piece : Piece) (different : matchesBlock block piece = false)
    (apart : independent block piece = true)
    (holds : ConstraintsHold before (piece.constraints (PilotData.canonicalConstraints ()))) :
    ConstraintsHold after (piece.constraints (PilotData.canonicalConstraints ())) := by
  cases piece with
  | constraint expression =>
    exact CircuitRelation.avoidsDead_congr agree _ _ rfl apart holds
  | ringProduct cells start lanes =>
    exact CircuitRelation.avoidsDead_congr agree _ _ rfl apart holds
  | permutation start lanes =>
    simp only [independent, different, Bool.false_or, Bool.and_eq_true, decide_eq_true_eq,
      List.all_eq_true] at apart
    intro current member
    simp only [Piece.constraints, List.mem_map] at member
    obtain ⟨pattern, patternMember, rfl⟩ := member
    have wires : ∀ column, column < 1112 →
        avoids block.covers (laneSubst 16 start lanes column) = true := by
      intro column below
      by_cases small : column < 16
      · rw [laneSubst_input lanes small]
        by_cases inside : column < lanes.length
        · rw [List.getD_eq_getElem _ _ inside]
          exact apart.2 _ (List.getElem_mem _)
        · rw [List.getD_eq_default _ _ (by omega)]
          rfl
      · simp only [laneSubst, small, ite_false, avoids, Block.covers, Bool.not_eq_true',
          Bool.and_eq_false_iff, decide_eq_false_iff_not]
        rcases apart.1 with earlier | later
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
    rw [eval_congr agree _ (avoids_subst _ wires _ (RelationCompleteness.canonical_varsBelow _ patternMember))]
    exact holds _ (List.mem_map_of_mem patternMember)

theorem complete (choice : Option Block) (pieces : List Piece) (env : Env)
    (holds : ∀ piece ∈ pieces, ConstraintsHold env
      (piece.constraints (PilotData.canonicalConstraints ()))) :
    ∀ piece ∈ kept choice pieces, ConstraintsHold env
      (piece.constraints (PilotData.canonicalConstraints ())) :=
  fun piece member => holds piece ((kept_sublist choice pieces).subset member)

theorem sound (protectedEnd : Nat) (candidate : Option Block) (pieces : List Piece)
    (witness : List WitnessBatch) (env : Env)
    (holds : ∀ piece ∈ kept (plan protectedEnd candidate pieces witness) pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    ∀ piece ∈ pieces, ConstraintsHold (restore (plan protectedEnd candidate pieces witness) env)
      (piece.constraints (PilotData.canonicalConstraints ())) := by
  cases selected : plan protectedEnd candidate pieces witness with
  | none => simpa [selected, kept, restore] using holds
  | some block =>
    have good := checked_of_plan selected
    simp only [checked, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at good
    have frame := checkFrame_spec good.1.1.1.2
    obtain ⟨agreement, filled⟩ := completed_holds frame (List.mem_singleton_self block) env
    intro piece member
    by_cases same : matchesBlock block piece = true
    · cases piece with
      | constraint _ | ringProduct _ _ _ => simp [matchesBlock] at same
      | permutation start lanes =>
        simp only [matchesBlock, decide_eq_true_eq] at same
        rcases same with ⟨rfl, rfl⟩
        exact permutation_sound 0 0 _ _ (frame.linear _ (List.mem_singleton_self block)) _ filled
    · have agree : ∀ column, block.covers column = false →
            restore (some block) env column = env column := by
        intro column outside
        simp only [Block.covers, Bool.and_eq_false_iff, decide_eq_false_iff_not] at outside
        exact agreement column (by omega)
      have prior : piece ∈ kept (plan protectedEnd candidate pieces witness) pieces := by
        rw [selected]
        exact List.mem_filter.mpr ⟨member, by simp [same]⟩
      exact other_holds block env _ agree piece (Bool.eq_false_iff.mpr same)
        (good.1.1.2 _ member) (holds _ prior)

theorem agrees_prefix (protectedEnd : Nat) (candidate : Option Block) (pieces : List Piece)
    (witness : List WitnessBatch) (env : Env) {column : Nat} (below : column < protectedEnd) :
    restore (plan protectedEnd candidate pieces witness) env column = env column := by
  cases selected : plan protectedEnd candidate pieces witness with
  | none => rfl
  | some block =>
    have good := checked_of_plan selected
    simp only [checked, Bool.and_eq_true, decide_eq_true_eq] at good
    exact (completed_holds (checkFrame_spec good.1.1.1.2) (List.mem_singleton_self block) env).1
      column (Or.inl (Nat.lt_of_lt_of_le below good.1.1.1.1))

end NightstreamFPrime.Export.Generic.FinalAdvance
