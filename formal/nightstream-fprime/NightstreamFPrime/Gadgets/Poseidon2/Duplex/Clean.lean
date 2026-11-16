import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Gadgets.Poseidon2.Clean

/-! The duplex's rows for the exporter's blocks (SPEC.md section 4.4). Each absorption places
the permutations of `Hash.compileAbsorptions` from the current state; a read places none. When the
initial state and the absorbed words are affine and live before the duplex, and its locals outside
its marks' inner locals are live, its marks have affine live lanes, and its state, samples and
assertion rows read only live columns (`marks_facts`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2

variable {live : Nat → Prop}

/-- The words that the actions absorb. -/
def absorbedWords : List Action → List Expr
  | [] => []
  | .absorb input :: actions => input ++ absorbedWords actions
  | .readK _ _ :: actions => absorbedWords actions

/-- The words that the actions' reads expect. -/
def expectedWords : List Action → List Expr
  | [] => []
  | .absorb _ :: actions => expectedWords actions
  | .readK _ expected :: actions => [expected.c0, expected.c1] ++ expectedWords actions

theorem recipeCount_absorb (input : List Expr) (actions : List Action) :
    recipeCount (.absorb input :: actions) =
      (Hash.inputChunks input).length * 1096 + recipeCount actions := by
  simp [recipeCount, Action.recipeCount]

theorem recipeCount_readK (pair : Read.Pair) (expected : KExpr) (actions : List Action) :
    recipeCount (.readK pair expected :: actions) = recipeCount actions := by
  simp [recipeCount, Action.recipeCount]

theorem marks_placed :
    ∀ (actions : List Action) (start : Nat) (state : EState),
      (∀ mark ∈ marks start state actions,
        start ≤ mark.start ∧ mark.start + 1096 ≤ start + recipeCount actions) ∧
      (marks start state actions).Pairwise
        fun first second => first.start + 1096 ≤ second.start
  | [], _, _ => ⟨fun _ member => by simp only [marks, List.not_mem_nil] at member,
      List.Pairwise.nil⟩
  | .absorb input :: actions, start, state => by
      obtain ⟨inside, ascend⟩ := marks_placed actions
        (start + (Hash.inputChunks input).length * 1096)
        (Hash.absorbedState start state (Hash.inputChunks input))
      simp only [marks, recipeCount_absorb, List.mem_append, List.pairwise_append]
      refine ⟨fun mark member => ?_, absorptionMarks_ascend _ _ _, ascend,
        fun early earlyMember late lateMember => ?_⟩
      · rcases member with member | member
        · have early := absorptionMarks_inside start state _ mark member
          exact ⟨early.1, Nat.le_trans early.2 (Nat.add_le_add_left (Nat.le_add_right _ _) _)⟩
        · have late := inside mark member
          exact ⟨Nat.le_trans (Nat.le_add_right _ _) late.1, by rw [← Nat.add_assoc]; exact late.2⟩
      · have early := absorptionMarks_inside start state _ early earlyMember
        have late := inside late lateMember
        exact Nat.le_trans early.2 late.1
  | .readK pair expected :: actions, start, state => by
      simpa [marks, recipeCount_readK] using marks_placed actions start state

/-- A state whose lanes are affine and read only live columns before `bound`. -/
def StateFacts (live : Nat → Prop) (bound : Nat) (state : EState) : Prop :=
  ∀ lane, (state lane).Affine ∧ (state lane).VarsSatisfy fun column => column < bound ∧ live column

theorem StateFacts.mono {bound larger : Nat} {state : EState} (facts : StateFacts live bound state)
    (le : bound ≤ larger) : StateFacts live larger state :=
  fun lane => ⟨(facts lane).1, (facts lane).2.mono _ fun _ inside => ⟨by omega, inside.2⟩⟩

/-- The duplex's marks have affine live lanes, and its assertion rows, samples and output state
read only live columns. -/
theorem marks_facts :
    ∀ (actions : List Action) (start : Nat) (state : EState),
      StateFacts live start state →
      (∀ word ∈ absorbedWords actions, word.Affine) →
      (∀ word ∈ absorbedWords actions,
        word.VarsSatisfy fun column => column < start ∧ live column) →
      (∀ word ∈ expectedWords actions, word.VarsSatisfy live) →
      (∀ column, start ≤ column → column < start + recipeCount actions →
        ¬ InnerOf (marks start state actions) column → live column) →
      (∀ mark ∈ marks start state actions, mark.Lanes live) ∧
        (∀ row ∈ (compile start state actions).assertions, row.VarsSatisfy live) ∧
        (∀ sample ∈ (compile start state actions).samples,
          sample.c0.VarsSatisfy live ∧ sample.c1.VarsSatisfy live) ∧
        StateFacts live (start + recipeCount actions) (compile start state actions).output
  | [], start, state, stateFacts, _, _, _, _ => by
      refine ⟨fun _ member => by simp only [marks, List.not_mem_nil] at member,
        fun _ member => by simp only [compile, List.not_mem_nil] at member,
        fun _ member => by simp only [compile, List.not_mem_nil] at member, ?_⟩
      simpa [compile, recipeCount] using stateFacts
  | .absorb input :: actions, start, state, stateFacts, wordsAffine, wordsRead, expected, own => by
      have chunkWord : ∀ chunk ∈ (Hash.inputChunks input), ∀ word ∈ chunk, word ∈ input := by
        intro chunk member word wordMember
        simp only [Hash.inputChunks, List.mem_map] at member
        obtain ⟨_, _, rfl⟩ := member
        exact List.mem_of_mem_drop (List.mem_of_mem_take wordMember)
      have inputWord : ∀ word ∈ input, word ∈ absorbedWords (.absorb input :: actions) :=
        fun word member => List.mem_append_left _ member
      have restWord : ∀ word ∈ absorbedWords actions,
          word ∈ absorbedWords (.absorb input :: actions) :=
        fun word member => List.mem_append_right _ member
      have tailPlaced := marks_placed actions (start + (Hash.inputChunks input).length * 1096)
        (Hash.absorbedState start state (Hash.inputChunks input))
      have absorptionLive : ∀ column, start ≤ column → column < start + (Hash.inputChunks input).length * 1096 →
          (∀ mark ∈ Hash.absorptionMarks start state (Hash.inputChunks input), ¬ mark.Inner column) →
          live column := by
        intro column lower upper outside
        refine own column lower (by rw [recipeCount_absorb]; exact le_add_add upper) ?_
        rintro ⟨mark, member, inner⟩
        simp only [marks, List.mem_append] at member
        rcases member with member | member
        · exact outside mark member inner
        · have tailInside := tailPlaced.1 mark member
          exact Nat.lt_irrefl _ (Nat.lt_of_lt_of_le upper (Nat.le_trans tailInside.1 inner.1))
      have outputs : ∀ lane : Fin 16, ∀ index, index < (Hash.inputChunks input).length →
          live (start + index * 1096 + 1080 + lane.val) := by
        intro lane index below
        refine absorptionLive _ ((Nat.le_add_right _ _).trans ((Nat.le_add_right _ _).trans
          (Nat.le_add_right _ _))) (block_output_lt (succ_block_le below) lane.isLt)
          fun mark member inner => ?_
        obtain ⟨other, _, starts⟩ :=
          absorptionMarks_starts start state (Hash.inputChunks input) mark member
        obtain ⟨lower, upper⟩ := inner
        rw [starts] at lower upper
        rcases Nat.lt_or_ge index other with smaller | larger
        · exact block_output_after (succ_block_le smaller) lane.isLt lower
        · exact block_output_before (Nat.mul_le_mul_right 1096 larger) upper
      have next := absorbedState_facts start state (Hash.inputChunks input) (fun lane => (stateFacts lane).1)
        (fun lane => (stateFacts lane).2) outputs
      obtain ⟨tailLanes, tailRows, tailSamples, tailOutput⟩ := marks_facts actions
        (start + (Hash.inputChunks input).length * 1096) (Hash.absorbedState start state (Hash.inputChunks input)) next
        (fun word member => wordsAffine word (restWord word member))
        (fun word member => (wordsRead word (restWord word member)).mono word
          fun column inside => ⟨Nat.lt_of_lt_of_le inside.1 (Nat.le_add_right _ _), inside.2⟩)
        (fun word member => expected word member)
        (fun column lower upper outside => own column (Nat.le_trans (Nat.le_add_right _ _) lower)
          (by rw [recipeCount_absorb, ← Nat.add_assoc]; exact upper) fun ⟨mark, member, inner⟩ => by
            simp only [marks, List.mem_append] at member
            rcases member with member | member
            · have early := absorptionMarks_inside start state (Hash.inputChunks input) mark member
              exact inner_before early.2 inner.2 lower
            · exact outside ⟨mark, member, inner⟩)
      refine ⟨fun mark member => ?_, ?_, ?_, ?_⟩
      · simp only [marks, List.mem_append] at member
        rcases member with member | member
        · exact absorptionMarks_lanes start state (Hash.inputChunks input) (fun lane => (stateFacts lane).1)
            (fun lane => (stateFacts lane).2)
            (fun chunk chunkMember word wordMember =>
              wordsAffine word (inputWord word (chunkWord chunk chunkMember word wordMember)))
            (fun chunk chunkMember word wordMember =>
              wordsRead word (inputWord word (chunkWord chunk chunkMember word wordMember)))
            absorptionLive mark member
        · exact tailLanes mark member
      · simp only [compile, Hash.compileAbsorptions_recipes_length,
          Hash.compileAbsorptions_output]
        exact tailRows
      · simp only [compile, Hash.compileAbsorptions_recipes_length,
          Hash.compileAbsorptions_output]
        exact tailSamples
      · simp only [compile, Hash.compileAbsorptions_recipes_length,
          Hash.compileAbsorptions_output, recipeCount_absorb]
        exact tailOutput.mono (Nat.add_assoc _ _ _).le
  | .readK pair expectedSample :: actions, start, state, stateFacts, wordsAffine, wordsRead,
      expected, own => by
      obtain ⟨tailLanes, tailRows, tailSamples, tailOutput⟩ := marks_facts actions start state
        stateFacts wordsAffine wordsRead
        (fun word member => expected word (List.mem_append_right _ member))
        (fun column lower upper outside => own column lower
          (by rw [recipeCount_readK]; exact upper) outside)
      have sampleLive : (Read.sample state pair).c0.VarsSatisfy live ∧
          (Read.sample state pair).c1.VarsSatisfy live :=
        ⟨((stateFacts _).2).mono _ fun _ inside => inside.2,
          ((stateFacts _).2).mono _ fun _ inside => inside.2⟩
      refine ⟨fun mark member => tailLanes mark member, fun row member => ?_,
        fun sample member => ?_, by simpa [compile, recipeCount_readK] using tailOutput⟩
      · simp only [compile, List.mem_append, KExpr.equalities, List.mem_cons,
          List.not_mem_nil, or_false] at member
        rcases member with (rfl | rfl) | member
        · exact Expr.VarsSatisfy.sub _ _ _ sampleLive.1
            (expected _ (List.mem_cons_self))
        · exact Expr.VarsSatisfy.sub _ _ _ sampleLive.2
            (expected _ (List.mem_cons_of_mem _ List.mem_cons_self))
        · exact tailRows row member
      · simp only [compile, List.mem_cons] at member
        rcases member with rfl | member
        · exact sampleLive
        · exact tailSamples sample member

/-! ### A duplex batch alone -/

/-- The operations of a duplex that places only its batch and its marks. -/
def batchOps (offset : Nat) (state : EState) (actions : List Action) : List Op :=
  [Op.witness (WitnessBatch.arithmetic offset (compile offset state actions).recipes),
    Op.mark (marks offset state actions)]

theorem batchOps_localLength (offset : Nat) (state : EState) (actions : List Action) :
    localLength (batchOps offset state actions) = recipeCount actions :=
  (localLength_batch offset _ _).trans (compile_recipes_length _ _ _)

theorem batch_placed (offset : Nat) (state : EState) (actions : List Action) :
    Placed offset (localLength (batchOps offset state actions)) (batchOps offset state actions) := by
  rw [batchOps_localLength]
  have counted := marks_placed actions offset state
  refine ⟨fun mark member => ?_, ?_⟩
  · simp only [batchOps, Circuit.marks, List.flatMap_cons, List.flatMap_nil, Op.marks,
      List.nil_append, List.append_nil] at member
    exact counted.1 mark member
  · simpa [batchOps, Circuit.marks, Op.marks] using counted.2

/-- A duplex batch's facts: clean rows, and live samples and output state. -/
theorem batch_clean (offset : Nat) (state : EState) (actions : List Action)
    (stateFacts : StateFacts live offset state)
    (wordsAffine : ∀ word ∈ absorbedWords actions, word.Affine)
    (wordsRead : ∀ word ∈ absorbedWords actions,
      word.VarsSatisfy fun column => column < offset ∧ live column)
    (expected : ∀ word ∈ expectedWords actions, word.VarsSatisfy live)
    (own : OwnLive live offset (localLength (batchOps offset state actions))
      (batchOps offset state actions)) :
    Clean live (batchOps offset state actions) ∧
      (∀ sample ∈ (compile offset state actions).samples,
        sample.c0.VarsSatisfy live ∧ sample.c1.VarsSatisfy live) ∧
      StateFacts live (offset + recipeCount actions) (compile offset state actions).output := by
  rw [batchOps_localLength] at own
  obtain ⟨lanes, _, samples, output⟩ := marks_facts actions offset state stateFacts wordsAffine
    wordsRead expected fun column lower upper outside => own column lower upper (by
      simpa [batchOps, Circuit.marks, Op.marks] using outside)
  exact ⟨.batch (compile_windows _ _ _) lanes, samples, output⟩

/-! ### The owned duplex -/

namespace Owned

theorem opsAt_split (interface : Interface) (offset : Nat) :
    opsAt interface offset =
      [Op.witness (WitnessBatch.arithmetic offset (program interface offset).recipes),
        Op.mark (marks offset (interface.initial offset) (interface.actions offset))] ++
        (allAssertions interface offset).map Op.assertZero :=
  rfl

theorem placed (interface : Interface) (offset : Nat) :
    Placed offset (localLength (Circuit.ops (main interface) offset))
      (Circuit.ops (main interface) offset) := by
  rw [main_ops, opsAt_localLength, opsAt_split, ← Nat.add_zero (List.length _)]
  have counted := marks_placed (interface.actions offset) offset (interface.initial offset)
  refine (Placed.batch (fun mark member => ?_) counted.2).append
    (.of_marks_nil (RowsOnly.map _).marks_nil)
  have := counted.1 mark member
  simpa [program, compile_recipes_length] using this

/-- The owned duplex's facts: clean rows, and live samples and output state. -/
theorem clean (interface : Interface) (offset : Nat)
    (stateFacts : StateFacts live offset (interface.initial offset))
    (wordsAffine : ∀ word ∈ absorbedWords (interface.actions offset), word.Affine)
    (wordsRead : ∀ word ∈ absorbedWords (interface.actions offset),
      word.VarsSatisfy fun column => column < offset ∧ live column)
    (expected : ∀ word ∈ expectedWords (interface.actions offset), word.VarsSatisfy live)
    (own : OwnLive live offset (localLength (Circuit.ops (main interface) offset))
      (Circuit.ops (main interface) offset)) :
    Clean live (Circuit.ops (main interface) offset) ∧
      (∀ sample ∈ (program interface offset).samples,
        sample.c0.VarsSatisfy live ∧ sample.c1.VarsSatisfy live) ∧
      StateFacts live (offset + (program interface offset).recipes.length)
        (program interface offset).output := by
  rw [main_ops, opsAt_localLength, opsAt_split] at own
  have rowsOwn : ∀ column, offset ≤ column →
      column < offset + recipeCount (interface.actions offset) →
      ¬ InnerOf (marks offset (interface.initial offset) (interface.actions offset)) column →
      live column := by
    intro column lower upper outside
    refine own column lower (by simpa [program, compile_recipes_length] using upper) ?_
    simpa [(RowsOnly.map (allAssertions interface offset)).marks_nil, Op.marks] using outside
  obtain ⟨lanes, rows, samples, output⟩ := marks_facts (interface.actions offset) offset
    (interface.initial offset) stateFacts wordsAffine wordsRead expected rowsOwn
  refine ⟨?_, samples, by simpa [program, compile_recipes_length] using output⟩
  rw [main_ops, opsAt_split]
  exact (Clean.batch (compile_windows _ _ _) lanes).append
    (.of_rows (RowsOnly.map _) fun row member =>
      rows row (by simpa [allAssertions, program] using member))

end Owned

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
