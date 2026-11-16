import NightstreamFPrime.Gadgets.Poseidon2.HashMarks

/-! A gadget's rows for the exporter's blocks (SPEC.md section 4.4). The exporter packs each marked
permutation and gives its 1,080 inner locals no column, so no other row may read them. A gadget
states what this needs of its operations at `offset` with `count` locals:

* `Placed`: its marks lie in its locals and ascend, 1,096 locals apart;
* `LiveRows live`: its rows are free rows that read only `live` columns and its marks' rows, in
  order, and each mark has 16 affine lanes that read only `live` columns before it.

A gadget proves `LiveRows live` for every `live` that its inputs read and that holds on its locals
outside its marks' inner locals (`OwnLive`). Gadgets compose: appended operations keep both
(`Placed.append`, `LiveRows.append`), and each part's locals inherit `OwnLive` because the other part's
marks lie in the other part's locals (`OwnLive.left`, `OwnLive.right`). At the step circuit,
`live` is the set of columns that are no inner local of a mark. -/

set_option autoImplicit false

namespace NightstreamFPrime.Gadgets.Poseidon2

open NightstreamFPrime.Circuit

/-- The operations' marks lie in their locals from `offset` and ascend. -/
structure Placed (offset count : Nat) (ops : List Op) : Prop where
  inside : ∀ mark ∈ marks ops, offset ≤ mark.start ∧ mark.start + 1096 ≤ offset + count
  ascend : (marks ops).Pairwise fun first second => first.start + 1096 ≤ second.start

/-- The operations' locals from `offset` that are no inner local of their marks are live. -/
def OwnLive (live : Nat → Prop) (offset count : Nat) (ops : List Op) : Prop :=
  ∀ column, offset ≤ column → column < offset + count → ¬ InnerOf (marks ops) column →
    live column

/-- The operations' rows are free rows that read only `live` columns and their marks' rows, in
order, and each mark has 16 affine lanes that read only `live` columns before it. -/
structure LiveRows (live : Nat → Prop) (ops : List Op) : Prop where
  segmented : Segmented live Permutation.markWindow (marks ops) (flatConstraints ops)
  lanes : ∀ mark ∈ marks ops, mark.Lanes live

variable {live : Nat → Prop}

/-! ### Arithmetic of 1,096-local permutations

Each fact states one step of the placement arithmetic, so that proofs do not hand `omega` goals
with nested sums of multiples of 1,096. -/

theorem succ_block_le {index count : Nat} (below : index < count) :
    index * 1096 + 1096 ≤ count * 1096 := by
  have := Nat.mul_le_mul_right 1096 (Nat.succ_le_of_lt below)
  rwa [Nat.succ_mul] at this

theorem block_output_lt {start before after lane : Nat} (room : before + 1096 ≤ after)
    (small : lane < 16) : start + before + 1080 + lane < start + after := by
  omega

theorem block_output_after {start before after lane : Nat} (room : before + 1096 ≤ after)
    (small : lane < 16) (lower : start + after ≤ start + before + 1080 + lane) : False := by
  omega

theorem block_output_before {start before after lane : Nat} (order : after ≤ before)
    (upper : start + before + 1080 + lane < start + after + 1080) : False := by
  omega

theorem inner_before {first column bound : Nat} (room : first + 1096 ≤ bound)
    (inner : column < first + 1080) (lower : bound ≤ column) : False := by
  omega

theorem le_add_add {column start count rest : Nat} (upper : column < start + count) :
    column < start + (count + rest) :=
  Nat.lt_of_lt_of_le upper (Nat.add_le_add_left (Nat.le_add_right _ _) _)

/-! ### Operations without marks -/

theorem Placed.of_marks_nil {offset count : Nat} {ops : List Op} (none : marks ops = []) :
    Placed offset count ops :=
  ⟨by simp [none], by simp [none]⟩

theorem LiveRows.of_free {ops : List Op} (none : marks ops = [])
    (free : ∀ row ∈ flatConstraints ops, row.VarsSatisfy live) : LiveRows live ops :=
  ⟨none ▸ Segmented.ofFree free, by simp [none]⟩

theorem OwnLive.of_marks_nil {offset count : Nat} {ops : List Op} (none : marks ops = [])
    (own : OwnLive live offset count ops) :
    ∀ column, offset ≤ column → column < offset + count → live column :=
  fun column lower upper => own column lower upper (by simp [none, InnerOf])

theorem LiveRows.free {ops : List Op} (liveRows : LiveRows live ops) (none : marks ops = []) :
    ∀ row ∈ flatConstraints ops, row.VarsSatisfy live := by
  have segmented := liveRows.segmented
  rw [none] at segmented
  exact segmented.free_of_nil

/-! ### Appended operations -/

theorem Placed.append {offset firstCount secondCount : Nat} {first second : List Op}
    (firstPlaced : Placed offset firstCount first)
    (secondPlaced : Placed (offset + firstCount) secondCount second) :
    Placed offset (firstCount + secondCount) (first ++ second) := by
  refine ⟨fun mark member => ?_, ?_⟩
  · rw [marks_append, List.mem_append] at member
    rcases member with member | member
    · have := firstPlaced.inside mark member
      omega
    · have := secondPlaced.inside mark member
      omega
  · rw [marks_append, List.pairwise_append]
    refine ⟨firstPlaced.ascend, secondPlaced.ascend, fun early earlyMember late lateMember => ?_⟩
    have := firstPlaced.inside early earlyMember
    have := secondPlaced.inside late lateMember
    omega

theorem LiveRows.append {first second : List Op} (firstLiveRows : LiveRows live first)
    (secondLiveRows : LiveRows live second) : LiveRows live (first ++ second) := by
  refine ⟨?_, ?_⟩
  · rw [marks_append, flatConstraints_append]
    exact firstLiveRows.segmented.append secondLiveRows.segmented
  · intro mark member
    rw [marks_append, List.mem_append] at member
    rcases member with member | member
    · exact firstLiveRows.lanes mark member
    · exact secondLiveRows.lanes mark member

theorem OwnLive.left {offset firstCount secondCount : Nat} {first second : List Op}
    (own : OwnLive live offset (firstCount + secondCount) (first ++ second))
    (secondPlaced : Placed (offset + firstCount) secondCount second) :
    OwnLive live offset firstCount first := by
  intro column lower upper outside
  refine own column lower (by omega) fun ⟨mark, member, inner⟩ => ?_
  rw [marks_append, List.mem_append] at member
  rcases member with member | member
  · exact outside ⟨mark, member, inner⟩
  · have := secondPlaced.inside mark member
    have := inner.1
    omega

theorem OwnLive.right {offset firstCount secondCount : Nat} {first second : List Op}
    (own : OwnLive live offset (firstCount + secondCount) (first ++ second))
    (firstPlaced : Placed offset firstCount first) :
    OwnLive live (offset + firstCount) secondCount second := by
  intro column lower upper outside
  refine own column (by omega) (by omega) fun ⟨mark, member, inner⟩ => ?_
  rw [marks_append, List.mem_append] at member
  rcases member with member | member
  · have := firstPlaced.inside mark member
    have := inner.2
    omega
  · exact outside ⟨mark, member, inner⟩

theorem OwnLive.mono {offset count : Nat} {ops other : List Op}
    (own : OwnLive live offset count ops) (same : marks other = marks ops) :
    OwnLive live offset count other := by
  intro column lower upper outside
  exact own column lower upper (same ▸ outside)

/-! ### Children -/

@[simp] theorem marks_subcircuit (circuit : FormalCircuit) (name : String) (offset : Nat) :
    marks [Op.subcircuit (circuit.asSubcircuit name offset)] =
      marks (Circuit.ops circuit.main offset) := by
  simp [marks, Op.marks]

@[simp] theorem flatConstraints_subcircuit (circuit : FormalCircuit) (name : String)
    (offset : Nat) :
    flatConstraints [Op.subcircuit (circuit.asSubcircuit name offset)] =
      flatConstraints (Circuit.ops circuit.main offset) := by
  simp [flatConstraints, Op.flatConstraints]

@[simp] theorem localLength_subcircuit (circuit : FormalCircuit) (name : String)
    (offset : Nat) :
    localLength [Op.subcircuit (circuit.asSubcircuit name offset)] =
      localLength (Circuit.ops circuit.main offset) := by
  simp [localLength, Op.localLength]

theorem Placed.subcircuit {circuit : FormalCircuit} {name : String} {offset count : Nat}
    (placed : Placed offset count (Circuit.ops circuit.main offset)) :
    Placed offset count [Op.subcircuit (circuit.asSubcircuit name offset)] := by
  refine ⟨fun mark member => ?_, ?_⟩
  · rw [marks_subcircuit] at member
    exact placed.inside mark member
  · rw [marks_subcircuit]
    exact placed.ascend

theorem LiveRows.subcircuit {circuit : FormalCircuit} {name : String} {offset : Nat}
    (liveRows : LiveRows live (Circuit.ops circuit.main offset)) :
    LiveRows live [Op.subcircuit (circuit.asSubcircuit name offset)] := by
  refine ⟨?_, ?_⟩
  · rw [marks_subcircuit, flatConstraints_subcircuit]
    exact liveRows.segmented
  · rw [marks_subcircuit]
    exact liveRows.lanes

theorem OwnLive.subcircuit {circuit : FormalCircuit} {name : String} {offset count : Nat}
    (own : OwnLive live offset count [Op.subcircuit (circuit.asSubcircuit name offset)]) :
    OwnLive live offset count (Circuit.ops circuit.main offset) :=
  own.mono (marks_subcircuit circuit name offset).symm

/-! ### Rows without locals -/

theorem Placed.assertions {offset : Nat} (rows : List Expr) :
    Placed offset 0 (rows.map Op.assertZero) :=
  .of_marks_nil (by simp [marks, Op.marks])

theorem LiveRows.assertions {rows : List Expr} (free : ∀ row ∈ rows, row.VarsSatisfy live) :
    LiveRows live (rows.map Op.assertZero) := by
  refine .of_free (by simp [marks, Op.marks]) fun row member => ?_
  simp only [flatConstraints, List.flatMap_map, Op.flatConstraints, List.flatMap_singleton']
    at member
  exact free row (by simpa using member)

/-- Operations that are rows only: no marks, no locals. -/
def RowsOnly (ops : List Op) : Prop := ∀ op ∈ ops, ∃ row, op = .assertZero row

theorem RowsOnly.marks_nil {ops : List Op} (rowsOnly : RowsOnly ops) : marks ops = [] := by
  induction ops with
  | nil => rfl
  | cons op ops ih =>
      obtain ⟨row, rfl⟩ := rowsOnly op List.mem_cons_self
      simp only [marks_cons, Op.marks, List.nil_append]
      exact ih fun other member => rowsOnly other (List.mem_cons_of_mem _ member)

theorem RowsOnly.localLength_zero {ops : List Op} (rowsOnly : RowsOnly ops) :
    localLength ops = 0 := by
  induction ops with
  | nil => rfl
  | cons op ops ih =>
      obtain ⟨row, rfl⟩ := rowsOnly op List.mem_cons_self
      simp only [localLength, List.map_cons, List.sum_cons, Op.localLength, Nat.zero_add]
      exact ih fun other member => rowsOnly other (List.mem_cons_of_mem _ member)

theorem LiveRows.of_rows {ops : List Op} (rowsOnly : RowsOnly ops)
    (free : ∀ row, Op.assertZero row ∈ ops → row.VarsSatisfy live) : LiveRows live ops := by
  refine .of_free rowsOnly.marks_nil fun row member => ?_
  simp only [flatConstraints, List.mem_flatMap] at member
  obtain ⟨op, opMember, rowMember⟩ := member
  obtain ⟨other, rfl⟩ := rowsOnly op opMember
  simp only [Op.flatConstraints, List.mem_singleton] at rowMember
  subst rowMember
  exact free row opMember

theorem RowsOnly.ofFn {count : Nat} (rows : Fin count → Expr) :
    RowsOnly (List.ofFn fun index => Op.assertZero (rows index)) := by
  intro op member
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  exact ⟨_, rfl⟩

theorem RowsOnly.map (rows : List Expr) : RowsOnly (rows.map Op.assertZero) := by
  intro op member
  obtain ⟨row, _, rfl⟩ := List.mem_map.mp member
  exact ⟨row, rfl⟩

/-! ### Permutation batches -/

/-- A batch of marked permutations and its mark. -/
theorem LiveRows.batch {start : Nat} {recipes : List Expr} {permutations : List PermutationMark}
    (rows : recipeConstraints start recipes = permutations.flatMap Permutation.markWindow)
    (lanes : ∀ mark ∈ permutations, mark.Lanes live) :
    LiveRows live [Op.witness (WitnessBatch.arithmetic start recipes), Op.mark permutations] := by
  refine ⟨?_, ?_⟩
  · simp only [marks, flatConstraints, List.flatMap_cons, List.flatMap_nil, Op.marks,
      Op.flatConstraints, WitnessBatch.arithmetic_start, WitnessBatch.arithmetic_recipes,
      List.nil_append, List.append_nil, rows]
    exact Segmented.ofMarks permutations
  · intro mark member
    simp only [marks, List.flatMap_cons, List.flatMap_nil, Op.marks, List.nil_append,
      List.append_nil] at member
    exact lanes mark member

theorem Placed.batch {start : Nat} {recipes : List Expr} {permutations : List PermutationMark}
    (inside : ∀ mark ∈ permutations, start ≤ mark.start ∧
      mark.start + 1096 ≤ start + recipes.length)
    (ascend : permutations.Pairwise fun first second => first.start + 1096 ≤ second.start) :
    Placed start recipes.length
      [Op.witness (WitnessBatch.arithmetic start recipes), Op.mark permutations] := by
  refine ⟨fun mark member => ?_, ?_⟩
  · simp only [marks, List.flatMap_cons, List.flatMap_nil, Op.marks, List.nil_append,
      List.append_nil] at member
    exact inside mark member
  · simpa [marks, Op.marks] using ascend

theorem localLength_batch (start : Nat) (recipes : List Expr)
    (permutations : List PermutationMark) :
    localLength [Op.witness (WitnessBatch.arithmetic start recipes), Op.mark permutations] =
      recipes.length := by
  simp [localLength, Op.localLength]

/-! ### The hash compiler's marks -/

theorem absorptionMarks_inside :
    ∀ (start : Nat) (state : Layer.EState) (blocks : List (List Expr)),
      ∀ mark ∈ Hash.absorptionMarks start state blocks,
        start ≤ mark.start ∧ mark.start + 1096 ≤ start + blocks.length * 1096
  | _, _, [] => by simp [Hash.absorptionMarks]
  | start, state, block :: rest => by
      intro mark member
      simp only [Hash.absorptionMarks, List.mem_cons] at member
      rcases member with rfl | member
      · simp only [Permutation.markOf_start, List.length_cons]
        constructor <;> omega
      · have := absorptionMarks_inside (start + 1096) _ rest mark member
        simp only [List.length_cons]
        constructor <;> omega

theorem absorptionMarks_ascend :
    ∀ (start : Nat) (state : Layer.EState) (blocks : List (List Expr)),
      (Hash.absorptionMarks start state blocks).Pairwise
        fun first second => first.start + 1096 ≤ second.start
  | _, _, [] => by simp [Hash.absorptionMarks]
  | start, state, block :: rest => by
      simp only [Hash.absorptionMarks]
      refine List.Pairwise.cons (fun mark member => ?_) (absorptionMarks_ascend _ _ rest)
      have := absorptionMarks_inside (start + 1096) _ rest mark member
      simp only [Permutation.markOf_start]
      omega

theorem affine_ofNat (value : Nat) : (OfNat.ofNat value : Expr).Affine :=
  .const _

theorem zeroE_affine (lane : Fin 16) : (Hash.zeroE lane).Affine :=
  affine_ofNat 0

/-- The lanes of the absorptions' marks: affine and live before their permutation, when the
state and the blocks are. -/
theorem absorptionMarks_lanes :
    ∀ (start : Nat) (state : Layer.EState) (blocks : List (List Expr)),
      (∀ lane, (state lane).Affine) →
      (∀ lane, (state lane).VarsSatisfy fun column => column < start ∧ live column) →
      (∀ block ∈ blocks, ∀ word ∈ block, word.Affine) →
      (∀ block ∈ blocks, ∀ word ∈ block,
        word.VarsSatisfy fun column => column < start ∧ live column) →
      (∀ column, start ≤ column → column < start + blocks.length * 1096 →
        (∀ mark ∈ Hash.absorptionMarks start state blocks, ¬ mark.Inner column) → live column) →
      ∀ mark ∈ Hash.absorptionMarks start state blocks, mark.Lanes live
  | _, _, [], _, _, _, _, _ => by simp [Hash.absorptionMarks]
  | start, state, block :: rest, stateAffine, stateReads, blocksAffine, blocksRead, own => by
      intro mark member
      simp only [Hash.absorptionMarks, List.mem_cons] at member
      have word : ∀ lane : Fin 16, (block.getD lane.val 0).Affine ∧
          (block.getD lane.val 0).VarsSatisfy fun column => column < start ∧ live column := by
        intro lane
        by_cases inside : lane.val < block.length
        · rw [List.getD_eq_getElem _ _ inside]
          exact ⟨blocksAffine block List.mem_cons_self _ (List.getElem_mem inside),
            blocksRead block List.mem_cons_self _ (List.getElem_mem inside)⟩
        · rw [List.getD_eq_default _ _ (by omega)]
          exact ⟨.const _, trivial⟩
      rcases member with rfl | member
      · exact Permutation.markOf_lanes_facts start _
          (fun lane => .add (stateAffine lane) (word lane).1)
          (fun lane => ⟨stateReads lane, (word lane).2⟩)
      · refine absorptionMarks_lanes (start + 1096) _ rest (fun _ => .var _) (fun lane => ?_)
          (fun other member => blocksAffine other (List.mem_cons_of_mem _ member))
          (fun other member word wordMember =>
            (blocksRead other (List.mem_cons_of_mem _ member) word wordMember).mono word
              fun column inside => ⟨by omega, inside.2⟩)
          (fun column lower upper outside => own column (by omega)
            (by simp only [List.length_cons]; omega) fun other otherMember => ?_) mark member
        · simp only [Expr.VarsSatisfy]
          refine ⟨by omega, own _ (by omega) (by simp only [List.length_cons]; omega) ?_⟩
          intro other otherMember inner
          simp only [Hash.absorptionMarks, List.mem_cons] at otherMember
          rcases otherMember with rfl | otherMember
          · simp only [PermutationMark.Inner, Permutation.markOf_start] at inner
            omega
          · have := absorptionMarks_inside (start + 1096) _ rest other otherMember
            have := inner.1
            omega
        · simp only [Hash.absorptionMarks, List.mem_cons] at otherMember
          rcases otherMember with rfl | otherMember
          · simp only [PermutationMark.Inner, Permutation.markOf_start]
            omega
          · exact outside other otherMember

/-! ### Live outputs -/

/-- Marks that ascend, 1,096 locals apart, do not overlap: an output of one mark is an inner local
of none. -/
theorem output_not_inner {permutations : List PermutationMark}
    (ascend : permutations.Pairwise fun first second => first.start + 1096 ≤ second.start)
    {owner : PermutationMark} (member : owner ∈ permutations) {lane : Nat} (small : lane < 16) :
    ¬ InnerOf permutations (owner.start + 1080 + lane) := by
  rintro ⟨other, otherMember, lower, upper⟩
  induction permutations with
  | nil => simp at member
  | cons head rest ih =>
      obtain ⟨headBefore, restAscend⟩ := List.pairwise_cons.mp ascend
      rcases List.mem_cons.mp member with ownerHead | ownerRest <;>
        rcases List.mem_cons.mp otherMember with otherHead | otherRest
      · subst ownerHead
        subst otherHead
        omega
      · subst ownerHead
        have := headBefore other otherRest
        omega
      · subst otherHead
        have := headBefore owner ownerRest
        omega
      · exact ih restAscend ownerRest otherRest

/-- The state after the absorptions: affine, and live before the end of the absorptions. -/
theorem absorbedState_facts (start : Nat) (state : Layer.EState) (blocks : List (List Expr))
    (stateAffine : ∀ lane, (state lane).Affine)
    (stateReads : ∀ lane, (state lane).VarsSatisfy fun column => column < start ∧ live column)
    (outputs : ∀ lane : Fin 16, ∀ index, index < blocks.length →
      live (start + index * 1096 + 1080 + lane.val)) :
    ∀ lane, (Hash.absorbedState start state blocks lane).Affine ∧
      (Hash.absorbedState start state blocks lane).VarsSatisfy
        fun column => column < start + blocks.length * 1096 ∧ live column := by
  induction blocks generalizing start state with
  | nil =>
      intro lane
      exact ⟨stateAffine lane, (stateReads lane).mono _ fun column inside => by simpa using inside⟩
  | cons block rest ih =>
      intro lane
      have := ih (start + 1096) (Permutation.scheduleOutput start) (fun _ => .var _)
        (fun lane => by
          show start + 1080 + lane.val < start + 1096 ∧ live (start + 1080 + lane.val)
          exact ⟨by omega, by simpa using outputs lane 0 (by simp)⟩)
        (fun lane index below => by
          have := outputs lane (index + 1) (by simp only [List.length_cons]; omega)
          rwa [Nat.add_mul, Nat.one_mul, ← Nat.add_assoc,
            Nat.add_right_comm start (index * 1096) 1096] at this)
        lane
      simp only [Hash.absorbedState, List.length_cons]
      refine ⟨this.1, this.2.mono _ fun column inside => ⟨by omega, inside.2⟩⟩

theorem absorptionMarks_starts :
    ∀ (start : Nat) (state : Layer.EState) (blocks : List (List Expr)),
      ∀ mark ∈ Hash.absorptionMarks start state blocks,
        ∃ index, index < blocks.length ∧ mark.start = start + index * 1096
  | _, _, [] => by simp [Hash.absorptionMarks]
  | start, state, block :: rest => by
      intro mark member
      simp only [Hash.absorptionMarks, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨0, by simp, by simp⟩
      · obtain ⟨index, below, starts⟩ := absorptionMarks_starts (start + 1096) _ rest mark member
        exact ⟨index + 1, by simp only [List.length_cons]; omega,
          by rw [starts, Nat.add_mul, Nat.one_mul]; omega⟩

theorem hashCompile_length (start : Nat) (input : List Expr) :
    (Hash.compile start input).recipes.length =
      (Spec.Poseidon2.fullBlocks input).length * 1096 + 1096 := by
  unfold Hash.compile
  dsimp only
  rw [List.length_append, Hash.compileAbsorptions_recipes_length,
    Permutation.compile_schedule_recipe_count]

theorem hashMarks_starts (start : Nat) (input : List Expr) :
    ∀ mark ∈ Hash.marks start input, ∃ index,
      index ≤ (Spec.Poseidon2.fullBlocks input).length ∧ mark.start = start + index * 1096 := by
  intro mark member
  simp only [Hash.marks, List.mem_append, List.mem_singleton] at member
  rcases member with member | rfl
  · obtain ⟨index, below, starts⟩ := absorptionMarks_starts start Hash.zeroE _ mark member
    exact ⟨index, below.le, starts⟩
  · exact ⟨_, le_rfl, rfl⟩

/-- The marks of the hash: in the hash's locals, ascending. -/
theorem hashMarks_placed (start : Nat) (input : List Expr) :
    (∀ mark ∈ Hash.marks start input, start ≤ mark.start ∧
      mark.start + 1096 ≤ start + (Hash.compile start input).recipes.length) ∧
    (Hash.marks start input).Pairwise fun first second => first.start + 1096 ≤ second.start := by
  rw [hashCompile_length]
  constructor
  · intro mark member
    obtain ⟨index, below, starts⟩ := hashMarks_starts start input mark member
    have := Nat.mul_le_mul_right 1096 below
    omega
  · simp only [Hash.marks]
    rw [List.pairwise_append]
    refine ⟨absorptionMarks_ascend _ _ _, by simp, fun early earlyMember late lateMember => ?_⟩
    simp only [List.mem_singleton] at lateMember
    subst lateMember
    have := absorptionMarks_inside start Hash.zeroE _ early earlyMember
    simp only [Permutation.markOf_start]
    omega

/-- The hash's locals that are no inner local of its marks. -/
def HashOwnLive (live : Nat → Prop) (start : Nat) (input : List Expr) : Prop :=
  OwnLive live start (Hash.compile start input).recipes.length
    [Op.witness (WitnessBatch.arithmetic start (Hash.compile start input).recipes),
      Op.mark (Hash.marks start input)]

theorem hashOwnLive_iff {start : Nat} {input : List Expr} :
    HashOwnLive live start input ↔
      ∀ column, start ≤ column → column < start + (Hash.compile start input).recipes.length →
        ¬ InnerOf (Hash.marks start input) column → live column := by
  simp [HashOwnLive, OwnLive, marks, Op.marks]

/-- Each permutation output of the hash is live. -/
theorem hash_output_live {start : Nat} {input : List Expr} (own : HashOwnLive live start input)
    {index : Nat} (below : index ≤ (Spec.Poseidon2.fullBlocks input).length) {lane : Nat}
    (small : lane < 16) : live (start + index * 1096 + 1080 + lane) := by
  refine hashOwnLive_iff.mp own _ (by omega) ?_ ?_
  · rw [hashCompile_length]
    have := Nat.mul_le_mul_right 1096 below
    omega
  · rintro ⟨mark, member, lower, upper⟩
    obtain ⟨other, _, starts⟩ := hashMarks_starts start input mark member
    rw [starts] at lower upper
    rcases Nat.lt_or_ge index other with smaller | larger
    · have := Nat.mul_le_mul_right 1096 (Nat.succ_le_of_lt smaller)
      rw [Nat.succ_mul] at this
      omega
    · have := Nat.mul_le_mul_right 1096 larger
      omega

/-- The lanes of the hash's marks are affine and live before their permutation, when the input
words are affine and live before the hash. -/
theorem hashMarks_lanes {start : Nat} {input : List Expr}
    (inputAffine : ∀ word ∈ input, word.Affine)
    (inputReads : ∀ word ∈ input, word.VarsSatisfy fun column => column < start ∧ live column)
    (own : HashOwnLive live start input) :
    ∀ mark ∈ Hash.marks start input, mark.Lanes live := by
  have blockWords : ∀ block ∈ Spec.Poseidon2.fullBlocks input, ∀ word ∈ block, word ∈ input := by
    intro block member word wordMember
    simp only [Spec.Poseidon2.fullBlocks, List.mem_map] at member
    obtain ⟨_, _, rfl⟩ := member
    exact List.mem_of_mem_drop (List.mem_of_mem_take wordMember)
  have absorbed := absorbedState_facts (live := live) start Hash.zeroE
    (Spec.Poseidon2.fullBlocks input) (fun _ => zeroE_affine _) (fun _ => trivial)
    (fun lane index below => hash_output_live own below.le lane.isLt)
  intro mark member
  simp only [Hash.marks, List.mem_append, List.mem_singleton] at member
  rcases member with member | rfl
  · refine absorptionMarks_lanes start Hash.zeroE (Spec.Poseidon2.fullBlocks input)
      (fun _ => zeroE_affine _) (fun _ => trivial)
      (fun block blockMember word wordMember =>
        inputAffine word (blockWords block blockMember word wordMember))
      (fun block blockMember word wordMember =>
        inputReads word (blockWords block blockMember word wordMember))
      (fun column lower upper outside => hashOwnLive_iff.mp own column lower ?_ ?_) mark member
    · rw [hashCompile_length]
      exact Nat.lt_of_lt_of_le upper (Nat.add_le_add_left (Nat.le_add_right _ _) _)
    · rintro ⟨other, otherMember, inner⟩
      simp only [Hash.marks, List.mem_append, List.mem_singleton] at otherMember
      rcases otherMember with otherMember | rfl
      · exact outside other otherMember inner
      · have := inner.1
        simp only [Permutation.markOf_start] at this
        omega
  · have lastWord : ∀ word ∈ Spec.Poseidon2.lastWords input, word ∈ input :=
      fun word member => List.mem_of_mem_drop member
    refine Permutation.markOf_lanes_facts _ _ (fun lane => ?_) (fun lane => ?_)
    · have stateAffine := (absorbed lane).1
      unfold Hash.padE
      split
      · rename_i inside
        rw [List.getD_eq_getElem _ _ inside]
        exact .add stateAffine (inputAffine _ (lastWord _ (List.getElem_mem inside)))
      · split
        · exact .add stateAffine (affine_ofNat 1)
        · exact stateAffine
    · have stateReads := (absorbed lane).2
      unfold Hash.padE
      split
      · rename_i inside
        rw [List.getD_eq_getElem _ _ inside]
        refine ⟨stateReads, (inputReads _ (lastWord _ (List.getElem_mem inside))).mono _
          fun column reads => ⟨by omega, reads.2⟩⟩
      · split
        · exact ⟨stateReads, trivial⟩
        · exact stateReads

/-- The hash's batch and mark. -/
theorem hash_liveRows {start : Nat} {input : List Expr}
    (inputAffine : ∀ word ∈ input, word.Affine)
    (inputReads : ∀ word ∈ input, word.VarsSatisfy fun column => column < start ∧ live column)
    (own : HashOwnLive live start input) :
    LiveRows live [Op.witness (WitnessBatch.arithmetic start (Hash.compile start input).recipes),
      Op.mark (Hash.marks start input)] :=
  .batch (Hash.compile_windows start input) (hashMarks_lanes inputAffine inputReads own)

theorem hash_placed (start : Nat) (input : List Expr) :
    Placed start (Hash.compile start input).recipes.length
      [Op.witness (WitnessBatch.arithmetic start (Hash.compile start input).recipes),
        Op.mark (Hash.marks start input)] :=
  .batch (hashMarks_placed start input).1 (hashMarks_placed start input).2

/-- The hash's digest lanes are live. -/
theorem hash_digest_live {start : Nat} {input : List Expr} (own : HashOwnLive live start input)
    (lane : Fin 4) : (Hash.digestE (Hash.compile start input).output lane).VarsSatisfy live := by
  have output : (Hash.compile start input).output = Permutation.scheduleOutput
      (start + (Spec.Poseidon2.fullBlocks input).length * 1096) := by
    unfold Hash.compile
    dsimp only
    rw [← Permutation.scheduleOutput_eq_compile, Hash.compileAbsorptions_recipes_length]
  rw [output]
  exact hash_output_live own le_rfl (Nat.lt_trans lane.isLt (by decide))

end NightstreamFPrime.Gadgets.Poseidon2
