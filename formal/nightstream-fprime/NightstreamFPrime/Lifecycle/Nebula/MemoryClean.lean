import NightstreamFPrime.Gadgets.Poseidon2.CleanChildren
import NightstreamFPrime.Lifecycle.Nebula.MemorySupport

/-! Owns the exporter's blocks of the memory-application circuit (SPEC.md section 4.4): its nine
sponge children are placed one after another (`marksPlaced`), and its rows are free rows that read
only live columns and the children's marks' rows (`marksClean`). The children's chunk words and
the squeeze states are affine and live before each child starts. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

/-! ### Affine words -/

section Affine

variable {p : Geometry} {app : PortApp p} {i : AppInterface p app} {offset : ℕ}

theorem wire_affine (affine : Application.InputsAffine i offset) (k : ℕ) :
    (wire p app i offset k).Affine := by
  unfold wire
  split
  · exact affine.witness _
  · exact affine_ofNat 0

theorem subst_affine (wires : ℕ → Circuit.Expr) (each : ∀ k, (wires k).Affine)
    {e : Circuit.Expr} (affine : e.Affine) : (Expr.subst wires e).Affine := by
  induction affine with
  | var k => exact each k
  | const value => exact .const value
  | @add left right _ _ leftIH rightIH => exact .add leftIH rightIH
  | @scaleLeft factor expression _ ih => exact .scaleLeft factor ih
  | @scaleRight factor expression _ ih => exact .scaleRight factor ih

theorem sub_affine (affine : Application.InputsAffine i offset) {e : Circuit.Expr}
    (eAffine : e.Affine) : (sub p app i offset e).Affine :=
  subst_affine _ (wire_affine affine) eAffine

theorem chunkFrom_affine : ∀ {acc : Circuit.Expr} {start : ℕ} {l : List Circuit.Expr},
    acc.Affine → (∀ x ∈ l, x.Affine) → (Expr.chunkFrom acc start l).Affine
  | _, _, [], accAffine, _ => accAffine
  | acc, start, b :: bs, accAffine, each =>
    chunkFrom_affine (acc := acc + .const ((2 : F) ^ start) * b) (start := start + 1)
      (.add accAffine (.scaleLeft _ (each b (by simp)))) fun x hx => each x (by simp [hx])

theorem chunk_affine :
    ∀ {l : List Circuit.Expr}, (∀ x ∈ l, x.Affine) → (Expr.chunk l).Affine
  | [], _ => affine_ofNat 0
  | b :: bs, each => chunkFrom_affine (each b (by simp)) fun x hx => each x (by simp [hx])

theorem pack_affine {l : List Circuit.Expr} (each : ∀ x ∈ l, x.Affine) :
    ∀ e ∈ Expr.pack l, e.Affine := by
  induction l using Expr.pack.induct with
  | case1 => simp [Expr.pack]
  | case2 l nonempty ih =>
    rw [Expr.pack, dite_eq_right nonempty]
    intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact chunk_affine fun x hx => each x (List.mem_of_mem_take hx)
    · exact ih (fun x hx => each x (List.mem_of_mem_drop hx)) e member

theorem transcriptChunks_affine {blocks : List (List Circuit.Expr)}
    (each : ∀ b ∈ blocks, ∀ e ∈ b, e.Affine) :
    ∀ chunk ∈ transcriptChunks blocks, ∀ e ∈ chunk, e.Affine := by
  intro chunk chunkMember e member
  simp only [transcriptChunks, List.mem_flatMap, Hash.inputChunks, List.mem_map] at chunkMember
  obtain ⟨b, blockMember, c, -, rfl⟩ := chunkMember
  have inBlock := List.mem_of_mem_drop (List.mem_of_mem_take member)
  rcases List.mem_cons.mp inBlock with rfl | inWords
  · exact .const _
  · exact each b blockMember e inWords

theorem wires_affine (affine : Application.InputsAffine i offset) (f : ℕ → ℕ) (n : ℕ) :
    ∀ e ∈ wires p app i offset f n, e.Affine := by
  intro e member
  obtain ⟨k, rfl⟩ := List.mem_ofFn.mp member
  exact wire_affine affine _

theorem textE_affine (text : String) : ∀ e ∈ textE text, e.Affine := by
  intro e member
  obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
  exact .const _

theorem opLane_affine (j : ℕ) : ∀ e ∈ Sym.opLane p j, e.Affine := by
  intro e member
  simp only [Sym.opLane, List.mem_append, List.mem_cons, List.mem_ofFn, List.not_mem_nil,
    or_false] at member
  rcases member with
    ((((rfl | rfl | rfl) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩ <;> exact .var _

theorem opsLanes_affine : ∀ e ∈ opsLanes p, e.Affine := by
  intro e member
  obtain ⟨lane, laneMember, member⟩ := List.mem_flatten.mp member
  obtain ⟨j, rfl⟩ := List.mem_ofFn.mp laneMember
  exact opLane_affine j e member

theorem scanLanes_affine (start : ℕ → ℕ) : ∀ e ∈ scanLanes p start, e.Affine := by
  intro e member
  obtain ⟨lane, laneMember, member⟩ := List.mem_flatten.mp member
  obtain ⟨j, rfl⟩ := List.mem_ofFn.mp laneMember
  simp only [Sym.scanLane, List.mem_append, List.mem_ofFn] at member
  rcases member with ⟨k, rfl⟩ | ⟨k, rfl⟩ <;> exact .var _

theorem stateBlocks_affine (affine : Application.InputsAffine i offset)
    (appWords carry : ℕ → ℕ) :
    ∀ b ∈ stateBlocks p app i offset appWords carry, ∀ e ∈ b, e.Affine := by
  intro b member
  simp only [stateBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_affine _
  · exact wires_affine affine _ _
  · exact wires_affine affine _ _

theorem chainBlocks_affine (affine : Application.InputsAffine i offset) (lane : Lane)
    (previous : ℕ) {lanes : List Circuit.Expr} (lanesAffine : ∀ e ∈ lanes, e.Affine) :
    ∀ b ∈ chainBlocks p app i offset lane previous lanes, ∀ e ∈ b, e.Affine := by
  intro b member
  simp only [chainBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_affine _
  · intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact wire_affine affine _
    · exact wires_affine affine _ _ e member
  · apply pack_affine
    intro x member
    obtain ⟨y, yMember, rfl⟩ := List.mem_map.mp member
    exact sub_affine affine (lanesAffine y yMember)

theorem etaBlocks_affine (affine : Application.InputsAffine i offset) :
    ∀ b ∈ etaBlocks p app i offset, ∀ e ∈ b, e.Affine := by
  intro b member
  simp only [etaBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact textE_affine _
  · intro e member
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
    exact .const _
  · intro e member
    rw [List.mem_singleton.mp member]
    refine sub_affine affine ?_
    unfold Sym.cIn Sym.carryWord
    split
    · exact .var _
    · exact affine_ofNat 0
  · intro e member
    simp only [List.mem_append] at member
    rcases member with (member | member) | member <;> exact wires_affine affine _ _ e member

end Affine

/-! ### Sponge children -/

section Children

variable {live : ℕ → Prop}

theorem spongePlaced_cons (child : Sponge.Interface) (name : String) {start next : ℕ}
    {rest : List Op} (ends : start + (child.chunks start).length * 1096 = next)
    (tail : Placed next (localLength rest) rest) :
    Placed start (localLength (Sequence.childOp name (Sponge.circuit child) start :: rest))
      (Sequence.childOp name (Sponge.circuit child) start :: rest) := by
  subst ends
  exact Placed.child_cons (circuit := Sponge.circuit child) (Sponge.localLength_eq child start)
    (Sponge.placed child start) tail

theorem spongeOwn_cons (child : Sponge.Interface) (name : String) {start next : ℕ}
    {rest : List Op} (ends : start + (child.chunks start).length * 1096 = next)
    (own : OwnLive live start
      (localLength (Sequence.childOp name (Sponge.circuit child) start :: rest))
      (Sequence.childOp name (Sponge.circuit child) start :: rest))
    (tail : Placed next (localLength rest) rest) :
    OwnLive live start (localLength (Sponge.opsAt child start)) (Sponge.opsAt child start) ∧
      OwnLive live next (localLength rest) rest := by
  subst ends
  exact OwnLive.child_cons (circuit := Sponge.circuit child) own
    (Sponge.localLength_eq child start) (Sponge.placed child start) tail

theorem spongeClean_cons (child : Sponge.Interface) (name : String) (start : ℕ)
    {rest : List Op} (clean : Clean live (Sponge.opsAt child start)) (tail : Clean live rest) :
    Clean live (Sequence.childOp name (Sponge.circuit child) start :: rest) :=
  Clean.child_cons (circuit := Sponge.circuit child) clean tail

/-- An absorbing child is clean when its blocks are affine and live before it. -/
theorem absorbing_clean {blocks : List (List Circuit.Expr)} {before start : ℕ}
    (le : before ≤ start) (blocksAffine : ∀ b ∈ blocks, ∀ e ∈ b, e.Affine)
    (blocksRead : ∀ b ∈ blocks, ∀ e ∈ b,
      e.VarsSatisfy fun column => column < before ∧ live column)
    (own : OwnLive live start (localLength (Sponge.opsAt (absorbing blocks) start))
      (Sponge.opsAt (absorbing blocks) start)) :
    Clean live (Sponge.opsAt (absorbing blocks) start) :=
  Sponge.clean (absorbing blocks) start (fun lane => zeroE_affine lane) (fun _ => trivial)
    (transcriptChunks_affine blocksAffine)
    (fun chunk member e eMember =>
      (transcriptChunks_supported blocksRead chunk member e eMember).mono e
        fun _ inside => ⟨Nat.lt_of_lt_of_le inside.1 le, inside.2⟩)
    own

theorem absorbing_output_live {blocks : List (List Circuit.Expr)} {start : ℕ}
    (own : OwnLive live start (localLength (Sponge.opsAt (absorbing blocks) start))
      (Sponge.opsAt (absorbing blocks) start)) (lane : Fin 16) :
    (Sponge.output (absorbing blocks) start lane).VarsSatisfy live :=
  (Sponge.output_live (absorbing blocks) start (fun _ => trivial) own lane).mono _
    fun _ inside => inside.2

theorem noChunks (property : Circuit.Expr → Prop) :
    ∀ chunk ∈ ([[]] : List (List Circuit.Expr)), ∀ e ∈ chunk, property e := by
  simp

end Children

/-! ### The memory circuit -/

variable (p : Geometry) (app : PortApp p) (i : AppInterface p app) (offset : ℕ)

/-- Each child ends where the next starts. -/
theorem child_ends :
    offset + ((stateIn p app i offset).chunks offset).length * 1096 =
      stateOutStart p app i offset ∧
    stateOutStart p app i offset +
      ((stateOut p app i offset).chunks (stateOutStart p app i offset)).length * 1096 =
      chainOpsStart p app i offset ∧
    chainOpsStart p app i offset +
      ((chainOps p app i offset).chunks (chainOpsStart p app i offset)).length * 1096 =
      chainInitialStart p app i offset ∧
    chainInitialStart p app i offset +
      ((chainInitial p app i offset).chunks (chainInitialStart p app i offset)).length * 1096 =
      chainFinalStart p app i offset ∧
    chainFinalStart p app i offset +
      ((chainFinal p app i offset).chunks (chainFinalStart p app i offset)).length * 1096 =
      etaStart p app i offset ∧
    etaStart p app i offset +
      ((eta p app i offset).chunks (etaStart p app i offset)).length * 1096 =
      squeeze1Start p app i offset ∧
    squeeze1Start p app i offset +
      ((squeeze1 p app i offset).chunks (squeeze1Start p app i offset)).length * 1096 =
      squeeze2Start p app i offset ∧
    squeeze2Start p app i offset +
      ((squeeze2 p app i offset).chunks (squeeze2Start p app i offset)).length * 1096 =
      squeeze3Start p app i offset ∧
    squeeze3Start p app i offset +
      ((squeeze3 p app i offset).chunks (squeeze3Start p app i offset)).length * 1096 =
      endOffset p app i offset := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, by rw [etaChunks_length]; rfl, ?_, ?_, ?_⟩
  · simp only [squeeze2Start, squeeze1, permuting, List.length_singleton, one_mul]
  · simp only [squeeze3Start, squeeze2, permuting, List.length_singleton, one_mul]
  · simp only [endOffset, squeeze3, permuting, List.length_singleton, one_mul]

theorem marksPlaced :
    Placed offset (localLength (opsAt p app i offset)) (opsAt p app i offset) := by
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p app i offset
  have children : Placed offset (localLength (childOps p app i offset)) (childOps p app i offset) :=
    spongePlaced_cons _ _ e1 <| spongePlaced_cons _ _ e2 <| spongePlaced_cons _ _ e3 <|
      spongePlaced_cons _ _ e4 <| spongePlaced_cons _ _ e5 <| spongePlaced_cons _ _ e6 <|
      spongePlaced_cons _ _ e7 <| spongePlaced_cons _ _ e8 <| spongePlaced_cons _ _ e9 <|
      Placed.nil _
  rw [opsAt, Sequence.localLength_append]
  exact children.append (.of_marks_nil (RowsOnly.map _).marks_nil)

/-- The circuit's rows are its children's rows and parent rows that read the children's
outputs, the interface words, and the wires. -/
theorem marksClean (live : ℕ → Prop) (affine : Application.InputsAffine i offset)
    (inputs : Application.InputsSupported i offset fun column => column < offset ∧ live column)
    (own : OwnLive live offset (localLength (opsAt p app i offset)) (opsAt p app i offset)) :
    Clean live (opsAt p app i offset) := by
  obtain ⟨e1, e2, e3, e4, e5, e6, e7, e8, e9⟩ := child_ends p app i offset
  have wires : WiresSupported p app i offset fun column => column < offset ∧ live column :=
    inputs.witness
  have wiresLive : WiresSupported p app i offset live :=
    fun index => (inputs.witness index).mono _ fun _ inside => inside.2
  have r8 := spongePlaced_cons (squeeze3 p app i offset) "nebula.eta_squeeze_3" e9
    (Placed.nil (endOffset p app i offset))
  have r7 := spongePlaced_cons (squeeze2 p app i offset) "nebula.eta_squeeze_2" e8 r8
  have r6 := spongePlaced_cons (squeeze1 p app i offset) "nebula.eta_squeeze_1" e7 r7
  have r5 := spongePlaced_cons (eta p app i offset) "nebula.eta" e6 r6
  have r4 := spongePlaced_cons (chainFinal p app i offset) "nebula.chain_fs" e5 r5
  have r3 := spongePlaced_cons (chainInitial p app i offset) "nebula.chain_is" e4 r4
  have r2 := spongePlaced_cons (chainOps p app i offset) "nebula.chain_ops" e3 r3
  have r1 := spongePlaced_cons (stateOut p app i offset) "nebula.state_out" e2 r2
  rw [opsAt, Sequence.localLength_append] at own
  have ownChildren : OwnLive live offset (localLength (childOps p app i offset))
      (childOps p app i offset) :=
    own.left (.of_marks_nil (RowsOnly.map _).marks_nil)
  obtain ⟨own0, rest⟩ := spongeOwn_cons (stateIn p app i offset) "nebula.state_in" e1
    ownChildren r1
  obtain ⟨own1, rest⟩ := spongeOwn_cons (stateOut p app i offset) "nebula.state_out" e2 rest r2
  obtain ⟨own2, rest⟩ := spongeOwn_cons (chainOps p app i offset) "nebula.chain_ops" e3 rest r3
  obtain ⟨own3, rest⟩ :=
    spongeOwn_cons (chainInitial p app i offset) "nebula.chain_is" e4 rest r4
  obtain ⟨own4, rest⟩ := spongeOwn_cons (chainFinal p app i offset) "nebula.chain_fs" e5 rest r5
  obtain ⟨own5, rest⟩ := spongeOwn_cons (eta p app i offset) "nebula.eta" e6 rest r6
  obtain ⟨own6, rest⟩ :=
    spongeOwn_cons (squeeze1 p app i offset) "nebula.eta_squeeze_1" e7 rest r7
  obtain ⟨own7, rest⟩ :=
    spongeOwn_cons (squeeze2 p app i offset) "nebula.eta_squeeze_2" e8 rest r8
  obtain ⟨own8, -⟩ := spongeOwn_cons (squeeze3 p app i offset) "nebula.eta_squeeze_3" e9 rest
    (Placed.nil (endOffset p app i offset))
  -- The squeeze states are live before the next squeeze.
  have etaReads : ∀ lane, (etaState p app i offset lane).VarsSatisfy
      fun column => column < squeeze1Start p app i offset ∧ live column := by
    intro lane
    rw [etaState_output, ← e6]
    exact Sponge.output_live (eta p app i offset) (etaStart p app i offset) (fun _ => trivial)
      own5 lane
  have squeeze1Reads : ∀ lane, (squeeze1State p app i offset lane).VarsSatisfy
      fun column => column < squeeze2Start p app i offset ∧ live column := by
    intro lane
    rw [← e7]
    exact Sponge.output_live (squeeze1 p app i offset) (squeeze1Start p app i offset) etaReads
      own6 lane
  have squeeze2Reads : ∀ lane, (squeeze2State p app i offset lane).VarsSatisfy
      fun column => column < squeeze3Start p app i offset ∧ live column := by
    intro lane
    rw [← e8]
    exact Sponge.output_live (squeeze2 p app i offset) (squeeze2Start p app i offset)
      squeeze1Reads own7 lane
  have squeeze3Reads : ∀ lane, (squeeze3State p app i offset lane).VarsSatisfy
      fun column => column < endOffset p app i offset ∧ live column := by
    intro lane
    rw [← e9]
    exact Sponge.output_live (squeeze3 p app i offset) (squeeze3Start p app i offset)
      squeeze2Reads own8 lane
  have freshAffine : ∀ start lane, (Permutation.freshState start lane).Affine :=
    fun _ _ => .var _
  -- The children.
  have children : Clean live (childOps p app i offset) :=
    spongeClean_cons _ _ _
      (absorbing_clean le_rfl (stateBlocks_affine affine _ _) (stateBlocks_supported wires _ _)
        own0) <|
    spongeClean_cons _ _ _
      (absorbing_clean (by omega) (stateBlocks_affine affine _ _)
        (stateBlocks_supported wires _ _) own1) <|
    spongeClean_cons _ _ _
      (absorbing_clean (by omega) (chainBlocks_affine affine _ _ opsLanes_affine)
        (chainBlocks_supported wires _ _ _) own2) <|
    spongeClean_cons _ _ _
      (absorbing_clean (by omega) (chainBlocks_affine affine _ _ (scanLanes_affine _))
        (chainBlocks_supported wires _ _ _) own3) <|
    spongeClean_cons _ _ _
      (absorbing_clean (by omega) (chainBlocks_affine affine _ _ (scanLanes_affine _))
        (chainBlocks_supported wires _ _ _) own4) <|
    spongeClean_cons _ _ _
      (absorbing_clean (by omega) (etaBlocks_affine affine) (etaBlocks_supported wires) own5) <|
    spongeClean_cons _ _ _
      (Sponge.clean (squeeze1 p app i offset) (squeeze1Start p app i offset) (freshAffine _)
        etaReads (noChunks _) (noChunks _) own6) <|
    spongeClean_cons _ _ _
      (Sponge.clean (squeeze2 p app i offset) (squeeze2Start p app i offset)
        (Sponge.output_affine (squeeze1 p app i offset) (squeeze1Start p app i offset)
          (freshAffine _))
        squeeze1Reads (noChunks _) (noChunks _) own7) <|
    spongeClean_cons _ _ _
      (Sponge.clean (squeeze3 p app i offset) (squeeze3Start p app i offset)
        (Sponge.output_affine (squeeze2 p app i offset) (squeeze2Start p app i offset)
          (Sponge.output_affine (squeeze1 p app i offset) (squeeze1Start p app i offset)
            (freshAffine _)))
        squeeze2Reads (noChunks _) (noChunks _) own8)
      Clean.nil
  -- The parent rows.
  have laneOf : ∀ {state : Sponge.EState}, (∀ lane, (state lane).VarsSatisfy live) →
      ∀ k, (lane state k).VarsSatisfy live := fun each k => each _
  have liveOf : ∀ {state : Sponge.EState} {bound : ℕ}, (∀ lane, (state lane).VarsSatisfy
      fun column => column < bound ∧ live column) → ∀ lane, (state lane).VarsSatisfy live :=
    fun each lane => (each lane).mono _ fun _ inside => inside.2
  have inputLive : ∀ k, (i.input offset k).VarsSatisfy live :=
    fun k => (inputs.input k).mono _ fun _ inside => inside.2
  have outputLive : ∀ k, (i.output offset k).VarsSatisfy live :=
    fun k => (inputs.output k).mono _ fun _ inside => inside.2
  have rows : ∀ row ∈ assertions p app i offset, row.VarsSatisfy live := by
    intro row member
    simp only [assertions, hashChecks, joinChecks, List.foldl_cons, List.foldl_nil,
      List.nil_append, List.mem_append, List.mem_map, List.mem_finRange, true_and,
      List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with
      (((((⟨k, rfl⟩ | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | (rfl | rfl | rfl | rfl)) |
        ⟨e', -, rfl⟩
    · exact .sub _ _ _ (laneOf (absorbing_output_live own0) k) (inputLive k)
    · exact .sub _ _ _ (laneOf (absorbing_output_live own1) k) (outputLive k)
    · exact .sub _ _ _ (laneOf (absorbing_output_live own2) k) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (absorbing_output_live own3) k) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (absorbing_output_live own4) k) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (liveOf etaReads) 0) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (liveOf squeeze1Reads) 0) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (liveOf squeeze2Reads) 0) (wire_supported wiresLive _)
    · exact .sub _ _ _ (laneOf (liveOf squeeze3Reads) 0) (wire_supported wiresLive _)
    · exact sub_supported wiresLive e'
  rw [opsAt]
  exact children.append (Clean.assertions rows)

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
