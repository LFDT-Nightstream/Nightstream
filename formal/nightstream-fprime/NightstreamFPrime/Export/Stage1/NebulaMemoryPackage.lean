import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness
import NightstreamFPrime.Lifecycle.Nebula.RowNames

/-!
Owns the package geometry of the memory application for any plan with two
ports: the affine inputs and direct rows of the sponge children, the fresh
columns, the local length, and the Stage 1 size bounds from a plan's
multiplication and row counts. A plan's package module supplies those two
counts and the fixed point (`NebulaMemoryV1Package`, `NebulaMemoryN1Package`).
No Poseidon2 recipe is evaluated. It does not authorize a Rust loader.
-/

namespace NightstreamFPrime.Export.Stage1.NebulaMemoryPackage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Spec

variable (plan : Spec.Nebula.Plan) (two : plan.bOps = 2)

/-- The memory application of `plan` as a Stage 1 program. -/
def application : Lifecycle.Stage1.Application.Program := MemoryApp.program plan two

abbrev wires : MemoryApp.AppInterface plan :=
  Layout.Stage1.ApplicationInputs.interface (application plan two)

abbrev start : ℕ := Layout.Stage1.ApplicationInputs.localStart (application plan two)

variable {plan two}

theorem constraints_eq :
    ApplicationPackage.constraints (application plan two) (ApplicationPackage.productionColumns (application plan two))
        (start plan two) = flatConstraints (MemoryApp.opsAt plan (wires plan two) (start plan two)) :=
  rfl

theorem operations_eq :
    ApplicationPackage.operations (application plan two) (ApplicationPackage.productionColumns (application plan two))
        (start plan two) = MemoryApp.opsAt plan (wires plan two) (start plan two) :=
  rfl

/-! ### Affine inputs -/

theorem wire_affine (k : ℕ) : R1CS.IsAffine (MemoryApp.wire plan (wires plan two) (start plan two) k) := by
  unfold MemoryApp.wire
  split
  · exact R1CS.isAffine_var _
  · exact R1CS.isAffine_const _

theorem sub_mulCount (e : Circuit.Expr) :
    R1CS.mulCount (MemoryApp.sub plan (wires plan two) (start plan two) e) = R1CS.mulCount e := by
  induction e with
  | var k =>
    change R1CS.mulCount (MemoryApp.wire plan (wires plan two) (start plan two) k) = 0
    unfold MemoryApp.wire
    split <;> rfl
  | const value => rfl
  | add left right ihl ihr =>
    change R1CS.mulCount (MemoryApp.sub plan (wires plan two) (start plan two) left) +
      R1CS.mulCount (MemoryApp.sub plan (wires plan two) (start plan two) right) = _
    rw [ihl, ihr]
    rfl
  | mul left right ihl ihr =>
    change R1CS.mulCount (MemoryApp.sub plan (wires plan two) (start plan two) left) +
      R1CS.mulCount (MemoryApp.sub plan (wires plan two) (start plan two) right) + 1 = _
    rw [ihl, ihr]
    rfl

theorem chunkFrom_affine : ∀ {acc : Circuit.Expr} {bitStart : ℕ} {l : List Circuit.Expr},
    R1CS.IsAffine acc → (∀ x ∈ l, R1CS.IsAffine x) →
      R1CS.IsAffine (Nebula.Expr.chunkFrom acc bitStart l)
  | _, _, [], accAffine, _ => accAffine
  | acc, bitStart, b :: bs, accAffine, each =>
    chunkFrom_affine (acc := acc + .const ((2 : F) ^ bitStart) * b) (bitStart := bitStart + 1)
      (R1CS.IsAffine.add accAffine (R1CS.IsAffine.const_mul _ (each b (by simp))))
      fun x hx => each x (by simp [hx])

theorem chunk_affine : ∀ {l : List Circuit.Expr}, (∀ x ∈ l, R1CS.IsAffine x) →
    R1CS.IsAffine (Nebula.Expr.chunk l)
  | [], _ => R1CS.isAffine_const _
  | b :: bs, each => chunkFrom_affine (each b (by simp)) fun x hx => each x (by simp [hx])

theorem pack_affine {l : List Circuit.Expr} (each : ∀ x ∈ l, R1CS.IsAffine x) :
    Poseidon2.ListAffine (Nebula.Expr.pack l) := by
  induction l using Nebula.Expr.pack.induct with
  | case1 => simp [Nebula.Expr.pack, Poseidon2.ListAffine]
  | case2 l nonempty ih =>
    rw [Nebula.Expr.pack, dif_neg nonempty]
    intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact chunk_affine fun x hx => each x (List.mem_of_mem_take hx)
    · exact ih (fun x hx => each x (List.mem_of_mem_drop hx)) e member

theorem transcriptChunks_affine {blocks : List (List Circuit.Expr)}
    (each : ∀ b ∈ blocks, Poseidon2.ListAffine b) :
    Poseidon2.BlocksAffine (transcriptChunks blocks) := by
  intro chunk chunkMember e member
  simp only [transcriptChunks, List.mem_flatMap, Gadgets.Poseidon2.Hash.inputChunks,
    List.mem_map] at chunkMember
  obtain ⟨b, blockMember, c, -, rfl⟩ := chunkMember
  have inBlock := List.mem_of_mem_drop (List.mem_of_mem_take member)
  rcases List.mem_cons.mp inBlock with rfl | inWords
  · exact R1CS.isAffine_const _
  · exact each b blockMember e inWords

theorem wires_affine (f : ℕ → ℕ) (n : ℕ) : Poseidon2.ListAffine (MemoryApp.wires plan (wires plan two) (start plan two) f n) := by
  intro e member
  obtain ⟨k, rfl⟩ := List.mem_ofFn.mp member
  exact wire_affine _

theorem textE_affine (text : String) : Poseidon2.ListAffine (MemoryApp.textE text) := by
  intro e member
  obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
  exact R1CS.isAffine_const _

theorem sub_word_affine (k : ℕ) : R1CS.IsAffine (MemoryApp.sub plan (wires plan two) (start plan two) (.var k)) :=
  wire_affine k

theorem stateBlocks_affine (app carry : ℕ → ℕ) :
    ∀ b ∈ MemoryApp.stateBlocks plan (wires plan two) (start plan two) app carry, Poseidon2.ListAffine b := by
  intro b member
  simp only [MemoryApp.stateBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_affine _
  · exact wires_affine _ _
  · exact wires_affine _ _

theorem chainBlocks_affine (lane : Spec.Nebula.Lane) (previous : ℕ) (lanes : List Circuit.Expr)
    (laneWords : ∀ x ∈ lanes, ∃ k, x = .var k) :
    ∀ b ∈ MemoryApp.chainBlocks plan (wires plan two) (start plan two) lane previous lanes, Poseidon2.ListAffine b := by
  intro b member
  simp only [MemoryApp.chainBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_affine _
  · intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact wire_affine _
    · exact wires_affine _ _ e member
  · apply pack_affine
    intro x member
    obtain ⟨y, yMember, rfl⟩ := List.mem_map.mp member
    obtain ⟨k, rfl⟩ := laneWords y yMember
    exact sub_word_affine k

theorem etaBlocks_affine : ∀ b ∈ MemoryApp.etaBlocks plan (wires plan two) (start plan two), Poseidon2.ListAffine b := by
  intro b member
  simp only [MemoryApp.etaBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact textE_affine _
  · intro e member
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
    exact R1CS.isAffine_const _
  · intro e member
    rw [List.mem_singleton.mp member]
    exact sub_word_affine _
  · intro e member
    simp only [List.mem_append] at member
    rcases member with (member | member) | member <;> exact wires_affine _ _ e member

/-! ### Lane words -/

theorem opsLanes_words : ∀ x ∈ MemoryApp.opsLanes plan, ∃ k, x = .var k := by
  intro x member
  simp only [MemoryApp.opsLanes, List.mem_flatten, List.mem_ofFn] at member
  obtain ⟨_, ⟨j, rfl⟩, member⟩ := member
  simp only [Sym.opLane, List.mem_append, List.mem_cons, List.mem_ofFn, List.not_mem_nil,
    or_false] at member
  rcases member with ((((rfl | rfl | rfl) | ⟨_, rfl⟩) | ⟨_, rfl⟩) | ⟨_, rfl⟩) | ⟨_, rfl⟩ <;>
    exact ⟨_, rfl⟩

theorem scanLanes_words (slotStart : ℕ → ℕ) : ∀ x ∈ MemoryApp.scanLanes plan slotStart, ∃ k, x = .var k := by
  intro x member
  simp only [MemoryApp.scanLanes, List.mem_flatten, List.mem_ofFn] at member
  obtain ⟨_, ⟨j, rfl⟩, member⟩ := member
  simp only [Sym.scanLane, List.mem_append, List.mem_ofFn] at member
  rcases member with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> exact ⟨_, rfl⟩

/-! ### The children's recipes are direct rows -/

theorem absorbing_direct (blocks : List (List Circuit.Expr)) (childStart : ℕ)
    (each : ∀ b ∈ blocks, Poseidon2.ListAffine b) :
    R1CS.RecipesDirect childStart (Sponge.program (MemoryApp.absorbing blocks) childStart).recipes :=
  Poseidon2.compileAbsorptions_direct childStart _ _ Poseidon2.zeroE_affine
    (transcriptChunks_affine each)

theorem absorbing_output_affine (blocks : List (List Circuit.Expr)) (childStart : ℕ)
    (each : ∀ b ∈ blocks, Poseidon2.ListAffine b) :
    Poseidon2.StateAffine (Sponge.output (MemoryApp.absorbing blocks) childStart) := by
  rw [Sponge.output_eq_program]
  exact Poseidon2.compileAbsorptions_output_affine childStart _ _ Poseidon2.zeroE_affine
    (transcriptChunks_affine each)

theorem noChunks_affine : Poseidon2.BlocksAffine [([] : List Circuit.Expr)] := by
  intro b member e eMember
  simp at member
  subst member
  simp at eMember

theorem permuting_direct (state : Sponge.EState) (childStart : ℕ)
    (affine : Poseidon2.StateAffine state) :
    R1CS.RecipesDirect childStart (Sponge.program (MemoryApp.permuting state) childStart).recipes :=
  Poseidon2.compileAbsorptions_direct childStart _ _ affine noChunks_affine

theorem permuting_output_affine (state : Sponge.EState) (childStart : ℕ)
    (affine : Poseidon2.StateAffine state) :
    Poseidon2.StateAffine (Sponge.output (MemoryApp.permuting state) childStart) := by
  rw [Sponge.output_eq_program]
  exact Poseidon2.compileAbsorptions_output_affine childStart _ _ affine noChunks_affine

/-! ### Fresh columns -/

theorem constraintFreshCount_le (e : Circuit.Expr) : R1CS.constraintFreshCount e ≤ R1CS.mulCount e := by
  unfold R1CS.constraintFreshCount
  split <;> omega

theorem totalFreshCount_le (l : List Circuit.Expr) :
    R1CS.totalFreshCount l ≤ (l.map R1CS.mulCount).sum := by
  induction l with
  | nil => exact le_rfl
  | cons e rest ih =>
    simp only [R1CS.totalFreshCount, List.map_cons, List.sum_cons] at ih ⊢
    exact Nat.add_le_add (constraintFreshCount_le e) ih

theorem sub_affine {a b : Circuit.Expr} (ha : R1CS.IsAffine a) (hb : R1CS.IsAffine b) :
    R1CS.IsAffine (a - b) :=
  R1CS.IsAffine.add ha (R1CS.IsAffine.const_mul (-1) hb)

theorem etaState_affine : Poseidon2.StateAffine (MemoryApp.etaState plan (wires plan two) (start plan two)) := by
  rw [MemoryApp.etaState_output]
  exact absorbing_output_affine _ _ etaBlocks_affine

theorem squeeze1State_affine : Poseidon2.StateAffine (MemoryApp.squeeze1State plan (wires plan two) (start plan two)) :=
  permuting_output_affine _ _ etaState_affine

theorem squeeze2State_affine : Poseidon2.StateAffine (MemoryApp.squeeze2State plan (wires plan two) (start plan two)) :=
  permuting_output_affine _ _ squeeze1State_affine

theorem squeeze3State_affine : Poseidon2.StateAffine (MemoryApp.squeeze3State plan (wires plan two) (start plan two)) :=
  permuting_output_affine _ _ squeeze2State_affine

theorem childConstraints_noFresh :
    R1CS.totalFreshCount (flatConstraints (MemoryApp.childOps plan (wires plan two) (start plan two))) = 0 := by
  apply R1CS.totalFreshCount_eq_zero_of_noFresh
  intro e member
  obtain ⟨op, opMember, member⟩ := List.mem_flatMap.mp member
  simp only [MemoryApp.childOps, List.mem_cons, List.not_mem_nil, or_false] at opMember
  rcases opMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    change e ∈ flatConstraints (Circuit.ops (Sponge.main _) _) at member <;>
    rw [Sponge.flatConstraints_eq] at member
  · exact R1CS.recipeConstraints_noFresh _ _ (absorbing_direct _ _ (stateBlocks_affine _ _)) e member
  · exact R1CS.recipeConstraints_noFresh _ _ (absorbing_direct _ _ (stateBlocks_affine _ _)) e member
  · exact R1CS.recipeConstraints_noFresh _ _
      (absorbing_direct _ _ (chainBlocks_affine _ _ _ opsLanes_words)) e member
  · exact R1CS.recipeConstraints_noFresh _ _
      (absorbing_direct _ _ (chainBlocks_affine _ _ _ (scanLanes_words _))) e member
  · exact R1CS.recipeConstraints_noFresh _ _
      (absorbing_direct _ _ (chainBlocks_affine _ _ _ (scanLanes_words _))) e member
  · exact R1CS.recipeConstraints_noFresh _ _ (absorbing_direct _ _ etaBlocks_affine) e member
  · exact R1CS.recipeConstraints_noFresh _ _ (permuting_direct _ _ etaState_affine) e member
  · exact R1CS.recipeConstraints_noFresh _ _ (permuting_direct _ _ squeeze1State_affine) e member
  · exact R1CS.recipeConstraints_noFresh _ _ (permuting_direct _ _ squeeze2State_affine) e member

theorem flatAssertions (l : List Circuit.Expr) : flatConstraints (l.map Op.assertZero) = l := by
  induction l with
  | nil => rfl
  | cons e rest ih =>
    change [e] ++ flatConstraints (rest.map Op.assertZero) = e :: rest
    rw [ih]
    rfl

/-- The (application plan two)'s R1CS lowering allocates at most one fresh column per
multiplication of the polynomial rows. -/
theorem freshCount_le :
    R1CS.totalFreshCount (flatConstraints (MemoryApp.opsAt plan (wires plan two) (start plan two))) ≤
      ((Rows.polyRows plan).map R1CS.mulCount).sum := by
  have wireAffine : ∀ k, R1CS.IsAffine (MemoryApp.wire plan (wires plan two) (start plan two) k) := wire_affine
  have laneAffine : ∀ {state : Sponge.EState}, Poseidon2.StateAffine state →
      ∀ (k : Fin 4) {target : Circuit.Expr}, R1CS.IsAffine target →
        R1CS.constraintFreshCount (MemoryApp.lane state k - target) = 0 :=
    fun affine k _ target => R1CS.constraintFreshCount_eq_zero_of_affine _
      (sub_affine (affine _) target)
  rw [MemoryApp.opsAt, flatConstraints_append, flatAssertions, R1CS.totalFreshCount_append,
    childConstraints_noFresh, Nat.zero_add, MemoryApp.assertions, R1CS.totalFreshCount_append]
  have lanes : R1CS.totalFreshCount (((((
      (List.finRange 4).map (fun k =>
        MemoryApp.lane (Sponge.output (MemoryApp.stateIn plan (wires plan two) (start plan two)) (start plan two)) k -
          (wires plan two).input (start plan two) k) ++
      (List.finRange 4).map (fun k =>
        MemoryApp.lane (Sponge.output (MemoryApp.stateOut plan (wires plan two) (start plan two))
          (MemoryApp.stateOutStart plan (wires plan two) (start plan two))) k - (wires plan two).output (start plan two) k)) ++
      (List.finRange 4).map (fun k =>
        MemoryApp.lane (Sponge.output (MemoryApp.chainOps plan (wires plan two) (start plan two))
          (MemoryApp.chainOpsStart plan (wires plan two) (start plan two))) k -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.carryOut (23 + k.val)))) ++
      (List.finRange 4).map (fun k =>
        MemoryApp.lane (Sponge.output (MemoryApp.chainInitial plan (wires plan two) (start plan two))
          (MemoryApp.chainInitialStart plan (wires plan two) (start plan two))) k -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.carryOut (27 + k.val)))) ++
      (List.finRange 4).map (fun k =>
        MemoryApp.lane (Sponge.output (MemoryApp.chainFinal plan (wires plan two) (start plan two))
          (MemoryApp.chainFinalStart plan (wires plan two) (start plan two))) k -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.carryOut (31 + k.val)))) ++
      [MemoryApp.lane (MemoryApp.etaState plan (wires plan two) (start plan two)) 0 -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.etaFresh 0),
        MemoryApp.lane (MemoryApp.squeeze1State plan (wires plan two) (start plan two)) 0 -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.etaFresh 1),
        MemoryApp.lane (MemoryApp.squeeze2State plan (wires plan two) (start plan two)) 0 -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.etaFresh 2),
        MemoryApp.lane (MemoryApp.squeeze3State plan (wires plan two) (start plan two)) 0 -
          MemoryApp.wire plan (wires plan two) (start plan two) (Words.etaFresh 3)]) = 0 := by
    apply R1CS.totalFreshCount_eq_zero_of_noFresh
    intro e member
    simp only [List.mem_append, List.mem_map, List.mem_finRange, true_and, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with (((((⟨k, rfl⟩ | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) | ⟨k, rfl⟩) |
      (rfl | rfl | rfl | rfl))
    · exact laneAffine (absorbing_output_affine _ _ (stateBlocks_affine _ _)) k (R1CS.isAffine_var _)
    · exact laneAffine (absorbing_output_affine _ _ (stateBlocks_affine _ _)) k (R1CS.isAffine_var _)
    · exact laneAffine (absorbing_output_affine _ _ (chainBlocks_affine _ _ _ opsLanes_words)) k
        (wireAffine _)
    · exact laneAffine (absorbing_output_affine _ _ (chainBlocks_affine _ _ _ (scanLanes_words _))) k
        (wireAffine _)
    · exact laneAffine (absorbing_output_affine _ _ (chainBlocks_affine _ _ _ (scanLanes_words _))) k
        (wireAffine _)
    · exact laneAffine etaState_affine 0 (wireAffine _)
    · exact laneAffine squeeze1State_affine 0 (wireAffine _)
    · exact laneAffine squeeze2State_affine 0 (wireAffine _)
    · exact laneAffine squeeze3State_affine 0 (wireAffine _)
  rw [lanes, Nat.zero_add]
  refine (totalFreshCount_le _).trans (le_of_eq ?_)
  rw [List.map_map]
  exact congrArg List.sum (List.map_congr_left fun e _ => sub_mulCount e)

/-! ### Local length -/

theorem transcriptChunks_length_le (blocks : List (List Circuit.Expr)) :
    (transcriptChunks blocks).length ≤ (blocks.map fun b => b.length + 1).sum := by
  induction blocks with
  | nil => simp [transcriptChunks]
  | cons b rest ih =>
    rw [transcriptChunks, List.flatMap_cons, List.length_append, ← transcriptChunks]
    simp only [Gadgets.Poseidon2.Hash.inputChunks, List.length_map, List.length_range, blockE,
      List.length_cons, List.map_cons, List.sum_cons, Spec.Poseidon2.rate]
    have chunks : (b.length + 1 + 12 - 1) / 12 ≤ b.length + 1 := by omega
    omega

theorem pack_length_le (l : List Circuit.Expr) : (Nebula.Expr.pack l).length ≤ l.length := by
  induction l using Nebula.Expr.pack.induct with
  | case1 => simp [Nebula.Expr.pack]
  | case2 l nonempty ih =>
    rw [Nebula.Expr.pack, dif_neg nonempty, List.length_cons]
    have positive : 0 < l.length := List.length_pos_of_ne_nil nonempty
    simp only [List.length_drop] at ih
    omega

theorem stateBlocks_size (app carry : ℕ → ℕ) :
    ((MemoryApp.stateBlocks plan (wires plan two) (start plan two) app carry).map fun b => b.length + 1).sum = 71 := by
  simp only [MemoryApp.stateBlocks, MemoryApp.wires, MemoryApp.textE, List.map_cons, List.map_nil,
    List.length_map, List.length_ofFn, List.sum_cons, List.sum_nil]
  decide

theorem chainBlocks_size (lane : Spec.Nebula.Lane) (previous : ℕ) (lanes : List Circuit.Expr) :
    ((MemoryApp.chainBlocks plan (wires plan two) (start plan two) lane previous lanes).map fun b => b.length + 1).sum ≤
      lanes.length + 39 := by
  have tag : (MemoryApp.textE (chainTag lane)).length = 31 := by
    cases lane <;> decide
  have packed := pack_length_le (lanes.map (MemoryApp.sub plan (wires plan two) (start plan two)))
  simp only [List.length_map] at packed
  simp only [MemoryApp.chainBlocks, MemoryApp.wires, List.map_cons, List.map_nil, List.length_cons,
    List.length_ofFn, List.sum_cons, List.sum_nil, tag]
  omega

theorem opsLanes_length : (MemoryApp.opsLanes plan).length = plan.bOps * plan.opWidth := by
  have each : ∀ j : Fin plan.bOps, (Sym.opLane plan j.val).length = plan.opWidth := fun j => by
    simp [Sym.opLane, Spec.Nebula.Plan.opWidth]
    omega
  simp only [MemoryApp.opsLanes, List.length_flatten, List.map_ofFn, Function.comp_def, each,
    List.ofFn_const, List.sum_replicate, smul_eq_mul]

theorem scanLanes_length (slotStart : ℕ → ℕ) :
    (MemoryApp.scanLanes plan slotStart).length = plan.bScan * plan.scanWidth := by
  have each : ∀ j : Fin plan.bScan, (Sym.scanLane plan (slotStart j.val)).length =
      plan.scanWidth := fun j => by
    simp [Sym.scanLane, Spec.Nebula.Plan.scanWidth]
    omega
  simp only [MemoryApp.scanLanes, List.length_flatten, List.map_ofFn, Function.comp_def, each,
    List.ofFn_const, List.sum_replicate, smul_eq_mul]

theorem span_le (blocks : List (List Circuit.Expr)) (bound : ℕ)
    (size : (blocks.map fun b => b.length + 1).sum ≤ bound) :
    MemoryApp.span blocks ≤ bound * 1096 :=
  Nat.mul_le_mul_right _ ((transcriptChunks_length_le blocks).trans size)

/-- A bound on the variables of the nine sponge children: chunks of 1,096
variables each. -/
def localBound (plan : Spec.Nebula.Plan) : ℕ :=
  (142 + (plan.bOps * plan.opWidth + 39) + 2 * (plan.bScan * plan.scanWidth + 39) + 7 + 3) * 1096

theorem localLength_le :
    localLength (MemoryApp.opsAt plan (wires plan two) (start plan two)) ≤ localBound plan := by
  have total := MemoryApp.localLength_opsAt plan (wires plan two) (start plan two)
  have s1 := span_le _ _ (le_of_eq (stateBlocks_size (plan := plan) (two := two) Words.appIn Words.carryIn))
  have s2 := span_le _ _ (le_of_eq (stateBlocks_size (plan := plan) (two := two) Words.appOut Words.carryOut))
  have s3 := span_le _ _ ((chainBlocks_size (plan := plan) (two := two) .ops 0 (MemoryApp.opsLanes plan)).trans
    (le_of_eq (by rw [opsLanes_length])))
  have s4 := span_le _ _ ((chainBlocks_size (plan := plan) (two := two) .mem 4 (MemoryApp.scanLanes plan fun j =>
    Words.initial plan j 0)).trans (le_of_eq (by rw [scanLanes_length])))
  have s5 := span_le _ _ ((chainBlocks_size (plan := plan) (two := two) .mem 8 (MemoryApp.scanLanes plan fun j =>
    Words.final plan j 0)).trans (le_of_eq (by rw [scanLanes_length])))
  have s6 : MemoryApp.etaChunkCount = 7 := by decide
  unfold MemoryApp.endOffset MemoryApp.squeeze3Start MemoryApp.squeeze2Start
    MemoryApp.squeeze1Start MemoryApp.etaStart MemoryApp.chainFinalStart
    MemoryApp.chainInitialStart MemoryApp.chainOpsStart MemoryApp.stateOutStart at total
  unfold localBound
  omega

theorem child_length (name : String) (child : Sponge.Interface) (childStart : ℕ) :
    (Sequence.childOp name (Sponge.circuit child) childStart).flatConstraints.length =
      (Sequence.childOp name (Sponge.circuit child) childStart).localLength := by
  rw [Sequence.childOp_localLength]
  change (flatConstraints (Circuit.ops (Sponge.main child) childStart)).length =
    localLength (Circuit.ops (Sponge.main child) childStart)
  rw [Sponge.rowCount_eq, Sponge.localLength_eq]

theorem flat_length_eq (ops : List Op)
    (each : ∀ op ∈ ops, op.flatConstraints.length = op.localLength) :
    (flatConstraints ops).length = localLength ops := by
  induction ops with
  | nil => rfl
  | cons op rest ih =>
    change (op.flatConstraints ++ flatConstraints rest).length = op.localLength + localLength rest
    rw [List.length_append, each op (by simp), ih fun o member => each o (by simp [member])]

theorem childOps_length :
    (flatConstraints (MemoryApp.childOps plan (wires plan two) (start plan two))).length =
      localLength (MemoryApp.childOps plan (wires plan two) (start plan two)) := by
  apply flat_length_eq
  intro op member
  simp only [MemoryApp.childOps, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> exact child_length _ _ _

theorem assertions_length :
    (MemoryApp.assertions plan (wires plan two) (start plan two)).length = 24 + (Rows.polyRows plan).length := by
  simp [MemoryApp.assertions]
  omega

theorem constraints_length :
    (flatConstraints (MemoryApp.opsAt plan (wires plan two) (start plan two))).length =
      localLength (MemoryApp.opsAt plan (wires plan two) (start plan two)) + 24 + (Rows.polyRows plan).length := by
  rw [MemoryApp.opsAt, flatConstraints_append, flatAssertions, List.length_append, childOps_length,
    assertions_length, Sequence.localLength_append, MemoryApp.assertions_localLength]
  omega

@[simp] theorem witnessWordCount : (application plan two).witnessWordCount = Words.count plan :=
  rfl

/-! ### Stage 1 size bounds from a plan's counts -/

theorem rows_le (plan : Spec.Nebula.Plan) (two : plan.bOps = 2) {mul polyRows rows : ℕ}
    (mulBound : ((Rows.polyRows plan).map R1CS.mulCount).sum ≤ mul)
    (lengthBound : (Rows.polyRows plan).length ≤ polyRows)
    (fits : mul + localBound plan + 24 + polyRows ≤ rows) :
    (PerApplicationPackage.applicationPlan (application plan two)).rowCount ≤ rows := by
  rw [PerApplicationPackage.applicationPlan, ApplicationPackage.productionPlan_rowCount]
  unfold ApplicationPackage.compiledRows
  rw [Rows.compileRowsTR_length, Rows.lowerConstraintsTR_eq, R1CS.lowerConstraints_rows_length,
    constraints_eq, R1CS.totalRowCount_eq_fresh_add_length, constraints_length]
  have := freshCount_le (plan := plan) (two := two)
  have := localLength_le (plan := plan) (two := two)
  omega

theorem columns_le (plan : Spec.Nebula.Plan) (two : plan.bOps = 2) {mul columns : ℕ}
    (mulBound : ((Rows.polyRows plan).map R1CS.mulCount).sum ≤ mul)
    (fits : Words.count plan + localBound plan + mul ≤ columns) :
    PerApplicationPackage.addedPrivateColumnCount (application plan two) ≤ columns := by
  rw [PerApplicationPackage.addedPrivateColumnCount, PerApplicationPackage.applicationPlan,
    ApplicationPackage.productionPlan_privateCount, witnessWordCount, operations_eq, constraints_eq]
  have := freshCount_le (plan := plan) (two := two)
  have := localLength_le (plan := plan) (two := two)
  omega

theorem carrier_le (plan : Spec.Nebula.Plan) (two : plan.bOps = 2) {mul words : ℕ}
    (mulBound : ((Rows.polyRows plan).map R1CS.mulCount).sum ≤ mul)
    (fits : Words.count plan + localBound plan + mul ≤ words) :
    (application plan two).witnessWordCount +
      ApplicationRetainedBlocks.localCount (application plan two) ≤ words := by
  unfold ApplicationRetainedBlocks.localCount ApplicationRetainedBlocks.sourceWidth
    ApplicationDirectSource.sourceWidth ApplicationPackage.r1csFreshStart
  rw [witnessWordCount, operations_eq, constraints_eq]
  have := freshCount_le (plan := plan) (two := two)
  have := localLength_le (plan := plan) (two := two)
  omega

/-- The row range `[first, end)` of each named assertion group
(`MemoryApp.assertionNames`), counted from the application's first package
row. The assertions are the last constraints of the circuit
(`MemoryApp.assertionNames_count`), and each constraint lowers to
`R1CS.constraintRowCount` consecutive rows. Conformance tests read it. -/
def namedRowRanges (plan : Spec.Nebula.Plan) (two : plan.bOps = 2) : List (String × ℕ × ℕ) :=
  let rows := (ApplicationPackage.constraints (application plan two)
    (ApplicationPackage.productionColumns (application plan two)) (start plan two)).map
      R1CS.constraintRowCount
  let names := MemoryApp.assertionNames plan
  let children := rows.length - (names.map Prod.snd).sum
  let step := fun (state : Array (String × ℕ × ℕ) × ℕ × List ℕ) (group : String × ℕ) =>
    let span := (state.2.2.take group.2).sum
    (state.1.push (group.1, state.2.1, state.2.1 + span), state.2.1 + span, state.2.2.drop group.2)
  (names.foldl step (#[], (rows.take children).sum, rows.drop children)).1.toList

end NightstreamFPrime.Export.Stage1.NebulaMemoryPackage
