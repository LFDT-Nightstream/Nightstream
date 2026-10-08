import NightstreamFPrime.Lifecycle.Nebula.MemoryEval

/-! Owns the variable support of the memory-application circuit's inputs: when
every witness wire reads only allowed variables, so does every block of every
sponge child. The child assumptions (variables below the call offset) are the
instance `allowed = (· < offset)`. It does not own the child-local support. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Plan) (i : AppInterface p) (offset : ℕ) (allowed : ℕ → Prop)

/-- Every witness wire reads only allowed variables. -/
def WiresSupported : Prop := ∀ index, (i.witness offset index).VarsSatisfy allowed

variable {p i offset allowed}

theorem wire_supported (wiresSupported : WiresSupported p i offset allowed) (k : ℕ) :
    (wire p i offset k).VarsSatisfy allowed := by
  unfold wire
  split
  · exact wiresSupported _
  · trivial

theorem sub_supported (wiresSupported : WiresSupported p i offset allowed) (e : Circuit.Expr) :
    (sub p i offset e).VarsSatisfy allowed :=
  Expr.varsSatisfy_subst _ allowed (wire_supported wiresSupported) e

theorem chunk_supported : ∀ {l : List Circuit.Expr}, (∀ x ∈ l, x.VarsSatisfy allowed) →
    (Expr.chunk l).VarsSatisfy allowed
  | [], _ => trivial
  | b :: bs, each => ⟨each b (by simp), trivial, chunk_supported fun x hx => each x (by simp [hx])⟩

theorem pack_supported {l : List Circuit.Expr} (each : ∀ x ∈ l, x.VarsSatisfy allowed) :
    ∀ e ∈ Expr.pack l, e.VarsSatisfy allowed := by
  induction l using Expr.pack.induct with
  | case1 => simp [Expr.pack]
  | case2 l nonempty ih =>
    rw [Expr.pack, dif_neg nonempty]
    intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact chunk_supported fun x hx => each x (List.mem_of_mem_take hx)
    · exact ih (fun x hx => each x (List.mem_of_mem_drop hx)) e member

/-- Every chunk of a transcript reads only allowed variables when every block
word does. -/
theorem transcriptChunks_supported {blocks : List (List Circuit.Expr)}
    (each : ∀ b ∈ blocks, ∀ e ∈ b, e.VarsSatisfy allowed) :
    ∀ chunk ∈ transcriptChunks blocks, ∀ e ∈ chunk, e.VarsSatisfy allowed := by
  intro chunk chunkMember e member
  simp only [transcriptChunks, List.mem_flatMap, Hash.inputChunks, List.mem_map] at chunkMember
  obtain ⟨b, blockMember, c, -, rfl⟩ := chunkMember
  have inBlock := List.mem_of_mem_drop (List.mem_of_mem_take member)
  rcases List.mem_cons.mp inBlock with rfl | inWords
  · trivial
  · exact each b blockMember e inWords

theorem wires_supported (wiresSupported : WiresSupported p i offset allowed) (f : ℕ → ℕ) (n : ℕ) :
    ∀ e ∈ wires p i offset f n, e.VarsSatisfy allowed := by
  intro e member
  obtain ⟨k, rfl⟩ := List.mem_ofFn.mp member
  exact wire_supported wiresSupported _

theorem textE_supported (text : String) : ∀ e ∈ textE text, e.VarsSatisfy allowed := by
  intro e member
  obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
  trivial

theorem stateBlocks_supported (wiresSupported : WiresSupported p i offset allowed) (app carry : ℕ → ℕ) :
    ∀ b ∈ stateBlocks p i offset app carry, ∀ e ∈ b, e.VarsSatisfy allowed := by
  intro b member
  simp only [stateBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_supported _
  · exact wires_supported wiresSupported _ _
  · exact wires_supported wiresSupported _ _

theorem chainBlocks_supported (wiresSupported : WiresSupported p i offset allowed) (lane : Lane)
    (previous : ℕ) (lanes : List Circuit.Expr) :
    ∀ b ∈ chainBlocks p i offset lane previous lanes, ∀ e ∈ b, e.VarsSatisfy allowed := by
  intro b member
  simp only [chainBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact textE_supported _
  · intro e member
    rcases List.mem_cons.mp member with rfl | member
    · exact wire_supported wiresSupported _
    · exact wires_supported wiresSupported _ _ e member
  · apply pack_supported
    intro x member
    obtain ⟨y, -, rfl⟩ := List.mem_map.mp member
    exact sub_supported wiresSupported y

theorem etaBlocks_supported (wiresSupported : WiresSupported p i offset allowed) :
    ∀ b ∈ etaBlocks p i offset, ∀ e ∈ b, e.VarsSatisfy allowed := by
  intro b member
  simp only [etaBlocks, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact textE_supported _
  · intro e member
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
    trivial
  · intro e member
    rw [List.mem_singleton.mp member]
    exact sub_supported wiresSupported _
  · intro e member
    simp only [List.mem_append] at member
    rcases member with (member | member) | member <;>
      exact wires_supported wiresSupported _ _ e member

/-! ### Below the call offset -/

theorem wiresSupported_below (inputs : Application.InputsBelow i offset) :
    WiresSupported p i offset (· < offset) :=
  fun index => (Circuit.Expr.varsSatisfy_lt_iff_varsBelow _ offset).2 (inputs.witness index)

theorem blocksBelow {blocks : List (List Circuit.Expr)} {start : ℕ} (le : offset ≤ start)
    (each : ∀ b ∈ blocks, ∀ e ∈ b, e.VarsSatisfy (· < offset)) :
    Hash.BlocksBelow start (transcriptChunks blocks) := by
  intro chunk chunkMember e member
  have supported := transcriptChunks_supported each chunk chunkMember e member
  exact Circuit.Expr.VarsBelow.mono e ((Circuit.Expr.varsSatisfy_lt_iff_varsBelow e offset).1 supported) le

theorem absorbing_assumptions {blocks : List (List Circuit.Expr)} {start : ℕ} (env : Env)
    (le : offset ≤ start) (each : ∀ b ∈ blocks, ∀ e ∈ b, e.VarsSatisfy (· < offset)) :
    Sponge.Assumptions (absorbing blocks) start env :=
  ⟨fun _ => trivial, blocksBelow le each⟩

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
