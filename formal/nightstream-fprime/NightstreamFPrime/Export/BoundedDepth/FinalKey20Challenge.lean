import NightstreamFPrime.Export.BoundedDepth.FinalKey20History
import NightstreamFPrime.Export.BoundedDepth.FinalMemory
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening

/-! The final verifier over the Akita opening of `key20.aks` with the fold decoders of the pinned
Akita code on the slots `Replay.CodeSlot K` (SPEC.md, Algorithm 10; Akita, ePrint 2026/1983,
revision of 2026-11-23, Section 3.2, (23), Definition 9.2 E1 and Theorem 10.17).

- `Opening.Drawn` has the hypotheses of `FinalKey20.sound` on an opening whose decoders are the
  decoders of the code of the families of the Akita code: `Replay.CodeSlot.open` of `(31, 10)`
  for the groups of `W` and `V` and for the level 1 and of `(31, 0)` for the key group and for
  the level 2, and `Replay.CodeSlot.kept128` of the base family `(31, 0)` of the level 3 and
  `Replay.CodeSlot.kept64` of `(31, 11)` of the levels 4 and 5 and of the terminal with the
  filter `TrigTable.codeAccepts` of the code. It has no hypothesis on units, on `‖c‖_1`, on the
  operator bound or on the slot probability: the filter of the code accepts only elements with
  `‖c‖_mul,2 ≤ Γ` (`TrigTable.filtered_code`; Akita Appendix H, Theorem H.1).
- `Opening.Drawn.sized` gives `Opening.Sized` with the slot probability `p = 1/q²`, an upper
  bound of the probability `2^-K` or `1/|Kept G|` of each decoder of the code, and
  `Opening.Drawn.decodes` gives a fold challenge for each slot at each level.
- `verifier_sound_drawn` is `verifier_sound` for a family of openings with `Opening.Drawn`.
- `history_sound_drawn` and `history_knowledge_drawn` are `history_sound` and
  `history_knowledge` for such a family.
- `memory_sound_drawn` is `FinalMemory.memory_sound` for such a family (SPEC.md Algorithm 8;
  `SECURITY_MODEL.md`, adversary 5). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey20

open NightstreamFPrime.Spec.Compression.Akita.Replay (CodeSlot)

open scoped BigOperators
open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness
open NightstreamFPrime.Export.BoundedDepth.FinalVerifier (Mode Claims Premises)
open NightstreamFPrime.Export.BoundedDepth.FinalAkita
open Key20 (root level₁ level₂ level₃ level₄ level₅ norm₃ norm₄ norm₅ normTerminal rootVars
  rootBatch quotientDigits root_sized root_room level₁_sized sized₂ sized₃ sized₄ sized₅ quarters)

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}

variable (setup : FinalVerifier.Setup mode logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)
  {Wire Coord : Type} {total : ℕ} (read : Reader Wire)
  (head : Wire → First logicalWidth publicFits)
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

section One

variable (o : Opening logicalWidth publicFits Wire Coord total)

local notation "S₀" => laidSetup setup o.shape

local notation "C" => compose S₀ packageWidths step cap control o.shape read head o.bound

local notation "vf" => FinalVerifier.Setup.verifier S₀ o.shape

local notation "zeroth" =>
  liftBatch o.β o.key o.openKey o.claim rootVars (And.left root_sized) 8 quotientDigits root_room
    rootBatch (And.left (And.right root_sized)) o.block o.decode (Compose.read C) o.mass
    (Compose.root C vf + 1)

local notation "first" =>
  Tier.tensored (liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ level₁.vars (And.left level₁_sized)
    quotientDigits (And.left (And.right level₁_sized)) level₁.batch
    (And.left (And.right (And.right level₁_sized))) o.block₁ o.decode₁ (Compose.read C) o.weight₁
    (Tier.finish zeroth)) level₁.tail

local notation "second" =>
  Tier.tensored (committedTrace (fold := level₂.fold) o.key₂ o.openKey₂ o.claim₂ level₂.vars
    (And.left sized₂) level₂.batch (And.left (And.right sized₂)) o.block₂ o.decode₂
    (Compose.read C) o.weight₂ (Tier.finish first)) level₂.tail

local notation "third" =>
  Tier.tensored (committedTraceSq (fold := level₃.fold) o.key₃ o.openKey₃ o.claim₃ level₃.vars
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) o.block₃ o.decode₃
    (Compose.read C) o.weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) o.key₄ o.openKey₄ o.claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) o.block₄ o.decode₄
    (Compose.read C) o.weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) o.key₅ o.openKey₅ o.claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) o.block₅ o.decode₅
    (Compose.read C) o.weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail
/-- The hypotheses of `FinalKey20.sound` on the Akita opening of `o`, when its decoders are the
decoders of the code of the sparse challenge families of the Akita code. `decode` to `decodeL`,
`mass`, `peak` and `weight₁` to `weight₅`: the decoders and the numbers `ω` and `c_max` of `o`
are those of the families, with the filter `TrigTable.codeAccepts` of the code for `Γ` = 13,
18, 18 and 18 at the levels 3 to 5 and at the terminal (Akita (23)). `kept`: `2^128` members or
more of each filtered family pass the filter (Akita Table 12). The other fields are those of
`Opening.Sized` that are not about a challenge family. -/
structure Opening.Drawn : Prop where
  laid : setup.Laid o.shape
  early : ∀ anchor g, g ≠ 2 → o.bound anchor g = o.bound (anchor.take 0) g
  injective : Function.Injective o.block
  decode : o.decode = Key20.rootDecode K
  mass : o.mass = Key20.rootMass
  peak : o.peak = Key20.rootPeak
  injective₁ : Function.Injective o.block₁
  decode₁ : o.decode₁ = Replay.CodeSlot.open Key20.open64 Key20.open64_fits
  weight₁ : o.weight₁ = 51
  injective₂ : Function.Injective o.block₂
  decode₂ : o.decode₂ = Replay.CodeSlot.open Key20.open128 Key20.open128_fits
  weight₂ : o.weight₂ = 31
  injective₃ : Function.Injective o.block₃
  decode₃ : o.decode₃ = Replay.CodeSlot.kept128
  weight₃ : o.weight₃ = 31
  injective₄ : Function.Injective o.block₄
  decode₄ : o.decode₄ = Replay.CodeSlot.kept64
  weight₄ : o.weight₄ = 53
  injective₅ : Function.Injective o.block₅
  decode₅ : o.decode₅ = Replay.CodeSlot.kept64
  weight₅ : o.weight₅ = 53
  next : Terminal.start o.terminal = Tier.finish fifth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  decodeL : Terminal.decode o.terminal = Replay.CodeSlot.kept64
  registered : ∀ anchor, o.honest.Opens o.factorBound o.sourceBound (o.key 0) (o.bound anchor 0)
    (keyBase (FinalVerifier.Setup.params S₀ o.shape))
  covers : o.factorBound * (root.fold 0).sourceBound (root.shape 0) + 2 * o.mass 0 * o.sourceBound ≤
    (root.fold 0).innerBound (root.shape 0)
  kept : Key20.Kept

variable {o}

/-- The condition `covers` of `Opening.Drawn` holds for the factor bound 1, the source bound `b/2`
of the key group and `ω = 31`: the norm `β̄_∞ + κ̄_1·b/2 = 293887` is at most `η_A = 32505732`
(`Descriptor.key20_covers`). -/
theorem covers_one : 1 * (root.fold 0).sourceBound (root.shape 0) +
    2 * Key20.rootMass 0 * ((root.shape 0).sourceBase / 2) ≤
      (root.fold 0).innerBound (root.shape 0) := by
  rw [Nat.one_mul]
  exact Descriptor.key20_covers

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- With `Opening.Drawn`, each slot gives a fold challenge of each group of the root, of
the levels 1 to 5 and of the terminal, so the conditions `decodes` of the verifier hold for each
table of the oracle on keys. -/
theorem Opening.Drawn.decodes
    (drawn : o.Drawn setup packageWidths step cap control read head) (slot : CodeSlot K) :
    (∀ g, (o.decode g slot).isSome) ∧ (o.decode₁ slot).isSome ∧ (o.decode₂ slot).isSome ∧
      (o.decode₃ slot).isSome ∧ (o.decode₄ slot).isSome ∧ (o.decode₅ slot).isSome ∧
      (Terminal.decode o.terminal slot).isSome := by
  rw [drawn.decode, drawn.decode₁, drawn.decode₂, drawn.decode₃, drawn.decode₄, drawn.decode₅,
    drawn.decodeL]
  exact ⟨fun g => Key20.rootDecode_isSome g slot, Replay.open_isSome _ rfl (by decide) slot,
    Replay.open_isSome _ rfl (by decide) slot, Replay.kept128_isSome drawn.kept slot,
    Replay.kept64_isSome drawn.kept slot, Replay.kept64_isSome drawn.kept slot,
    Replay.kept64_isSome drawn.kept slot⟩

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- An opening with `Opening.Drawn` has `Opening.Sized` with the slot probability `p = 1/q²`:
the families have units as differences (`Key20.units_of_mem`), their members have the norms `ω`
and `c_max` (`Key20.sparse_of_mem`), the slot probability of each decoder of the code is at most
`2^-128 ≤ 1/q²` (`Replay.slotChance_open_field`, `Replay.slotChance_kept64_field`,
`Replay.slotChance_kept128_field`), and the filter of the code gives the operator bound
(`TrigTable.filtered_code`). -/
theorem Opening.Drawn.sized
    (drawn : o.Drawn setup packageWidths step cap control read head) :
    o.Sized setup packageWidths step cap control read head (1 / (fieldModulus : ℝ) ^ 2) where
  laid := drawn.laid
  early := drawn.early
  injective := drawn.injective
  units g := by
    rw [drawn.decode]
    exact Key20.rootDecode_units g
  sparse g challenge decoded := by
    rw [drawn.decode] at decoded
    rw [drawn.mass, drawn.peak]
    exact Key20.rootDecode_sparse g challenge decoded
  injective₁ := drawn.injective₁
  units₁ := by
    rw [drawn.decode₁]
    exact Key20.units_of_mem 5 (by decide) fun decoded => (Replay.open_mem decoded).1
  sparse₁ := ⟨2, by
    rw [drawn.decode₁, drawn.weight₁]
    exact Key20.sparse_of_mem (ones := 31) (twos := 10)
      fun decoded => (Replay.open_mem decoded).1⟩
  injective₂ := drawn.injective₂
  units₂ := by
    rw [drawn.decode₂]
    exact Key20.units_of_mem 6 (by decide) fun decoded => (Replay.open_mem decoded).1
  sparse₂ := ⟨1, by
    rw [drawn.decode₂, drawn.weight₂]
    exact Key20.sparse_of_mem (ones := 31) (twos := 0)
      fun decoded => (Replay.open_mem decoded).1⟩
  injective₃ := drawn.injective₃
  units₃ := by
    rw [drawn.decode₃]
    exact Key20.units_of_mem 6 (by decide) fun decoded => (Replay.kept128_mem decoded).1
  sparse₃ := ⟨1, by
    rw [drawn.decode₃, drawn.weight₃]
    exact Key20.sparse_of_mem (ones := 31) (twos := 0)
      fun decoded => (Replay.kept128_mem decoded).1⟩
  bounded₃ challenge decoded := by
    rw [drawn.decode₃] at decoded
    exact (TrigTable.filtered_code (by decide) _).mulBound (by decide)
      (Replay.kept128_mem decoded.choose_spec).2
  injective₄ := drawn.injective₄
  units₄ := by
    rw [drawn.decode₄]
    exact Key20.units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1
  sparse₄ := ⟨2, by
    rw [drawn.decode₄, drawn.weight₄]
    exact Key20.sparse_of_mem (ones := 31) (twos := 11)
      fun decoded => (Replay.kept64_mem decoded).1⟩
  bounded₄ challenge decoded := by
    rw [drawn.decode₄] at decoded
    exact (TrigTable.filtered_code (by decide) _).mulBound (by decide)
      (Replay.kept64_mem decoded.choose_spec).2
  injective₅ := drawn.injective₅
  units₅ := by
    rw [drawn.decode₅]
    exact Key20.units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1
  sparse₅ := ⟨2, by
    rw [drawn.decode₅, drawn.weight₅]
    exact Key20.sparse_of_mem (ones := 31) (twos := 11)
      fun decoded => (Replay.kept64_mem decoded).1⟩
  bounded₅ challenge decoded := by
    rw [drawn.decode₅] at decoded
    exact (TrigTable.filtered_code (by decide) _).mulBound (by decide)
      (Replay.kept64_mem decoded.choose_spec).2
  next := drawn.next
  inside := drawn.inside
  injectiveL := drawn.injectiveL
  unitsL := by
    unfold Terminal.family
    rw [drawn.decodeL]
    exact Key20.units_of_mem 5 (by decide) fun decoded => (Replay.kept64_mem decoded).1
  boundedL challenge decoded := by
    unfold Terminal.family at decoded
    rw [drawn.decodeL] at decoded
    exact (TrigTable.filtered_code (by decide) _).mulBound (by decide)
      (Replay.kept64_mem decoded.choose_spec).2
  chance g := by
    rw [drawn.decode]
    unfold Key20.rootDecode
    exact Replay.slotChance_open_field _ _ (Key20.rootGroup_wide g)
  chance₁ := by
    rw [drawn.decode₁]
    exact Replay.slotChance_open_field _ _ Key20.open64_bits.ge
  chance₂ := by
    rw [drawn.decode₂]
    exact Replay.slotChance_open_field _ _ ((by decide : 128 ≤ 129).trans Key20.open128_bits.ge)
  chance₃ := by
    rw [drawn.decode₃]
    exact Replay.slotChance_kept128_field drawn.kept
  chance₄ := by
    rw [drawn.decode₄]
    exact Replay.slotChance_kept64_field drawn.kept
  chance₅ := by
    rw [drawn.decode₅]
    exact Replay.slotChance_kept64_field drawn.kept
  chanceL := by
    rw [drawn.decodeL]
    exact Replay.slotChance_kept64_field drawn.kept
  registered := drawn.registered
  covers := drawn.covers

end One

section All

variable {ι : Type} [Fintype ι] (kind : Wire → Option ι)
  (family : ι → Opening logicalWidth publicFits Wire Coord total)
  (prover : OracleComp (Address Wire total) (Option Coord → CodeSlot K) (ℕ → Wire))
    [Finite Wire]

/-- Soundness of the final verifier with Akita openings whose decoders are the decoders of the
code of the sparse challenge families: `verifier_sound` with `p = 1/q²`. For each prover with at most
`Q` queries to the oracle, the probability over the table of the oracle that `finalAccepts`
accepts and no proof with witnesses satisfies `Verifies` for the statement of the first message
is at most `shapesBound` for `p = 1/q²`. `drawn`: the hypotheses `Opening.Drawn` on the opening
of each shape. The other hypotheses are those of `verifier_sound`. -/
theorem verifier_sound_drawn
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table,
        (if finalAccepts setup complete.packageWidth step cap control read head kind family
              (played prover table) ∧
            ¬ ∃ proof, Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
              control (head (played prover table).first).statement proof
          then (1 : ℝ) else 0) ≤
      shapesBound setup complete.packageWidth step cap control read head family prover
        (1 / (fieldModulus : ℝ) ^ 2) queries budget :=
  verifier_sound setup step cap control read head kind family prover exports complete rows blocks
    coords (fun index => (drawn index).sized) bounded positive

variable (budget : ℕ) {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
  {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) →
    Prop}
  {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
    (IvcWalk.Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary :
      HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
    HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary →
      Prop}
  {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
    (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
  (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
    (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
  (complete : StepExtend.Complete exports)

local notation "ivc" =>
  adversary setup complete.packageWidth step cap control read head kind family prover budget

/-- The final verifier with Akita openings against a false history, for openings whose decoders
are the decoders of the code of the sparse challenge families: `history_sound` with `p = 1/q²`. For
each prover with at most `Q` queries to the oracle, the probability of a false history is at most
`shapesBound` for `p = 1/q²` plus `IvcSecurity.walkBound` of the IVC adversary `adversary`.
`drawn`: the hypotheses `Opening.Drawn` on the opening of each shape. The other hypotheses are
those of `history_sound`. -/
theorem history_sound_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head)
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step cap control read head kind family
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((ivc).tape.toOuterMeasure {sample |
        FalseHistory setup complete.packageWidth step cap control read head kind family prover
          exports.program sample.1}).toReal ≤
      shapesBound setup complete.packageWidth step cap control read head family prover
          (1 / (fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth :=
  history_sound setup step cap control read head kind family prover budget exports assumption
    closed complete rows keysShaped blocks coords (fun index => (drawn index).sized) bounded
    positive admitted depth depthBound

/-- The knowledge form of `history_sound_drawn`: `history_knowledge` with `p = 1/q²`. The
probability that the final verifier accepts and the witnesses that the reverse extractor returns
are not a history of the accepted statement is at most `shapesBound` for `p = 1/q²` plus
`IvcSecurity.walkBound`. -/
theorem history_knowledge_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head)
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step cap control read head kind family
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((HyperNova.ReverseStages.reverseStages assumption closed (IvcSecurity.walkAdversary ivc)
          admitted depth).1.tape.toOuterMeasure {tape |
        AdviceFailure setup step cap control read head kind family prover budget exports
          assumption closed complete admitted depth tape}).toReal ≤
      shapesBound setup complete.packageWidth step cap control read head family prover
          (1 / (fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth :=
  history_knowledge setup step cap control read head kind family prover budget exports assumption
    closed complete rows keysShaped blocks coords (fun index => (drawn index).sized) bounded
    positive admitted depth depthBound

variable {p : Nebula.Plan} (app : Nebula.PortApp p.toGeometry)
  (claim : (Address Wire total → Option Coord → CodeSlot K) →
    PMF (Lifecycle.Nebula.EndToEnd.MemoryStatement app.width))

local notation "verifiers" => fun index =>
  Opening.accepts setup complete.packageWidth step cap control read head (family index)

local notation "extracted" => fun index =>
  Opening.proof setup complete.packageWidth step cap control read head (family index) prover

local notation "memory" =>
  FinalMemory.adversary head kind verifiers extracted prover budget app claim

local notation:max "claimedAt" admitted:max =>
  FinalMemory.claimed setup step head kind verifiers extracted prover budget app exports assumption
    closed claim admitted (p.sMax * p.n)

/-- The memory verifier with Akita openings whose decoders are the decoders of the code of the
sparse challenge families, in the form of `Claims.Final.memory_sound` (`FinalMemory.memory_sound` with
`Opening.knowledge` at `p = 1/q²`). For each prover with at most `Q` queries to the oracle that
names a memory statement with the law `claim` for each table, over the walk's `S_max · N`
stages, the probability that the final verifier accepts `(i, z0, zi)`, the checks of SPEC.md
Algorithm 8 hold, and the decoded history does not attest the memory statement is at most
`shapesBound` for `p = 1/q²` plus the terms of `NebulaSecurity.checked_le` for the IVC adversary
`FinalMemory.adversary`. `drawn`: the hypotheses `Opening.Drawn` on the opening of each shape.
The other hypotheses are those of `history_sound_drawn` and `NebulaFinal.soundness`. -/
theorem memory_sound_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head)
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary memory)))
    (valid : p.Valid) (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    {memoryError : NebulaSecurity.ClaimedStage app (logicalWidth := logicalWidth)
      (publicFits := publicFits) → ℝ}
    (round : NebulaSecurity.MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted
      memoryError) :
    ((HyperNova.ReverseStages.reverseStages assumption closed (IvcSecurity.walkAdversary memory)
          admitted (p.sMax * p.n)).1.tape.toOuterMeasure {tape |
        FinalMemory.FalseMemory setup step head kind verifiers prover app exports
          (FinalMemory.tapeOf setup step head kind verifiers extracted prover budget app exports
            assumption closed claim admitted (p.sMax * p.n) tape).1.1
          (FinalMemory.sampleOf setup step head kind verifiers extracted prover budget app exports
            assumption closed claim admitted (p.sMax * p.n) tape)}).toReal ≤
      shapesBound setup complete.packageWidth step cap control read head family prover
          (1 / (fieldModulus : ℝ) ^ 2) queries budget +
        (IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed memory
            admitted (p.sMax * p.n) +
          ((claimedAt admitted).law.toOuterMeasure
            {sample | (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys
              exports).StateLinkCollision app sample}).toReal +
          ((claimedAt admitted).law.toOuterMeasure
            {sample | (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys
              exports).ChainCollision app sample}).toReal +
          memoryError (claimedAt admitted)) :=
  FinalMemory.memory_sound setup step cap control head kind _ _ prover budget app exports
    assumption closed claim admitted complete
    (fun index =>
      (family index).knowledge setup complete.packageWidth step cap control read head prover
        (FinalRows.premises_of_rows setup exports complete rows) blocks coords
        (drawn index).sized bounded positive)
    valid keysShaped programEq round

end All

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
