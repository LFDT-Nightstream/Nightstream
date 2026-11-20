import NightstreamFPrime.Export.BoundedDepth.FinalKey20All
import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge

/-! The final verifier over the Akita opening of `key20.aks` with the decoders `draw` of the
sparse challenge families (SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision of
2026-10-15, Section 3.2, (23), Definition 9.2 E1 and Theorem 10.17).

- `Opening.Drawn` has the hypotheses of `FinalKey20.sound` on an opening whose decoders are
  `draw` of the families of the Akita code: `(31, 10)` for the groups of `W` and `V` and for the
  level 1, `(31, 0)` for the key group and for the level 2, and the base families `(31, 0)` of
  the level 3 and `(31, 11)` of the levels 4 and 5 and of the terminal with the filters `keep₃`,
  `keep₄`, `keep₅` and `keepL`. It has no hypothesis on units, on `‖c‖_1` or on the slot
  probability. Its hypotheses on the challenges are `bounded₃` to `boundedL`: a member that
  passes the filter has `‖c‖_mul,2 ≤ Γ` (Akita Appendix H, Theorem H.1).
- `Opening.Drawn.sized` gives `Opening.Sized` with the slot probability `p = 1/q²`.
- `verifier_sound_drawn` is `verifier_sound` for a family of openings with `Opening.Drawn`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey20

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
    (And.left sized₃) level₃.batch (And.left (And.right sized₃)) norm₃.vars o.block₃ o.decode₃
    (Compose.read C) o.weight₃ norm₃.operator norm₃.limit (Tier.finish second)) level₃.tail

local notation "fourth" =>
  Tier.tensored (committedTraceSq (fold := level₄.fold) o.key₄ o.openKey₄ o.claim₄ level₄.vars
    (And.left sized₄) level₄.batch (And.left (And.right sized₄)) norm₄.vars o.block₄ o.decode₄
    (Compose.read C) o.weight₄ norm₄.operator norm₄.limit (Tier.finish third)) level₄.tail

local notation "fifth" =>
  Tier.tensored (committedTraceSq (fold := level₅.fold) o.key₅ o.openKey₅ o.claim₅ level₅.vars
    (And.left sized₅) level₅.batch (And.left (And.right sized₅)) norm₅.vars o.block₅ o.decode₅
    (Compose.read C) o.weight₅ norm₅.operator norm₅.limit (Tier.finish fourth)) level₅.tail
/-- The hypotheses of `FinalKey20.sound` on the Akita opening of `o`, when its decoders are the
decoders `draw` of the sparse challenge families of the Akita code. `decode` to `decodeL`,
`mass`, `peak` and `weight₁` to `weight₅`: the decoders and the numbers `ω` and `c_max` of `o`
are those of the families. `keep₃`, `keep₄`, `keep₅` and `keepL` are the filters of the levels 3
to 5 and of the terminal, and `bounded₃` to `boundedL` state that a member of the base family
that passes the filter has `‖c‖_mul,2 ≤ Γ` for `Γ` = 13, 18, 18 and 18 (Akita (23) and
Theorem H.1). The other fields are those of `Opening.Sized` that are not about a challenge
family. -/
structure Opening.Drawn (keep₃ : Rq level₃.shape.innerDegree → Prop)
    (keep₄ : Rq level₄.shape.innerDegree → Prop) (keep₅ : Rq level₅.shape.innerDegree → Prop)
    (keepL : Rq Key20.terminal.innerDegree → Prop) : Prop where
  laid : setup.Laid o.shape
  early : ∀ anchor g, g ≠ 2 → o.bound anchor g = o.bound (anchor.take 0) g
  injective : Function.Injective o.block
  decode : o.decode = Key20.rootDecode K
  mass : o.mass = Key20.rootMass
  peak : o.peak = Key20.rootPeak
  injective₁ : Function.Injective o.block₁
  decode₁ : o.decode₁ = draw (SparseFamily _ 31 10) fun _ => True
  weight₁ : o.weight₁ = 51
  injective₂ : Function.Injective o.block₂
  decode₂ : o.decode₂ = draw (SparseFamily _ 31 0) fun _ => True
  weight₂ : o.weight₂ = 31
  injective₃ : Function.Injective o.block₃
  decode₃ : o.decode₃ = draw (SparseFamily _ 31 0) keep₃
  weight₃ : o.weight₃ = 31
  bounded₃ : ∀ c ∈ SparseFamily _ 31 0, keep₃ c → MulBound norm₃.operator c
  injective₄ : Function.Injective o.block₄
  decode₄ : o.decode₄ = draw (SparseFamily _ 31 11) keep₄
  weight₄ : o.weight₄ = 53
  bounded₄ : ∀ c ∈ SparseFamily _ 31 11, keep₄ c → MulBound norm₄.operator c
  injective₅ : Function.Injective o.block₅
  decode₅ : o.decode₅ = draw (SparseFamily _ 31 11) keep₅
  weight₅ : o.weight₅ = 53
  bounded₅ : ∀ c ∈ SparseFamily _ 31 11, keep₅ c → MulBound norm₅.operator c
  next : Terminal.start o.terminal = Tier.finish fifth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  decodeL : Terminal.decode o.terminal = draw (SparseFamily _ 31 11) keepL
  boundedL : ∀ c ∈ SparseFamily _ 31 11, keepL c → MulBound normTerminal.operator c
  registered : ∀ anchor, o.honest.Opens o.factorBound o.sourceBound (o.key 0) (o.bound anchor 0)
    (keyBase (FinalVerifier.Setup.params S₀ o.shape))
  covers : o.factorBound * (root.fold 0).sourceBound (root.shape 0) + 2 * o.mass 0 * o.sourceBound ≤
    o.innerBound

variable {o} {keep₃ : Rq level₃.shape.innerDegree → Prop}
  {keep₄ : Rq level₄.shape.innerDegree → Prop} {keep₅ : Rq level₅.shape.innerDegree → Prop}
  {keepL : Rq Key20.terminal.innerDegree → Prop}

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- An opening with `Opening.Drawn` has `Opening.Sized` with the slot probability `p = 1/q²`:
the families have units as differences (`Key20.draw_units`), their members have the norms `ω`
and `c_max` (`Key20.draw_sparse`), and a decoder `draw` gives each challenge for at most one of
the `q²` slots (`Key20.slotChance_draw`). -/
theorem Opening.Drawn.sized
    (drawn : o.Drawn setup packageWidths step cap control read head keep₃ keep₄ keep₅ keepL) :
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
    exact Key20.draw_units 5 (by decide)
  sparse₁ := ⟨2, by
    rw [drawn.decode₁, drawn.weight₁]
    exact Key20.draw_sparse⟩
  injective₂ := drawn.injective₂
  units₂ := by
    rw [drawn.decode₂]
    exact Key20.draw_units 6 (by decide)
  sparse₂ := ⟨1, by
    rw [drawn.decode₂, drawn.weight₂]
    exact Key20.draw_sparse⟩
  injective₃ := drawn.injective₃
  units₃ := by
    rw [drawn.decode₃]
    exact Key20.draw_units 6 (by decide)
  sparse₃ := ⟨1, by
    rw [drawn.decode₃, drawn.weight₃]
    exact Key20.draw_sparse⟩
  bounded₃ challenge decoded := by
    rw [drawn.decode₃] at decoded
    exact drawn.bounded₃ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2
  injective₄ := drawn.injective₄
  units₄ := by
    rw [drawn.decode₄]
    exact Key20.draw_units 5 (by decide)
  sparse₄ := ⟨2, by
    rw [drawn.decode₄, drawn.weight₄]
    exact Key20.draw_sparse⟩
  bounded₄ challenge decoded := by
    rw [drawn.decode₄] at decoded
    exact drawn.bounded₄ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2
  injective₅ := drawn.injective₅
  units₅ := by
    rw [drawn.decode₅]
    exact Key20.draw_units 5 (by decide)
  sparse₅ := ⟨2, by
    rw [drawn.decode₅, drawn.weight₅]
    exact Key20.draw_sparse⟩
  bounded₅ challenge decoded := by
    rw [drawn.decode₅] at decoded
    exact drawn.bounded₅ _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2
  next := drawn.next
  inside := drawn.inside
  injectiveL := drawn.injectiveL
  unitsL := by
    unfold Terminal.family
    rw [drawn.decodeL]
    exact Key20.draw_units 5 (by decide)
  boundedL challenge decoded := by
    unfold Terminal.family at decoded
    rw [drawn.decodeL] at decoded
    exact drawn.boundedL _ (draw_mem decoded.choose_spec).1 (draw_mem decoded.choose_spec).2
  chance g := by
    rw [drawn.decode]
    exact Key20.slotChance_draw _ _
  chance₁ := by
    rw [drawn.decode₁]
    exact Key20.slotChance_draw _ _
  chance₂ := by
    rw [drawn.decode₂]
    exact Key20.slotChance_draw _ _
  chance₃ := by
    rw [drawn.decode₃]
    exact Key20.slotChance_draw _ _
  chance₄ := by
    rw [drawn.decode₄]
    exact Key20.slotChance_draw _ _
  chance₅ := by
    rw [drawn.decode₅]
    exact Key20.slotChance_draw _ _
  chanceL := by
    rw [drawn.decodeL]
    exact Key20.slotChance_draw _ _
  registered := drawn.registered
  covers := drawn.covers

end One

section All

variable {ι : Type} [Fintype ι] (kind : Wire → Option ι)
  (family : ι → Opening logicalWidth publicFits Wire Coord total)
  (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire)) [Finite Wire]

/-- Soundness of the final verifier with Akita openings whose decoders are the decoders `draw`
of the sparse challenge families: `verifier_sound` with `p = 1/q²`. For each prover with at most
`Q` queries to the oracle, the probability over the table of the oracle that `finalAccepts`
accepts and no proof with witnesses satisfies `Verifies` for the statement of the first message
is at most `shapesBound` for `p = 1/q²`. `drawn`: the hypotheses `Opening.Drawn` on the opening
of each shape, for filters `keep₃` to `keepL`. The other hypotheses are those of
`verifier_sound`. -/
theorem verifier_sound_drawn
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    {keep₃ : ι → Rq level₃.shape.innerDegree → Prop}
    {keep₄ : ι → Rq level₄.shape.innerDegree → Prop}
    {keep₅ : ι → Rq level₅.shape.innerDegree → Prop}
    {keepL : ι → Rq Key20.terminal.innerDegree → Prop}
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head
      (keep₃ index) (keep₄ index) (keep₅ index) (keepL index))
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

end All

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
