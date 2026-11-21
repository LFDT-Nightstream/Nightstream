import NightstreamFPrime.Export.BoundedDepth.FinalKey20
import NightstreamFPrime.Export.BoundedDepth.FinalRows

/-! The final verifier over the Akita opening of the schedule file
`crates/ns-compress/artifacts/key20.aks` for each claim shape and for the proof of zero steps
(SPEC.md, Algorithm 10; Akita, ePrint 2026/1983, revision of 2026-10-15, Theorem 10.17).

`FinalKey20.sound` is for the final proofs of one claim shape: one `pc` and one depth for each
slot. The layout of the argument, the zeros of the opening point of `V` and the key polynomial
are functions of the shape. Here a finite family of shapes has one `Opening` for each shape, and
the bound is the sum of the bounds of the shapes. The factor is the number of the shapes: at most
the pairs of `pc` and depths that pass the layout of step 5.

- `laidSetup` is the setup with the zeros of the layout of a shape, and `laidSetup_use` gives
  `Use` for it.
- `Opening` has the data of the Akita opening for one shape, `Opening.accepts` is the verifier
  `towerAccepts` of `sound` for it, and `Opening.Sized` has the hypotheses of `sound` on it.
- `Opening.proof` is the proof with witnesses that the table `W` of an output of the extractor
  gives, and `Opening.knowledge` bounds the event "the verifier accepts and `Verifies` rejects
  this proof".
- `finalAccepts` is the final verifier: the check of the proof of zero steps, or
  `Opening.accepts` for the shape that the first message names. `finalProof` is the proof of the
  extractor for it, and `knowledge_all` is the bound for all shapes. The proof of zero steps has
  no draw, and `Verifies` accepts its proof (`FinalVerifier.verifies_of_bottom`).
- `sound_all` is the bound for "the final verifier accepts and no proof with witnesses satisfies
  `Verifies`", and `verifier_sound` is `sound_all` for a setup whose rows are the rows of the
  packages (`FinalRows.premises_of_rows`). -/

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

section Transcript

variable {Wire Reply : Type} [Nonempty Reply] {total : ℕ}

/-- Each stage input of a transcript has the first message of the proof. -/
theorem path_first (accept : Prefix Wire Reply total → Prop) (proof : ℕ → Wire)
    (table : Prefix Wire Reply total → Reply) :
    ∀ level, ((transcript total accept).path proof table level).first = proof 0
  | 0 => rfl
  | level + 1 => by
      have before := path_first accept proof table level
      show (if room : ((transcript total accept).path proof table level).rest.length < total
        then _ else (transcript total accept).path proof table level).first = _
      split_ifs
      · exact before
      · exact before

/-- The stage inputs of a transcript are the same for each predicate of the verifier. -/
theorem path_accept (accept other : Prefix Wire Reply total → Prop) (proof : ℕ → Wire)
    (table : Prefix Wire Reply total → Reply) :
    ∀ level, (transcript total accept).path proof table level =
      (transcript total other).path proof table level
  | 0 => rfl
  | level + 1 => by
      rw [Protocol.path, Protocol.path, path_accept accept other proof table level]
      rfl

/-- The complete transcript of a prover for a table of the oracle. -/
noncomputable def played (prover : OracleComp (Prefix Wire Reply total) Reply (ℕ → Wire))
    (table : Prefix Wire Reply total → Reply) : Prefix Wire Reply total :=
  (transcript total fun _ => True).final prover table

theorem played_eq (accept : Prefix Wire Reply total → Prop)
    (prover : OracleComp (Prefix Wire Reply total) Reply (ℕ → Wire))
    (table : Prefix Wire Reply total → Reply) :
    (transcript total accept).final prover table = played prover table :=
  path_accept accept _ (prover.run table) table total

theorem played_first (prover : OracleComp (Prefix Wire Reply total) Reply (ℕ → Wire))
    (table : Prefix Wire Reply total → Reply) :
    (played prover table).first = prover.run table 0 :=
  path_first _ (prover.run table) table total

end Transcript

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}

/-- The setup with the zeros of the opening point of `V` for the layout of `shape`
(`laidZeros`). The verifier computes these zeros from the layout of the claims of a proof. -/
def laidSetup (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : Claims logicalWidth publicFits) : FinalVerifier.Setup mode logicalWidth publicFits :=
  { setup with zeros := laidZeros setup shape }

/-- The conditions `Use` for the setup `laidSetup` of a shape that passes the layout of step 5,
under a key polynomial of 512 blocks. -/
theorem laidSetup_use (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : Claims logicalWidth publicFits) (laid : setup.Laid shape)
    (blocks : setup.keyBlocks = 512) : Use (laidSetup setup shape) shape :=
  ⟨laid, blocks, rfl⟩

/-- The data of the Akita opening of `key20.aks` for the final proofs of one claim shape: the
parameters of `FinalKey20.sound`. `shape` has `pc` and the depths. `bound anchor g` is the
commitment of the group `g` that a stage input fixes. `honest` is a certified decommitment of the
key commitment, with the bounds `factorBound`, `sourceBound` and `innerBound` of
`FinalAkita.final_extract_sound`. The other fields are the keys, the claims, the decoders of the
fold challenges and the readers of the messages of the root, of the levels 1 to 5 and of the
terminal. -/
structure Opening (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) (Wire Coord : Type) (total : ℕ) where
  shape : Claims logicalWidth publicFits
  bound : Prefix Wire (Option Coord → K) total → (g : Fin 3) → (root.shape g).Payload
  β : ∀ g, Module.Basis (Fin (root.planes g)) Zq K
  key : (g : Fin 3) → Key (root.shape g)
  openKey : root.OpenKey
  claim : Wire → root.Claim K
  block : root.Block → Coord
  decode : (g : Fin 3) → K → Option (Rq (root.carrierDegree g))
  mass : Fin 3 → ℕ
  peak : Fin 3 → ℕ
  β₁ : Module.Basis (Fin level₁.opening.planes) Zq K
  key₁ : Key level₁.shape
  openKey₁ : OpeningKey level₁.opening
  claim₁ : Wire → Claim level₁.opening K
  block₁ : Fin level₁.shape.blocks → Coord
  decode₁ : K → Option (Rq level₁.opening.carrierDegree)
  weight₁ : ℕ
  key₂ : Key level₂.shape
  openKey₂ : OpeningKey level₂.opening
  claim₂ : Wire → TraceClaim level₂.opening K
  block₂ : Fin level₂.shape.blocks → Coord
  decode₂ : K → Option (Rq level₂.shape.innerDegree)
  weight₂ : ℕ
  key₃ : Key level₃.shape
  openKey₃ : OpeningKey level₃.opening
  claim₃ : Wire → TraceClaim level₃.opening K
  block₃ : Fin level₃.shape.blocks → Coord
  decode₃ : K → Option (Rq level₃.shape.innerDegree)
  weight₃ : ℕ
  key₄ : Key level₄.shape
  openKey₄ : OpeningKey level₄.opening
  claim₄ : Wire → TraceClaim level₄.opening K
  block₄ : Fin level₄.shape.blocks → Coord
  decode₄ : K → Option (Rq level₄.shape.innerDegree)
  weight₄ : ℕ
  key₅ : Key level₅.shape
  openKey₅ : OpeningKey level₅.opening
  claim₅ : Wire → TraceClaim level₅.opening K
  block₅ : Fin level₅.shape.blocks → Coord
  decode₅ : K → Option (Rq level₅.shape.innerDegree)
  weight₅ : ℕ
  terminal : Terminal Key20.terminal K Wire Coord K total
    (.euclidean normTerminal.operator normTerminal.limit)
  commit : Wire → (g : Fin 3) → (root.shape g).Payload
  value : Wire → Fin 3 → K
  place : Wire → Fin 3 → List K
  commit₁ : Wire → level₁.shape.Payload
  commit₂ : Wire → level₂.shape.Payload
  commit₃ : Wire → level₃.shape.Payload
  commit₄ : Wire → level₄.shape.Payload
  commit₅ : Wire → level₅.shape.Payload
  stateOf : Wire → Fin Key20.terminal.blocks → Fin Key20.terminal.innerRank →
    Rq Key20.terminal.innerDegree
  honest : Certified (root.shape 0)
  factorBound : ℕ
  sourceBound : ℕ
  innerBound : ℕ

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

/-- The final verifier for the proofs with steps of the shape of `o`: the verifier
`towerAccepts` of `FinalKey20.sound` under the setup `laidSetup`. -/
def Opening.accepts : Prefix Wire (Option Coord → K) total → Prop :=
  towerAccepts (weight₃ := o.weight₃) (weight₄ := o.weight₄) (weight₅ := o.weight₅) C vf
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    zeroth first second third fourth fifth o.terminal o.commit o.value o.place (And.left quarters)
    (And.left (And.right quarters)) (And.left (And.right (And.right quarters)))
    (And.left (And.right (And.right (And.right quarters))))
    (And.right (And.right (And.right (And.right quarters)))) o.commit₁ o.commit₂ o.commit₃
    o.commit₄ o.commit₅ o.stateOf

/-- The checkpoint structure of the Akita opening of `o`: `towerRoot` under the verifier
`Opening.accepts`. -/
noncomputable def Opening.tower :=
  towerRoot (weight₃ := o.weight₃) (weight₄ := o.weight₄) (weight₅ := o.weight₅)
    Route.coefficient (Route.euclidean norm₃.operator norm₃.limit)
    (Route.euclidean norm₄.operator norm₄.limit) (Route.euclidean norm₅.operator norm₅.limit)
    (o.accepts setup packageWidths step cap control read head) zeroth first second third fourth
    fifth o.terminal 32 32 16 16 16 (Compose.root C vf) (Compose.read C) o.commit o.value o.place

local notation "whole" =>
  Compose.chain C vf (Opening.accepts setup packageWidths step cap control read head o)
    (Opening.tower setup packageWidths step cap control read head o) o.key o.openKey o.mass
    Sum.inl

/-- The hypotheses of `FinalKey20.sound` on the Akita opening of `o`. `laid`: the shape passes
the layout of step 5 (SPEC.md Algorithm 10). `early`: the commitments of the key polynomial and
of `W` are fixed by the first message. `registered` and `covers` are those of
`FinalAkita.final_extract_sound`. `chance` to `chanceL`: each challenge family has the slot
probability at most `p`. The other fields are the hypotheses of `Key20.tower_suffix`. -/
structure Opening.Sized (p : ℝ) : Prop where
  laid : setup.Laid o.shape
  early : ∀ anchor g, g ≠ 2 → o.bound anchor g = o.bound (anchor.take 0) g
  injective : Function.Injective o.block
  units : ∀ g, UnitDifferences {challenge | ∃ slot, o.decode g slot = some challenge}
  sparse : ∀ g challenge, (∃ slot, o.decode g slot = some challenge) →
    Sparse (o.mass g) (o.peak g) challenge
  injective₁ : Function.Injective o.block₁
  units₁ : UnitDifferences {challenge | ∃ slot, o.decode₁ slot = some challenge}
  sparse₁ : ∃ peak, ∀ challenge, (∃ slot, o.decode₁ slot = some challenge) →
    Sparse o.weight₁ peak challenge
  injective₂ : Function.Injective o.block₂
  units₂ : UnitDifferences {challenge | ∃ slot, o.decode₂ slot = some challenge}
  sparse₂ : ∃ peak, ∀ challenge, (∃ slot, o.decode₂ slot = some challenge) →
    Sparse o.weight₂ peak challenge
  injective₃ : Function.Injective o.block₃
  units₃ : UnitDifferences {challenge | ∃ slot, o.decode₃ slot = some challenge}
  sparse₃ : ∃ peak, ∀ challenge, (∃ slot, o.decode₃ slot = some challenge) →
    Sparse o.weight₃ peak challenge
  bounded₃ : ∀ challenge, (∃ slot, o.decode₃ slot = some challenge) →
    MulBound norm₃.operator challenge
  injective₄ : Function.Injective o.block₄
  units₄ : UnitDifferences {challenge | ∃ slot, o.decode₄ slot = some challenge}
  sparse₄ : ∃ peak, ∀ challenge, (∃ slot, o.decode₄ slot = some challenge) →
    Sparse o.weight₄ peak challenge
  bounded₄ : ∀ challenge, (∃ slot, o.decode₄ slot = some challenge) →
    MulBound norm₄.operator challenge
  injective₅ : Function.Injective o.block₅
  units₅ : UnitDifferences {challenge | ∃ slot, o.decode₅ slot = some challenge}
  sparse₅ : ∃ peak, ∀ challenge, (∃ slot, o.decode₅ slot = some challenge) →
    Sparse o.weight₅ peak challenge
  bounded₅ : ∀ challenge, (∃ slot, o.decode₅ slot = some challenge) →
    MulBound norm₅.operator challenge
  next : Terminal.start o.terminal = Tier.finish fifth
  inside : Terminal.start o.terminal < total
  injectiveL : Function.Injective (Terminal.block o.terminal)
  unitsL : UnitDifferences (Terminal.family o.terminal)
  boundedL : ∀ challenge ∈ Terminal.family o.terminal, MulBound normTerminal.operator challenge
  chance : ∀ g, slotChance (o.decode g) ≤ p
  chance₁ : slotChance o.decode₁ ≤ p
  chance₂ : slotChance o.decode₂ ≤ p
  chance₃ : slotChance o.decode₃ ≤ p
  chance₄ : slotChance o.decode₄ ≤ p
  chance₅ : slotChance o.decode₅ ≤ p
  chanceL : slotChance (Terminal.decode o.terminal) ≤ p
  registered : ∀ anchor, o.honest.Opens o.factorBound o.sourceBound (o.key 0) (o.bound anchor 0)
    (keyBase (FinalVerifier.Setup.params S₀ o.shape))
  covers : o.factorBound * (root.fold 0).sourceBound (root.shape 0) + 2 * o.mass 0 * o.sourceBound ≤
    o.innerBound

/-- The conditions `Compose.Sized` of the final verifier for the shape of `o`, and the suffix of
its Akita opening with 142 retry loops and the sum of the stage errors at most `317479/(1024·|K|) +
135·p`
(`FinalAkita.mixed_ready` for the numbers of `Key20.lean`, with `Key20.towerError_le`).
`blocks`: the key polynomial has 512 blocks. `coords`: an answer of the oracle has 28 slots. -/
theorem Opening.ready {p : ℝ} (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (sized : o.Sized setup packageWidths step cap control read head p) :
    Compose.Sized C vf ∧ ∃ error loops tree,
      Tier.Suffix (o.accepts setup packageWidths step cap control read head) (Compose.root C vf)
        (OpensAt (Compose.root C vf) o.commit o.value o.place o.key o.mass)
        (o.tower setup packageWidths step cap control read head) error loops tree ∧
      error ≤ 317479 / (1024 * Fintype.card K) + 135 * p ∧ 2 + loops = 144 := by
  have use := laidSetup_use setup o.shape sized.laid blocks
  obtain ⟨-, -, -, span⟩ := key20_points S₀ o.shape use
  have slots : 22 ≤ Fintype.card Coord := (by decide : 22 ≤ 28).trans coords
  have lookupsFit : (FinalVerifier.Setup.params S₀ o.shape).lookups ≤ Fintype.card Coord :=
    (by decide : 4 ≤ 28).trans coords
  have cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord := coords
  have challengesFit : 4 + ((FinalVerifier.Setup.params S₀ o.shape).lookups +
      ((FinalVerifier.Setup.params S₀ o.shape).claims +
        (FinalVerifier.Setup.params S₀ o.shape).claims +
        (FinalVerifier.Setup.params S₀ o.shape).spanVars)) ≤ Fintype.card Coord := by
    have claims : (FinalVerifier.Setup.params S₀ o.shape).claims ≤ 3 :=
      Nat.succ_le_succ mode.slots_le
    have lookups : (FinalVerifier.Setup.params S₀ o.shape).lookups = 4 := rfl
    omega
  have both : Compose.Sized C vf ∧
      Tier.Suffix (o.accepts setup packageWidths step cap control read head) (Compose.root C vf)
        (OpensAt (Compose.root C vf) o.commit o.value o.place o.key o.mass)
        (o.tower setup packageWidths step cap control read head) _ _ _ :=
    mixed_ready S₀ packageWidths step cap control o.shape read head o.bound
      ⟨premises.represents, premises.support⟩ use.laid lookupsFit challengesFit cube sized.early
      sized.injective sized.units sized.sparse root_sized.2.2.1 root_sized.2.2.2.1
      root_sized.2.2.2.2 ((by decide : rootVars ≤ 22).trans slots)
      ((by decide : rootBatch ≤ 22).trans slots) ((by decide : 3 ≤ 22).trans slots)
      (Key20.levelSized slots (by decide) (by decide) level₁_sized.2.2.2.1 level₁_sized.2.2.2.2
        sized.injective₁ sized.units₁ sized.sparse₁)
      (Key20.levelSized slots (by decide) (by decide) sized₂.2.2.1 sized₂.2.2.2 sized.injective₂
        sized.units₂ sized.sparse₂)
      (Key20.levelSized slots (by decide) (by decide) sized₃.2.2.1 sized₃.2.2.2 sized.injective₃
        sized.units₃ sized.sparse₃)
      (Key20.levelSized slots (by decide) (by decide) sized₄.2.2.1 sized₄.2.2.2 sized.injective₄
        sized.units₄ sized.sparse₄)
      (Key20.levelSized slots (by decide) (by decide) sized₅.2.2.1 sized₅.2.2.2 sized.injective₅
        sized.units₅ sized.sparse₅)
      (⟨sized.bounded₃, (Key20.norms_sized (level₃, norm₃) (by simp)).1,
        (Key20.norms_sized (level₃, norm₃) (by simp)).2⟩ :
          NormSized level₃.shape level₃.fold o.decode₃ norm₃.operator norm₃.limit level₃.vars)
      (⟨sized.bounded₄, (Key20.norms_sized (level₄, norm₄) (by simp)).1,
        (Key20.norms_sized (level₄, norm₄) (by simp)).2⟩ :
          NormSized level₄.shape level₄.fold o.decode₄ norm₄.operator norm₄.limit level₄.vars)
      (⟨sized.bounded₅, (Key20.norms_sized (level₅, norm₅) (by simp)).1,
        (Key20.norms_sized (level₅, norm₅) (by simp)).2⟩ :
          NormSized level₅.shape level₅.fold o.decode₅ norm₅.operator norm₅.limit level₅.vars)
      sized.inside sized.injectiveL sized.unitsL sized.boundedL sized.next
  exact ⟨both.1, _, _, _, both.2,
    Key20.towerError_le (β := o.β) (key := o.key) (openKey := o.openKey) (claim := o.claim)
      (block := o.block) (mass := o.mass) (β₁ := o.β₁) (key₁ := o.key₁) (openKey₁ := o.openKey₁)
      (claim₁ := o.claim₁) (block₁ := o.block₁) (weight₁ := o.weight₁) (key₂ := o.key₂)
      (openKey₂ := o.openKey₂) (claim₂ := o.claim₂) (block₂ := o.block₂) (weight₂ := o.weight₂)
      (key₃ := o.key₃) (openKey₃ := o.openKey₃) (claim₃ := o.claim₃) (block₃ := o.block₃)
      (weight₃ := o.weight₃) (key₄ := o.key₄) (openKey₄ := o.openKey₄) (claim₄ := o.claim₄)
      (block₄ := o.block₄) (weight₄ := o.weight₄) (key₅ := o.key₅) (openKey₅ := o.openKey₅)
      (claim₅ := o.claim₅) (block₅ := o.block₅) (weight₅ := o.weight₅) (read := Compose.read C)
      (start := Compose.root C vf) sized.chance sized.chance₁ sized.chance₂ sized.chance₃
      sized.chance₄ sized.chance₅ sized.chanceL, by decide⟩

variable (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))

/-- The proof with witnesses that the extractor gives for the shape of `o`: the proof
`Setup.envelope` of the table `W` of a certificate, for the statement and the claims of the first
message of the prover. It is the bottom proof when the extractor returns no certificate. -/
noncomputable def Opening.proof (budget : ℕ) (table : Prefix Wire (Option Coord → K) total → Option
  Coord → K)
    (tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget total) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  match Checkpoint.extract whole prover budget table tape with
  | some (.inl certificate) =>
      FinalVerifier.Setup.envelope S₀ (head (prover.run table 0)).statement
        ((head (prover.run table 0)).claims o.shape)
        (Batch.tablesOf (Structure.witness whole certificate) 1)
  | _ => .bottom

variable [Finite Wire]

/-- The probability that the extractor for the shape of `o` returns a solution of Module-SIS
(`FinalAkita.breakChance`): for the key commitment, for a matrix of a group, of a level, or for
`A_term`. -/
noncomputable def Opening.breakChance (budget : ℕ) : ℝ :=
  FinalAkita.breakChance S₀ packageWidths step cap control o.shape read head o.bound
    (o.accepts setup packageWidths step cap control read head)
    (o.tower setup packageWidths step cap control read head) o.key o.openKey o.mass Sum.inl prover
    o.honest o.innerBound budget

/-- Knowledge soundness of the final verifier for the shape of `o` (Akita Theorem 10.17). For
each prover with at most `Q` queries to the oracle, the probability that the verifier
`Opening.accepts` accepts and `Verifies` rejects the proof `Opening.proof` of the extractor is at
most `(Q + 1)·(runCost + 317479/(1024·|K|) + 135·p) + (Q + 1)·144/budget + breakChance`. The
probability
is over the table of the oracle and the tape of the extractor. The terms are those of
`FinalKey20.sound`. -/
theorem Opening.knowledge {p : ℝ} (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (sized : o.Sized setup packageWidths step cap control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget total,
        (if o.accepts setup packageWidths step cap control read head (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
              (head (played prover table).first).statement
              (o.proof setup packageWidths step cap control read head prover budget table tape)
          then (1 : ℝ) else 0) ≤
      (queries + 1) * (FinalVerifier.Setup.runCost S₀ o.shape +
          (317479 / (1024 * Fintype.card K) + 135 * p)) +
        (queries + 1) * (144 : ℕ) / budget +
        o.breakChance setup packageWidths step cap control read head prover budget := by
  obtain ⟨composed, error, loops, tree, later, small, count⟩ :=
    o.ready setup packageWidths step cap control read head premises blocks coords sized
  have suffix := final_suffix S₀ packageWidths step cap control o.shape read head o.bound Sum.inl
    composed (fun _ accepted => accepted.1) (fun _ found => found) later
  have missed := final_extraction_le S₀ packageWidths step cap control o.shape read head o.bound
    Sum.inl prover composed (fun _ accepted => accepted.1) (fun _ found => found) later bounded
    positive
  rw [count] at missed
  refine le_trans ?_ (add_le_add (missed.trans (add_le_add (mul_le_mul_of_nonneg_left
    (add_le_add le_rfl small) (by positivity)) le_rfl)) le_rfl)
  unfold Opening.breakChance FinalAkita.breakChance
  simp only [← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  have same : (Structure.protocol whole).final prover table = played prover table :=
    played_eq (o.accepts setup packageWidths step cap control read head) prover table
  have cases : o.accepts setup packageWidths step cap control read head (played prover table) ∧
      ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
        (head (played prover table).first).statement
        (o.proof setup packageWidths step cap control read head prover budget table tape) →
      (o.accepts setup packageWidths step cap control read head
          ((Structure.protocol whole).final prover table) ∧
        Checkpoint.extract whole prover budget table tape = none) ∨
      ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
        Solved (o.tower setup packageWidths step cap control read head) o.key o.honest
          o.innerBound out := by
    rintro ⟨accepted, refuted⟩
    rw [played_first] at refuted
    rw [← same] at accepted
    rcases result : Checkpoint.extract whole prover budget table tape with _ | out
    · exact Or.inl ⟨accepted, rfl⟩
    · refine Or.inr ⟨out, rfl, ?_⟩
      obtain ⟨-, extracted⟩ := suffix.extract_sound rfl prover budget table tape out result
      rcases out with certificate | broken
      · rcases Compose.Extracted.relation_or_collision extracted (sized.registered _)
          sized.covers with good | collision
        · refine absurd ?_ refuted
          unfold Opening.proof
          rw [result]
          exact good
        · exact collision
      · exact extracted
  split_ifs <;> first
    | exact absurd (cases ‹_›) (not_or.mpr ⟨‹_›, ‹_›⟩)
    | norm_num

end One

section All

variable {ι : Type} [Fintype ι] (kind : Wire → Option ι)
  (family : ι → Opening logicalWidth publicFits Wire Coord total)

/-- The final verifier over the Akita openings of `key20.aks` (SPEC.md Algorithm 10). It accepts
a transcript when the statement of the first message is a statement of zero steps (step 1), or
when the first message names the shape `kind` of the family and the verifier `Opening.accepts`
of this shape accepts. A first message with a shape outside the family has `kind` equal to
`none`. -/
def finalAccepts (point : Prefix Wire (Option Coord → K) total) : Prop :=
  (Terminal.StatementValid (head point.first).statement ∧
      (head point.first).statement.iteration = 0 ∧
      (head point.first).statement.zi = (head point.first).statement.z0) ∨
    ∃ index, kind point.first = some index ∧
      (family index).accepts setup packageWidths step cap control read head point

variable (prover : OracleComp (Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))

/-- The proof with witnesses of the extractor for the final verifier: the bottom proof for a
statement of zero steps, and `Opening.proof` of the shape that the first message names. -/
noncomputable def finalProof (budget : ℕ) (table : Prefix Wire (Option Coord → K) total → Option
  Coord → K)
    (tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget total) :
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  if Terminal.StatementValid (head (prover.run table 0)).statement ∧
      (head (prover.run table 0)).statement.iteration = 0 ∧
      (head (prover.run table 0)).statement.zi = (head (prover.run table 0)).statement.z0 then
    .bottom
  else
    match kind (prover.run table 0) with
    | none => .bottom
    | some index => (family index).proof setup packageWidths step cap control read head prover
        budget table tape

variable [Finite Wire]

/-- The bound of the final verifier for the shapes of the family: the sum over the shapes of
`(Q + 1)·(runCost + 317479/(1024·|K|) + 135·p) + (Q + 1)·144/budget + breakChance`. The terms of a
shape
are those of `FinalKey20.sound` for its setup `laidSetup` and its opening, and the sum is the
union bound over the shapes. -/
noncomputable def shapesBound (p : ℝ) (queries budget : ℕ) : ℝ :=
  ∑ index, ((queries + 1) *
      (FinalVerifier.Setup.runCost (laidSetup setup (family index).shape) (family index).shape +
        (317479 / (1024 * Fintype.card K) + 135 * p)) +
    (queries + 1) * (144 : ℕ) / budget +
    (family index).breakChance setup packageWidths step cap control read head prover budget)

/-- Knowledge soundness of the final verifier over the Akita openings of `key20.aks`, for all
shapes of the family and for the proof of zero steps. For each prover with at most `Q` queries
to the oracle, the probability that `finalAccepts` accepts and `Verifies` rejects the proof
`finalProof` of the extractor is at most `shapesBound`, the sum of the bounds of
`Opening.knowledge` over the shapes of the family. A statement of zero steps adds no term: it has
no draw, and `Verifies` accepts its proof (`FinalVerifier.verifies_of_bottom`). -/
theorem knowledge_all {p : ℝ} (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup packageWidths step cap control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget total,
        (if finalAccepts setup packageWidths step cap control read head kind family
              (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
              (head (played prover table).first).statement
              (finalProof setup packageWidths step cap control read head kind family prover budget
                table tape)
          then (1 : ℝ) else 0) ≤
      shapesBound setup packageWidths step cap control read head family prover p queries
        budget := by
  unfold shapesBound
  refine le_trans ?_ (Finset.sum_le_sum fun index _ =>
    (family index).knowledge setup packageWidths step cap control read head prover premises blocks
      coords (sized index) bounded positive)
  simp only [← Finset.expect_sum_comm]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  split_ifs with event
  · obtain ⟨accepted, refuted⟩ := event
    unfold finalAccepts at accepted
    rw [played_first] at accepted refuted
    by_cases bottom : Terminal.StatementValid (head (prover.run table 0)).statement ∧
        (head (prover.run table 0)).statement.iteration = 0 ∧
        (head (prover.run table 0)).statement.zi = (head (prover.run table 0)).statement.z0
    · refine absurd ?_ refuted
      unfold finalProof
      rw [ite_eq_left bottom]
      exact FinalVerifier.verifies_of_bottom setup step cap control bottom
    · rcases accepted with zero | ⟨index, named, accepted⟩
      · exact absurd zero bottom
      · have same : finalProof setup packageWidths step cap control read head kind family prover
            budget table tape =
            (family index).proof setup packageWidths step cap control read head prover budget
              table tape := by
          unfold finalProof
          rw [ite_eq_right bottom, named]
        rw [same, ← played_first] at refuted
        refine le_trans (le_of_eq (ite_eq_left (show _ ∧ _ from ⟨accepted, refuted⟩)).symm)
          (Finset.single_le_sum (f := fun index =>
            if (family index).accepts setup packageWidths step cap control read head
                  (played prover table) ∧
                ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
                  (head (played prover table).first).statement
                  ((family index).proof setup packageWidths step cap control read head prover
                    budget table tape)
              then (1 : ℝ) else 0)
            (fun _ _ => by split_ifs <;> norm_num) (Finset.mem_univ index))
  · exact Finset.sum_nonneg fun _ _ => by split_ifs <;> norm_num

/-- Soundness of the final verifier over the Akita openings of `key20.aks`, for all shapes of the
family and for the proof of zero steps. For each prover with at most `Q` queries to the oracle,
the probability that `finalAccepts` accepts and no proof with witnesses satisfies `Verifies` for
the statement of the first message is at most `shapesBound`: the sum over the shapes of the
family of `(Q + 1)·(runCost + 317479/(1024·|K|) + 135·p) + (Q + 1)·144/budget + breakChance`.
`blocks`: the
key polynomial has 512 blocks. `coords`: an answer of the oracle has 28 slots. `sized`: the
hypotheses of `FinalKey20.sound` on the opening of each shape. -/
theorem sound_all {p : ℝ} (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup packageWidths step cap control read head p) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table,
        (if finalAccepts setup packageWidths step cap control read head kind family
              (played prover table) ∧
            ¬ ∃ proof, Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
              (head (played prover table).first).statement proof
          then (1 : ℝ) else 0) ≤
      shapesBound setup packageWidths step cap control read head family prover p queries
        budget := by
  have flat (value : ℝ) : value =
      𝔼 _tape : Tape (Prefix Wire (Option Coord → K) total) (Option Coord) K budget total, value :=
    (Finset.expect_const Finset.univ_nonempty value).symm
  refine le_trans (Finset.expect_le_expect fun table _ => ?_)
    (knowledge_all setup packageWidths step cap control read head kind family prover premises
      blocks coords sized bounded positive)
  refine (flat _).le.trans (Finset.expect_le_expect fun tape _ => ?_)
  split_ifs with lost found
  · exact le_rfl
  · exact absurd ⟨lost.1, fun verifies => lost.2 ⟨_, verifies⟩⟩ found
  · exact zero_le_one
  · exact le_rfl

/-- Soundness of the final verifier with Akita openings, in the form of
`Claims.Final.verifier_sound`: `sound_all` for a setup whose rows are the rows of the packages
(`FinalRows.premises_of_rows`). For each prover with at most `Q` queries to the oracle, the
probability over the table of the oracle that `finalAccepts` accepts and no proof with witnesses
satisfies `Verifies` for the statement of the first message is at most `shapesBound`. -/
theorem verifier_sound {p : ℝ} {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 512) (coords : 28 ≤ Fintype.card Coord)
    (sized : ∀ index,
      (family index).Sized setup complete.packageWidth step cap control read head p)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table,
        (if finalAccepts setup complete.packageWidth step cap control read head kind family
              (played prover table) ∧
            ¬ ∃ proof, Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
              control (head (played prover table).first).statement proof
          then (1 : ℝ) else 0) ≤
      shapesBound setup complete.packageWidth step cap control read head family prover p queries
        budget :=
  sound_all setup complete.packageWidth step cap control read head kind family prover
    (FinalRows.premises_of_rows setup exports complete rows) blocks coords sized bounded positive

end All

end NightstreamFPrime.Export.BoundedDepth.FinalKey20
