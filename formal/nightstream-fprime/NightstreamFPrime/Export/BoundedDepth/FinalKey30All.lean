import NightstreamFPrime.Export.BoundedDepth.FinalKey30
import NightstreamFPrime.Export.BoundedDepth.FinalShapes
import NightstreamFPrime.Spec.Compression.Akita.Key30Instances

/-! The final verifier over the Akita opening of the schedule file
`crates/ns-compress/artifacts/key30.aks` for each claim shape and for the proof of zero steps,
with the decoders `draw` of the sparse challenge families (SPEC.md, Algorithm 10;
`SECURITY_MODEL.md`, adversaries 4 and 6; Akita, ePrint 2026/1983, revision of 2026-10-15,
Section 3.2, (23), Section 9 with Lemma 9.1, Definition 9.2 E1 and Theorem 10.17).

A finite family of claim shapes has one `Opening` for each shape. The row of the file, the layout
of the argument, the zeros of the opening point of `V` and the key polynomial are functions of
the shape. The bound is the sum of the bounds of the shapes.

- `Opening` has the data of the Akita opening for one shape: the row of the file and the data
  `Key30.Tower` of the opening. The split `Key30.Tower.Split` of the public weight of the root
  (Akita (132)) is the hypothesis `split` of each definition and theorem. `Opening.accepts` is
  the verifier for it: the conditions `Compose.Ran` and the verifier `Key30.Tower.links` at the
  stage `Compose.root`, which is `Extract.offAccepts` of `FinalKey30.sound` for the decoders
  `draw`. `Opening.body` is its checkpoint structure.
- `Opening.Drawn` has the hypotheses on an opening. It has no hypothesis on units, on `‖c‖_1` or
  on the slot probability. Its hypotheses on the challenges are `bounded₅`, `bounded₆` and
  `boundedL` of `Key30.Tower.Drawn`: a member that passes the filter has `‖c‖_mul,2 ≤ Γ` (Akita
  Appendix H, Theorem H.1).
- `Opening.proof` is the proof with witnesses that the table `W` of an output of the extractor
  gives, and `Opening.knowledge` bounds the event "the verifier accepts and `Verifies` rejects
  this proof".
- `finalAccepts`, `finalProof`, `adversary`, `FalseHistory` and `AdviceFailure` are those of
  `FinalShapes.lean` for the family. `shapesBound` is the sum of the bounds of the shapes.
- `verifier_sound_drawn`, `history_sound_drawn` and `history_knowledge_drawn` are
  `FinalShapes.sound_all`, `FinalShapes.history_sound` and `FinalShapes.history_knowledge` for a
  setup whose rows are the rows of the packages (`FinalRows.premises_of_rows`).
- `Opening.reduced` states that the list `Key30.instances` is complete for the event `Solved` of
  `FinalAkita.breakChance`. Lean has no statement on the hardness of an instance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.FinalKey30

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
open FinalKey20 (laidSetup played played_eq played_first)

attribute [local instance low] Classical.propDecidable

attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK
  NightstreamFPrime.Export.BoundedDepth.FinalKey20.instDegree2K

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {mode : Mode}

/-- The data of the Akita opening of `key30.aks` for the final proofs of one claim shape. `row`
is the row of the file, and `shape` has `pc` and the depths. `bound anchor g` is the commitment
of the group `g` that a stage input fixes. `tower` has the keys, the claims, the coordinates of
the fold challenges and the readers of the messages of the root, of the levels 1 to 6 and of the
terminal, with the setup prefix and its commitment. `honest` is a certified decommitment of the
key commitment, with the bounds `factorBound`, `sourceBound` and `innerBound` of
`FinalAkita.final_extract_sound`. -/
structure Opening (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) (Wire Coord : Type) (total : ℕ) where
  row : Key30.Row
  member : row ∈ Key30.rows
  shape : Claims logicalWidth publicFits
  bound : Prefix Wire (Coord → K) total → (g : Fin 3) → ((Key30.root row).shape g).Payload
  tower : Key30.Tower row member K Wire Coord total
  honest : Certified ((Key30.root row).shape 0)
  factorBound : ℕ
  sourceBound : ℕ
  innerBound : ℕ

variable (setup : FinalVerifier.Setup mode logicalWidth publicFits) (packageWidths : Fin 2 → ℕ)
  (step : AppState → AppWitness → AppState) (cap : ℕ) (control : AppState → AppWitness → Fin 2)
  {Wire Coord : Type} {total : ℕ} (read : Reader Wire)
  (head : Wire → First logicalWidth publicFits)
  [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

section One

variable (o : Opening logicalWidth publicFits Wire Coord total) (split : o.tower.Split)

local notation "S₀" => laidSetup setup o.shape

local notation "C" => compose S₀ packageWidths step cap control o.shape read head o.bound

local notation "vf" => FinalVerifier.Setup.verifier S₀ o.shape

/-- The final verifier for the proofs with steps of the shape of `o` (SPEC.md Algorithm 10),
under the setup `laidSetup`: the conditions `Compose.Ran` of the stages before the opening, and
the verifier `Key30.Tower.links` of the opening with the batching root at the stage
`Compose.root`. -/
def Opening.accepts (point : Prefix Wire (Coord → K) total) : Prop :=
  Compose.Ran C vf o.tower.commit o.tower.value o.tower.place point ∧
    o.tower.links split (Compose.read C) (Compose.root C vf) point

/-- The checkpoint structure of the Akita opening of `o`: `Key30.Tower.body` under the verifier
`Opening.accepts`. -/
noncomputable def Opening.body :=
  o.tower.body split (Compose.read C) (Compose.root C vf)
    (o.accepts setup packageWidths step cap control read head split)

local notation "whole" =>
  Compose.chain C vf (Opening.accepts setup packageWidths step cap control read head o split)
    (Opening.body setup packageWidths step cap control read head o split) o.tower.key
    o.tower.openKey Key30.mass Sum.inl

/-- The hypotheses on the Akita opening of `o`, whose decoders are the decoders `draw` of the
sparse challenge families of the Akita code. `laid`: the shape passes the layout of step 5
(SPEC.md Algorithm 10). `row`: `W` has the count of variables of the row of `o`. `early`: the
commitments of the key polynomial and of `W` are fixed by the first message. `opening`: the
hypotheses `Key30.Tower.Drawn` of the opening at the stage `Compose.root`. `registered` and
`covers` are those of `FinalAkita.final_extract_sound`. -/
structure Opening.Drawn (keepL : Rq Key30.terminal.innerDegree → Prop) : Prop where
  laid : setup.Laid o.shape
  row : (FinalVerifier.Setup.params S₀ o.shape).witnessVars = o.row.vars
  early : ∀ anchor g, g ≠ 2 → o.bound anchor g = o.bound (anchor.take 0) g
  opening : o.tower.Drawn split (Compose.read C) (Compose.root C vf) keepL
  registered : ∀ anchor, o.honest.Opens o.factorBound o.sourceBound (o.tower.key 0)
    (o.bound anchor 0) (keyBase (FinalVerifier.Setup.params S₀ o.shape))
  covers : o.factorBound * ((Key30.root o.row).fold 0).sourceBound ((Key30.root o.row).shape 0) +
    2 * Key30.mass 0 * o.sourceBound ≤ o.innerBound

variable (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire))

/-- The proof with witnesses that the extractor gives for the shape of `o`: the proof
`Setup.envelope` of the table `W` of a certificate, for the statement and the claims of the first
message of the prover. It is the bottom proof when the extractor returns no certificate. -/
noncomputable def Opening.proof (budget : ℕ) (table : Prefix Wire (Coord → K) total → Coord → K)
    (tape : Tape (Prefix Wire (Coord → K) total) Coord K budget total) :
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
    (o.accepts setup packageWidths step cap control read head split)
    (o.body setup packageWidths step cap control read head split) o.tower.key o.tower.openKey
    Key30.mass Sum.inl prover o.honest o.innerBound budget

/-- The bound of the final verifier for the shape of `o`:
`(Q + 1)·(runCost + 9104/|K| + (1458 + B_1 + B_2)/q²) + (Q + 1)·(1469 + B_1 + B_2)/budget +
breakChance`. `B_1` and `B_2` are the live blocks of the witness group of level 1 and of level 2
of the row. The count of the retry loops is 2 for the compression argument and
`1467 + B_1 + B_2` for the opening: 1680 for `W` of 27 variables and 1670 for `W` of 26
variables. -/
noncomputable def Opening.shapeBound (queries budget : ℕ) : ℝ :=
  (queries + 1) * (FinalVerifier.Setup.runCost S₀ o.shape + (9104 / (Fintype.card K : ℝ) +
      ((1458 + o.row.blocks₁ + o.row.blocks₂ : ℕ) : ℝ) * (1 / (fieldModulus : ℝ) ^ 2))) +
    (queries + 1) * (2 + (1467 + o.row.blocks₁ + o.row.blocks₂) : ℕ) / budget +
    o.breakChance setup packageWidths step cap control read head split prover budget

variable {o split} {keepL : Rq Key30.terminal.innerDegree → Prop}

/-- Knowledge soundness of the final verifier for the shape of `o` (Akita Theorem 10.17 with
Lemma 9.1). For each prover with at most `Q` queries to the oracle, the probability that the
verifier `Opening.accepts` accepts and `Verifies` rejects the proof `Opening.proof` of the
extractor is at most `Opening.shapeBound`. The probability is over the table of the oracle and the
tape of the extractor. `blocks`: the key polynomial has `2^19` blocks. `coords`: an answer of
the oracle has 40 slots. -/
theorem Opening.knowledge (premises : Premises setup packageWidths)
    (blocks : setup.keyBlocks = 524288) (coords : 40 ≤ Fintype.card Coord)
    (drawn : o.Drawn setup packageWidths step cap control read head split keepL) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → K) total) Coord K budget total,
        (if o.accepts setup packageWidths step cap control read head split
              (played prover table) ∧
            ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
              (head (played prover table).first).statement
              (o.proof setup packageWidths step cap control read head split prover budget table
                tape)
          then (1 : ℝ) else 0) ≤
      o.shapeBound setup packageWidths step cap control read head split prover queries
        budget := by
  have use : Use S₀ o.shape o.row := ⟨drawn.laid, blocks, rfl, drawn.row⟩
  obtain ⟨-, -, span, -⟩ := key30_points S₀ o.shape use
  have slots : 28 ≤ Fintype.card Coord := (by decide : 28 ≤ 40).trans coords
  have lookupsFit : (FinalVerifier.Setup.params S₀ o.shape).lookups ≤ Fintype.card Coord :=
    (by decide : 4 ≤ 40).trans coords
  have cube : CompressionFresh.oneShape.cubeVariables ≤ Fintype.card Coord := slots
  have challengesFit : 4 + ((FinalVerifier.Setup.params S₀ o.shape).lookups +
      ((FinalVerifier.Setup.params S₀ o.shape).claims +
        (FinalVerifier.Setup.params S₀ o.shape).claims +
        (FinalVerifier.Setup.params S₀ o.shape).spanVars)) ≤ Fintype.card Coord := by
    have claims : (FinalVerifier.Setup.params S₀ o.shape).claims ≤ 3 :=
      Nat.succ_le_succ mode.slots_le
    have lookups : (FinalVerifier.Setup.params S₀ o.shape).lookups = 4 := rfl
    omega
  have composed := compose_sized S₀ packageWidths step cap control o.shape read head o.bound
    ⟨premises.represents, premises.support⟩ use.laid lookupsFit challengesFit cube
    ((by decide : 3 ≤ 40).trans coords) drawn.early drawn.opening.fits
  obtain ⟨error, later, small⟩ := Key30.Tower.suffix drawn.opening slots
    (accept := o.accepts setup packageWidths step cap control read head split)
    fun _ accepted => accepted.2
  have later : Tier.Suffix (o.accepts setup packageWidths step cap control read head split)
      (Compose.root C vf)
      (OpensAt (Compose.root C vf) o.tower.commit o.tower.value o.tower.place o.tower.key
        Key30.mass)
      (o.body setup packageWidths step cap control read head split) error
      (1467 + o.row.blocks₁ + o.row.blocks₂)
      (54400647168 * ((129 + o.row.blocks₁) * (o.row.blocks₂ + 1))) := later
  have suffix := final_suffix S₀ packageWidths step cap control o.shape read head o.bound Sum.inl
    composed (fun _ accepted => accepted.1) (fun _ found => found) later
  have missed := final_extraction_le S₀ packageWidths step cap control o.shape read head o.bound
    Sum.inl prover composed (fun _ accepted => accepted.1) (fun _ found => found) later bounded
    positive
  refine le_trans ?_ (add_le_add (missed.trans (add_le_add (mul_le_mul_of_nonneg_left
    (add_le_add le_rfl small) (by positivity)) le_rfl)) le_rfl)
  unfold Opening.breakChance FinalAkita.breakChance
  simp only [← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_
  have same : (Structure.protocol whole).final prover table = played prover table :=
    played_eq (o.accepts setup packageWidths step cap control read head split) prover table
  have cases :
      o.accepts setup packageWidths step cap control read head split (played prover table) ∧
      ¬ Verifies setup.ajtai setup.circuitKeys step packageWidths cap control
        (head (played prover table).first).statement
        (o.proof setup packageWidths step cap control read head split prover budget table
          tape) →
      (o.accepts setup packageWidths step cap control read head split
          ((Structure.protocol whole).final prover table) ∧
        Checkpoint.extract whole prover budget table tape = none) ∨
      ∃ out, Checkpoint.extract whole prover budget table tape = some out ∧
        Solved (o.body setup packageWidths step cap control read head split) o.tower.key
          o.honest o.innerBound out := by
    rintro ⟨accepted, refuted⟩
    rw [played_first] at refuted
    rw [← same] at accepted
    rcases result : Checkpoint.extract whole prover budget table tape with _ | out
    · exact Or.inl ⟨accepted, rfl⟩
    · refine Or.inr ⟨out, rfl, ?_⟩
      obtain ⟨-, extracted⟩ := suffix.extract_sound rfl prover budget table tape out result
      rcases out with certificate | broken
      · rcases Compose.Extracted.relation_or_collision extracted (drawn.registered _)
          drawn.covers with good | collision
        · refine absurd ?_ refuted
          unfold Opening.proof
          rw [result]
          exact good
        · exact collision
      · exact extracted
  split_ifs <;> first
    | exact absurd (cases ‹_›) (not_or.mpr ⟨‹_›, ‹_›⟩)
    | norm_num

omit [Nonempty Coord] [Finite Wire] in
/-- The list `Key30.instances` is complete for the solutions of Module-SIS of the final verifier
over the Akita opening of `key30.aks` (Akita Lemma 10.20, Table 3 and Lemma 9.1). `out` is an
output of the extractor with `Solved`, the event of `FinalAkita.breakChance`: a solution for the
key commitment from `tableCollision`, or a witness of `R_bind` of the opening. Then a matrix of
the opening has a solution that gives a solution of an instance of the list (`Key30.Reduced`).
`inner`: the bound `innerBound` of the reduction for the key commitment is at most `η_A` of the
key group. -/
theorem Opening.reduced
    (inner : o.innerBound ≤
      ((Key30.root o.row).fold 0).innerBound ((Key30.root o.row).shape 0))
    {out : _ ⊕ _}
    (solved : Solved (o.body setup packageWidths step cap control read head split) o.tower.key
      o.honest o.innerBound out) :
    Key30.Reduced Key30.instances o.tower.key o.tower.openKey o.tower.key₁ o.tower.openKey₁
      o.tower.key₂ o.tower.openKey₂ o.tower.key₃ o.tower.openKey₃ o.tower.key₄ o.tower.openKey₄
      o.tower.key₅ o.tower.openKey₅ o.tower.key₆ o.tower.openKey₆ o.tower.terminal.matrix := by
  rcases out with certificate | brk
  · obtain ⟨ω, -, found⟩ := solved
    obtain ⟨z, -⟩ := Option.isSome_iff_exists.mp found
    exact Or.inl (Or.inl ⟨0, Key30.reduced_key o.member inner z⟩)
  · exact Key30.reduced o.member brk solved

end One

section All

variable {ι : Type} [Fintype ι] (kind : Wire → Option ι)
  (family : ι → Opening logicalWidth publicFits Wire Coord total)
  (split : ∀ index, (family index).tower.Split)
  (prover : OracleComp (Prefix Wire (Coord → K) total) (Coord → K) (ℕ → Wire))

/-- The final verifier over the Akita openings of `key30.aks` (SPEC.md Algorithm 10):
`FinalShapes.finalAccepts` for the verifiers `Opening.accepts` of the family. It accepts a
transcript when the statement of the first message is a statement of zero steps (step 1), or
when the first message names the shape `kind` of the family and the verifier of this shape
accepts. -/
def finalAccepts : Prefix Wire (Coord → K) total → Prop :=
  FinalShapes.finalAccepts head kind fun index =>
    (family index).accepts setup packageWidths step cap control read head (split index)

variable [Finite Wire]

/-- The bound of the final verifier for the shapes of the family: the sum over the shapes of
`Opening.shapeBound`. The sum is the union bound over the shapes. -/
noncomputable def shapesBound (queries budget : ℕ) : ℝ :=
  ∑ index, (family index).shapeBound setup packageWidths step cap control read head
    (split index) prover queries budget

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
  (complete : StepExtend.Complete exports)

/-- Soundness of the final verifier with the Akita openings of `key30.aks`, for all shapes of
the family and for the proof of zero steps, in the form of `Claims.Final.verifier_sound`
(`FinalShapes.sound_all` with `Opening.knowledge`). For each prover with at most `Q` queries to
the oracle, the probability over the table of the oracle that `finalAccepts` accepts and no
proof with witnesses satisfies `Verifies` for the statement of the first message is at most
`shapesBound`. `rows`: the rows of the setup are the rows of the packages
(`FinalRows.premises_of_rows`). `blocks`: the key polynomial has `2^19` blocks. `coords`: an
answer of the oracle has 40 slots. `drawn`: the hypotheses `Opening.Drawn` on the opening of
each shape, for a filter `keepL` of the terminal. -/
theorem verifier_sound_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 524288) (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Rq Key30.terminal.innerDegree → Prop}
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head
      (split index) (keepL index))
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table,
        (if finalAccepts setup complete.packageWidth step cap control read head kind family split
              (played prover table) ∧
            ¬ ∃ proof, Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
              control (head (played prover table).first).statement proof
          then (1 : ℝ) else 0) ≤
      shapesBound setup complete.packageWidth step cap control read head family split prover queries
        budget :=
  FinalShapes.sound_all setup complete.packageWidth step cap control head kind _
    (fun index => (family index).proof setup complete.packageWidth step cap control read head
      (split index) prover)
    prover budget fun index =>
      Opening.knowledge setup complete.packageWidth step cap control read head prover
        (FinalRows.premises_of_rows setup exports complete rows) blocks coords (drawn index)
        bounded positive

variable (budget : ℕ)
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

/-- The prover of the final verifier over the Akita openings of `key30.aks`, with the extractor
of at most `budget` attempts in each retry loop, as an IVC adversary (`FinalShapes.adversary`).
Its tape is the table of the oracle and the tape of the extractor, with the uniform law. Its
output is the accepted statement with the proof of the extractor. -/
noncomputable def adversary :
    HyperNova.ReverseStages.IvcAdversary (IvcSecurity.Output logicalWidth publicFits) :=
  FinalShapes.adversary head kind
    (fun index => (family index).accepts setup packageWidths step cap control read head
      (split index))
    (fun index => (family index).proof setup packageWidths step cap control read head
      (split index) prover)
    prover budget

/-- The final verifier accepts `(i, z0, zi)` for the table of the oracle, and no `i` valid steps
of `program` lead from `z0` to `zi` (`FinalShapes.FalseHistory`). -/
def FalseHistory (program : Stage1.Application.Program)
    (table : Prefix Wire (Coord → K) total → Coord → K) : Prop :=
  FinalShapes.FalseHistory head kind
    (fun index => (family index).accepts setup packageWidths step cap control read head
      (split index))
    prover program table

local notation "ivc" =>
  adversary setup complete.packageWidth step cap control read head kind family split prover
    budget

/-- The final verifier with the Akita openings of `key30.aks` against a false history, in the
form of `Claims.Final.history_sound` (`FinalShapes.history_sound` with `Opening.knowledge`). For
each prover with at most `Q` queries to the oracle, the probability of a false history is at
most `shapesBound` plus `IvcSecurity.walkBound` of the IVC adversary `adversary`. `drawn`: the
hypotheses `Opening.Drawn` on the opening of each shape. `depthBound`: each accepted statement
has at most `depth` steps. -/
theorem history_sound_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 524288) (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Rq Key30.terminal.innerDegree → Prop}
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head
      (split index) (keepL index))
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step cap control read head kind family split
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((ivc).tape.toOuterMeasure {sample |
        FalseHistory setup complete.packageWidth step cap control read head kind family split
          prover exports.program sample.1}).toReal ≤
      shapesBound setup complete.packageWidth step cap control read head family split prover queries
          budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth :=
  FinalShapes.history_sound setup step cap control head kind _ _ prover budget exports assumption
    closed complete
    (fun index =>
      Opening.knowledge setup complete.packageWidth step cap control read head prover
        (FinalRows.premises_of_rows setup exports complete rows) blocks coords (drawn index)
        bounded positive)
    keysShaped admitted depth depthBound

/-- The final verifier accepted the statement of the table that the stage `depth` of the reverse
extractor reads from its tape, and the witnesses that the extractor decodes from the answers of
the stage are not a history of this statement (`FinalShapes.AdviceFailure`, HyperNova
Definition 11 (ii)). -/
def AdviceFailure
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (tape : (HyperNova.ReverseStages.reverseStages assumption closed
      (IvcSecurity.walkAdversary ivc) admitted depth).1.Tape) : Prop :=
  FinalShapes.AdviceFailure setup step head kind
    (fun index => (family index).accepts setup complete.packageWidth step cap control read head
      (split index))
    (fun index => (family index).proof setup complete.packageWidth step cap control read head
      (split index) prover)
    prover budget exports assumption closed admitted depth tape

/-- The knowledge form of `history_sound_drawn`, in the form of
`Claims.Final.history_knowledge` (`FinalShapes.history_knowledge` with `Opening.knowledge`). The
probability that the final verifier accepts and the witnesses that the reverse extractor returns
are not a history of the accepted statement is at most `shapesBound` plus
`IvcSecurity.walkBound`. -/
theorem history_knowledge_drawn
    (rows : ∀ circuit : Fin 2, (setup.rows circuit).matrices =
      (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ key, (setup.circuitKeys key).digest.length = 4)
    (blocks : setup.keyBlocks = 524288) (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Rq Key30.terminal.innerDegree → Prop}
    (drawn : ∀ index, (family index).Drawn setup complete.packageWidth step cap control read head
      (split index) (keepL index))
    {queries : ℕ} (bounded : prover.QueryBound queries) (positive : 0 < budget)
    (admitted : StageAdmitted
      (HyperNova.ReverseStages.Stage.start (IvcSecurity.walkAdversary ivc))) (depth : ℕ)
    (depthBound : ∀ table,
      finalAccepts setup complete.packageWidth step cap control read head kind family split
        (played prover table) → (head (played prover table).first).statement.iteration ≤ depth) :
    ((HyperNova.ReverseStages.reverseStages assumption closed (IvcSecurity.walkAdversary ivc)
          admitted depth).1.tape.toOuterMeasure {tape |
        AdviceFailure setup step cap control read head kind family split prover exports
          complete budget assumption closed admitted depth tape}).toReal ≤
      shapesBound setup complete.packageWidth step cap control read head family split prover queries
          budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed ivc admitted
          depth :=
  FinalShapes.history_knowledge setup step cap control head kind _ _ prover budget exports
    assumption closed complete
    (fun index =>
      Opening.knowledge setup complete.packageWidth step cap control read head prover
        (FinalRows.premises_of_rows setup exports complete rows) blocks coords (drawn index)
        bounded positive)
    keysShaped admitted depth depthBound

end All

end NightstreamFPrime.Export.BoundedDepth.FinalKey30
