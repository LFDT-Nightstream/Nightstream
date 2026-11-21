import NightstreamFPrime.Export.BoundedDepth.FinalKey30All

/-! # Claims: the Akita openings of the schedule `key30.aks`

The results for the Akita openings of the compressed proof under the schedule file
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks (Akita, ePrint
2026/1983, revision of 2026-10-15, Section 9; `SPEC.md` Algorithm 10 and section 9;
`SECURITY_MODEL.md` adversary 6 and trust item 7). The claims for `key20.aks` and the claims for
each schedule are in `Claims.Akita`.

- Model: the model of `Claims.Akita`. The answers of the stages are one uniform table with
  independent slots. The verifier is the Lean verifier of the opening with an offloaded root. It
  is not the verifier of the pinned Akita code.
- Assumptions: `[MSIS]` and `[KECCAK]`. Each bound has the term `breakChance`, or it bounds only
  the event that the extractor returns no output. `key30_instances` and `key30_instance_list`
  give the instances of Module-SIS that `breakChance` is about, and Lean does not prove their
  hardness.
- Open: the split of the public weight of the root into the setup part (Akita (132)) is the
  hypothesis `split`. The operator bound `Γ` of the filtered challenge families is a hypothesis
  (`bounded₅`, `bounded₆`, `boundedL`). The commitment of the setup prefix in the verifier setup
  is the output of the Lean commitment algorithm by hypothesis (`registered`). No claim states that a verifier of
  this file accepts an honest proof. No Lean result compares the Lean decoders or the Lean
  verifier with the code. -/

namespace NightstreamFPrime.Claims.AkitaKey30

open Spec Spec.Compression Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped BigOperators Spec.FieldExtensionRing Spec.Compression.Production
attribute [local instance low] Classical.propDecidable
attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK

/-- The output of the extractor for the opening of a row of
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks (Akita Proposition
10.44 (i) and Lemma 9.1 for this schedule). The verifier is the Lean verifier
`Key30.Tower.accepts` of the opening with an offloaded root, with the decoders
`Akita.Extract.draw` of the sparse families. It is not the verifier of the pinned Akita code.
Each output of the extractor is a certified decommitment of the commitment of each group of the
root to a table that has the claimed value at the claimed point, or a solution of Module-SIS for
a matrix of a group, of a level or for `A_term` (`[MSIS]`). `split`: the split of the public
weight of the root into the setup part (Akita (132)) is a hypothesis, and Lean does not prove it
for the setup weight of the code. `drawn.registered`: the commitment of the setup prefix in the
verifier setup is the output of the Lean commitment algorithm for the table of the setup prefix
(Akita Lemma 9.1, "correct preprocessing"). -/
theorem key30_opening_sound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {r : Akita.Key30.Row}
    {member : r ∈ Akita.Key30.rows}
    {o : Akita.Key30.Tower r member E Wire Coord total}
    {split : o.Split}
    {read : Akita.Extract.Reader Wire}
    {keepL : Akita.Rq Akita.Key30.terminal.innerDegree → Prop}
    (drawn : o.Drawn split read 0 keepL)
    (slots : 28 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
    (budget : ℕ)
    (table : Akita.Extract.Prefix Wire (Option Coord → E) total → Option Coord → E)
    (tape : Checkpoint.Tape (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord) E budget
          (o.body split read 0 (o.accepts split read)).protocol.stages)
    (out : Akita.Extract.Opened (Akita.Key30.root r).Weak
            (Akita.Extract.Opened (Akita.Key30.level₁ r).Weak
              (Akita.Extract.Opened (Akita.Weak (Akita.Key30.level₂ r).opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₅.opening)
                      (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₆.opening)
                        (Akita.Extract.Prefix Wire (Option Coord → E) total ×
                          Option (Akita.TerminalFork Akita.Key30.terminal)))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * Akita.Key30.mass g) *
                ((Akita.Key30.root r).fold g).sourceBound ((Akita.Key30.root r).shape g)) ⊕
            (Akita.Batch.LevelBreak o.key₁ o.openKey₁ fun g =>
                2 * (2 * Akita.Key30.mass₁ g) *
                  ((Akita.Key30.level₁ r).fold g).sourceBound ((Akita.Key30.level₁ r).shape g)) ⊕
              Akita.Extract.LevelBreak o.key₂ o.openKey₂
                  (2 * (2 * 51) * (Akita.Key30.level₂ r).fold.sourceBound (Akita.Key30.level₂ r).shape) ⊕
                Akita.Extract.LevelBreak o.key₃ o.openKey₃
                    (2 * (2 * 51) * Akita.Key30.level₃.fold.sourceBound Akita.Key30.level₃.shape) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ 31
                      (Akita.Key30.level₄.fold.sourceBound Akita.Key30.level₄.shape)
                      Akita.Extract.Route.coefficient ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ 53
                        (Akita.Key30.level₅.fold.sourceBound Akita.Key30.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit) ⊕
                      Akita.Extract.Route.Break o.key₆ o.openKey₆ 53
                          (Akita.Key30.level₆.fold.sourceBound Akita.Key30.level₆.shape)
                          (Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit) ⊕
                        Option
                          ((Akita.Extract.Route.euclidean Akita.Key30.normTerminal.operator
                                Akita.Key30.normTerminal.limit).Kernel
                            o.terminal.matrix o.terminal.weight o.terminal.bound))
    (result : out ∈ Checkpoint.extract (o.body split read 0 (o.accepts split read)) prover budget table tape) :
    o.accepts split read ((o.body split read 0 (o.accepts split read)).protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          Akita.Extract.OpensAt 0 o.commit o.value o.place o.key Akita.Key30.mass
            ((o.body split read 0 (o.accepts split read)).protocol.input prover table 0)
            ((o.body split read 0 (o.accepts split read)).witness certificate))
        (o.body split read 0 (o.accepts split read)).breaks out :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extract_sound E fieldE fintypeE decidableEqE
    algebraZqE degree2E Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total r member o split read
    keepL drawn slots prover budget table tape out result

/-- The extraction bound for the opening of a row of `crates/ns-compress/artifacts/key30.aks`
(Akita Theorem 10.46 with Lemma 9.1 for this schedule). The model is the model of
`Claims.Akita.checkpoint_extraction`, and the verifier is the Lean verifier
`Key30.Tower.accepts`, which differs from the verifier of the pinned Akita code. For each prover
with at most `Q` queries, the probability that the verifier accepts and the extractor returns no
output is at most
`(Q + 1)·(1623435/(4096·q²) + (1752 + B_1 + 4·B_2)/(256·q²)) +
(Q + 1)·(1467 + B_1 + B_2)/budget`. `B_1` and `B_2` are the live blocks of the witness group of
level 1 and of level 2 of the row: 167 and 44 for `W` of 27 variables, and 158 and 43 for `W` of
26 variables. The draw of each site has the factor `2^{−g}` for the work bits `g` of the site in
the Akita code (Akita Remark 10.18, `Extract.workBits`, `Extract.foldWorkBits`). The bound has no
term for the
probability that an output is a solution of Module-SIS (`[MSIS]`). The operator bound of the
filtered challenge families is the hypothesis `drawn.bounded₅`, `drawn.bounded₆` and
`drawn.boundedL`. -/
theorem key30_opening_bound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {r : Akita.Key30.Row}
    {member : r ∈ Akita.Key30.rows}
    {o : Akita.Key30.Tower r member E Wire Coord total}
    {split : o.Split}
    {read : Akita.Extract.Reader Wire}
    {keepL : Akita.Rq Akita.Key30.terminal.innerDegree → Prop}
    [finiteWire : Finite Wire]
    (drawn : o.Drawn split read 0 keepL)
    (slots : 28 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        𝔼 tape,
          if
              o.accepts split read
                  ((o.body split read 0 (o.accepts split read)).protocol.final prover table) ∧
                Checkpoint.extract (o.body split read 0 (o.accepts split read)) prover budget table tape =
                  none then
            1
          else 0) ≤
      ((↑queries : ℝ) + 1) *
          (1623435 / (4096 * (↑(Fintype.card E) : ℝ)) +
            (↑(1752 + r.blocks₁ + 4 * r.blocks₂) : ℝ) / 256 * (1 / (↑fieldModulus : ℝ) ^ 2)) +
        ((↑queries : ℝ) + 1) * (↑(1467 + r.blocks₁ + r.blocks₂) : ℝ) / (↑budget : ℝ) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extraction_le E fieldE fintypeE decidableEqE
    algebraZqE degree2E Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total r member o split read
    keepL finiteWire drawn slots prover queries bounded budget positive

/-- Soundness of the final verifier with the Akita openings of
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks:
`Claims.Final.verifier_sound` in the game with Akita openings (Akita Theorem 10.17 with
Section 9). The claim is for each claim shape of the family and for the proof of zero steps, and
for the two rows of the file. The verifier is the Lean verifier `FinalKey30.finalAccepts`, not
the verifier of the pinned Akita code. For each prover with at most `Q` queries to the oracle,
the probability that the verifier accepts and no proof with witnesses satisfies `Verifies` is at
most `FinalKey30.shapesBound`: for each shape,
`(Q + 1)·(runCost + 1623435/(4096·q²) + (1752 + B_1 + 4·B_2)/(256·q²)) +
(Q + 1)·(1469 + B_1 + B_2)/budget + breakChance`. Lean gives no bound on `breakChance`, the
probability that the extractor returns a solution of Module-SIS (`[MSIS]`). `drawn`: the
hypotheses `FinalKey30.Opening.Drawn` on the opening of each shape. `split`: the split of the
public weight of the root into the setup part (Akita (132)) is a hypothesis for each shape, and
Lean does not prove it for the setup weight of the code. -/
theorem key30_verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (split : ∀ (index : ι), (family index).tower.Split)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Akita.Rq Akita.Key30.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (split index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        if
            FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family
                split (FinalKey20.played prover table) ∧
              ¬∃ proof,
                  RunCompleteness.Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
                    control (head (FinalKey20.played prover table).first).statement proof then
          1
        else 0) ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family split prover
        queries budget :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.verifier_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family split prover finiteWire relation cubeFits exports complete rows blocks coords keepL drawn queries
    bounded budget positive

/-- The final verifier with the Akita openings of `crates/ns-compress/artifacts/key30.aks`
against a false history: `Claims.Final.history_sound` in the game with Akita openings (Akita
Theorem 10.17 with Section 9). The model and the verifier are those of `key30_verifier_sound`.
For each prover with at most `Q` queries, the probability that the verifier accepts
`(i, z0, zi)` and no `i` valid steps lead from `z0` to `zi` is at most `FinalKey30.shapesBound`
plus `IvcSecurity.walkBound` under `[HN-A1]`. Lean gives no bound on the term `breakChance` of
`shapesBound` (`[MSIS]`). -/
theorem key30_history_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (split : ∀ (index : ι), (family index).tower.Split)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (budget : ℕ)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
            (IvcWalk.Answer logicalWidth publicFits) →
          Prop}
    {Efficient : (adversary :
            HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
          HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
          (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
          (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Akita.Rq Akita.Key30.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (split index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family split
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Prefix Wire (Option Coord → K) total → Option Coord → K),
          FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family split
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((FinalKey30.adversary setup complete.packageWidth step cap control read head kind family split prover
                  budget).tape.toOuterMeasure :
            Set
                (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family
                    split prover budget).Tape →
              ENNReal)
          {sample |
            FinalKey30.FalseHistory setup complete.packageWidth step cap control read head kind family
              split prover exports.program sample.1}).toReal ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family split prover
          queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family split
            prover budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family split prover finiteWire relation cubeFits exports complete budget Admitted StageAdmitted
    Efficient error assumption closed rows keysShaped blocks coords keepL drawn queries bounded positive
    admitted depth depthBound

/-- The knowledge form of `key30_history_sound`: `Claims.Final.history_knowledge` in the game
with the Akita openings of `crates/ns-compress/artifacts/key30.aks`. The probability that the
final verifier accepts and the witnesses that the reverse extractor returns are not a history of
the accepted statement is at most `FinalKey30.shapesBound` plus `IvcSecurity.walkBound` under
`[HN-A1]`. Lean gives no bound on the term `breakChance` of `shapesBound` (`[MSIS]`). -/
theorem key30_history_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (split : ∀ (index : ι), (family index).tower.Split)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (budget : ℕ)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
            (IvcWalk.Answer logicalWidth publicFits) →
          Prop}
    {Efficient : (adversary :
            HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
          HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
          (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
          (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    {keepL : ι → Akita.Rq Akita.Key30.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (split index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family split
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Prefix Wire (Option Coord → K) total → Option Coord → K),
          FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family split
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((↑(HyperNova.ReverseStages.reverseStages assumption closed
                      (IvcSecurity.walkAdversary
                        (FinalKey30.adversary setup complete.packageWidth step cap control read head kind
                          family split prover budget))
                      admitted depth) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalKey30.adversary setup complete.packageWidth step cap control read head
                            kind family split prover budget))
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalKey30.AdviceFailure setup step cap control read head kind family split prover exports
              complete budget assumption closed admitted depth tape}).toReal ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family split prover
          queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family split
            prover budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_knowledge_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family split prover finiteWire relation cubeFits exports complete budget Admitted StageAdmitted
    Efficient error assumption closed rows keysShaped blocks coords keepL drawn queries bounded positive
    admitted depth depthBound

/-- The numbers of the three opening claims of the final verifier for the key of
`crates/ns-compress/artifacts/key30.aks` and the row `r` of the file. For a setup with
`FinalKey30.Use`, the point of the key polynomial has 30 coordinates, the point of `V` has 28
coordinates, the span has at most 26 variables, and the point of `W` has the count of variables
of the row, 27 or 26. These are the sizes of the tables of the groups of `Key30.root`. -/
theorem key30_points
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : FinalVerifier.Claims logicalWidth publicFits)
    {r : Akita.Key30.Row}
    (use : FinalKey30.Use setup shape r) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars + (setup.params shape).rowVars = 30 ∧
      (setup.params shape).spanVars + (setup.verifier shape).columnVars + (setup.verifier shape).zeros =
          28 ∧
        (setup.params shape).spanVars ≤ 26 ∧ (setup.params shape).witnessVars = r.vars :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.key30_points logicalWidth publicFits mode setup shape r
    use

/-- The instances of Module-SIS of the opening of the two rows of
`crates/ns-compress/artifacts/key30.aks` (Akita Table 3 and Appendix G). Each of the 54 matrix
views of each row is below one of the 18 instances `(d, n, m, p, η)` of the list: the same
degree, rank and norm, and a width and a bound that are not larger. The list has no instance
without a view, and no instance below another instance. A Euclidean instance has the power of
two that the file prices. `tests/AkitaSchedules.lean` compares the list with the file. Lean does
not prove that an instance is hard (`[MSIS]`). -/
theorem key30_instance_list :
    (∀ r ∈ Akita.Key30.rows, Akita.Covered Akita.Key30.instances (Akita.Key30.views r)) ∧
      List.map Akita.Setup.Instance.key Akita.Key30.instances =
          [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
            (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
            (128, 1, 416, false, 31), (128, 1, 4224, false, 15), (128, 1, 11264, false, 7),
            (128, 4, 512, false, 126852), (128, 5, 512, false, 835380), (128, 6, 2048, false, 835380),
            (256, 1, 8192, false, 15), (256, 1, 28160, false, 7), (512, 2, 1024, false, 8126340),
            (512, 2, 2048, false, 507780), (512, 3, 5120, false, 2080374660),
            (2048, 1, 4096, false, 16642998148)] ∧
        (∀ r ∈ Akita.Key30.rows, (Akita.Key30.views r).length = 54) ∧
          (∀ I ∈ Akita.Key30.instances,
              ∃ r ∈ Akita.Key30.rows,
                ∃ v ∈ Akita.Key30.views r,
                  v.Below I ∧
                    v.cols = I.cols ∧ bif I.euclidean then I.bound < 2 * v.bound else v.bound = I.bound) ∧
            List.Pairwise (fun I J => ¬I.Below J ∧ ¬J.Below I) Akita.Key30.instances :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.instances_exact

/-- The list `Key30.instances` is complete for the term `breakChance` of
`key30_verifier_sound` (Akita Lemma 10.20, Table 3 and Lemma 9.1). `out` is an output of the
extractor that is a solution of Module-SIS (`solved`): for the key commitment, or a witness of
`R_bind` of the opening. A witness of `R_bind` of the level 1 has also the solution of Akita
Lemma 9.1 for the commitment of the setup prefix. Then a matrix of the opening has a solution
that gives a solution of an instance of the list, at the bound that the file prices. `inner`:
the bound of the reduction for the key commitment is at most `η_A = 16642998148` of the key
group. Lean does not prove that an instance is hard (`[MSIS]`). -/
theorem key30_instances
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey30.Opening logicalWidth publicFits Wire Coord total}
    {split : o.tower.Split}
    (inner : o.innerBound ≤ ((Akita.Key30.root o.row).fold 0).innerBound ((Akita.Key30.root o.row).shape 0))
    {out : Akita.Extract.Opened (Akita.Key30.root o.row).Weak
            (Akita.Extract.Opened (Akita.Key30.level₁ o.row).Weak
              (Akita.Extract.Opened (Akita.Weak (Akita.Key30.level₂ o.row).opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₅.opening)
                      (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₆.opening)
                        (Akita.Extract.Prefix Wire (Option Coord → K) total ×
                          Option (Akita.TerminalFork Akita.Key30.terminal)))))))) ⊕
          (Akita.Batch.LevelBreak o.tower.key o.tower.openKey fun g =>
              2 * (2 * Akita.Key30.mass g) *
                ((Akita.Key30.root o.row).fold g).sourceBound ((Akita.Key30.root o.row).shape g)) ⊕
            (Akita.Batch.LevelBreak o.tower.key₁ o.tower.openKey₁ fun g =>
                2 * (2 * Akita.Key30.mass₁ g) *
                  ((Akita.Key30.level₁ o.row).fold g).sourceBound ((Akita.Key30.level₁ o.row).shape g)) ⊕
              Akita.Extract.LevelBreak o.tower.key₂ o.tower.openKey₂
                  (2 * (2 * 51) *
                    (Akita.Key30.level₂ o.row).fold.sourceBound (Akita.Key30.level₂ o.row).shape) ⊕
                Akita.Extract.LevelBreak o.tower.key₃ o.tower.openKey₃
                    (2 * (2 * 51) * Akita.Key30.level₃.fold.sourceBound Akita.Key30.level₃.shape) ⊕
                  Akita.Extract.Route.Break o.tower.key₄ o.tower.openKey₄ 31
                      (Akita.Key30.level₄.fold.sourceBound Akita.Key30.level₄.shape)
                      Akita.Extract.Route.coefficient ⊕
                    Akita.Extract.Route.Break o.tower.key₅ o.tower.openKey₅ 53
                        (Akita.Key30.level₅.fold.sourceBound Akita.Key30.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit) ⊕
                      Akita.Extract.Route.Break o.tower.key₆ o.tower.openKey₆ 53
                          (Akita.Key30.level₆.fold.sourceBound Akita.Key30.level₆.shape)
                          (Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit) ⊕
                        Option
                          ((Akita.Extract.Route.euclidean Akita.Key30.normTerminal.operator
                                Akita.Key30.normTerminal.limit).Kernel
                            o.tower.terminal.matrix o.tower.terminal.weight o.tower.terminal.bound)}
    (solved : FinalAkita.Solved
          (FinalKey30.Opening.body setup packageWidths step cap control read head o split) o.tower.key
          o.honest o.innerBound out) :
    Akita.Key30.Reduced Akita.Key30.instances o.tower.key o.tower.openKey o.tower.key₁ o.tower.openKey₁
      o.tower.key₂ o.tower.openKey₂ o.tower.key₃ o.tower.openKey₃ o.tower.key₄ o.tower.openKey₄
      o.tower.key₅ o.tower.openKey₅ o.tower.key₆ o.tower.openKey₆ o.tower.terminal.matrix :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.Opening.reduced logicalWidth publicFits mode setup
    packageWidths step cap control Wire Coord total read head fintypeCoord decidableEqCoord o split inner
    out solved

end NightstreamFPrime.Claims.AkitaKey30
