import NightstreamFPrime.Export.BoundedDepth.FinalKey20Challenge
import NightstreamFPrime.Export.BoundedDepth.FinalKey20Instances
import NightstreamFPrime.Spec.Checkpoint.Runs
import NightstreamFPrime.Spec.Compression.Akita.Key20Accept
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening
import NightstreamFPrime.Spec.Compression.Akita.ShortUnit
import NightstreamFPrime.Spec.Compression.Akita.TraceFold

/-! # Claims: the Akita polynomial commitment scheme

The results for the Akita openings of the compressed proof (Akita, ePrint 2026/1983, revision of
2026-10-15; `SPEC.md` Algorithm 10 and section 9; `SECURITY_MODEL.md` adversary 6 and trust
item 7).

- Model: the answers of the stages are one uniform table with independent slots, and an answer
  of one stage has one slot for each fold coordinate. The verifier is the Lean verifier of the
  opening. It is not the verifier of the pinned Akita code (`SECURITY_MODEL.md`, trust item 7).
- Scope: each claim that starts with `key20_` is for the schedule file
  `crates/ns-compress/artifacts/key20.aks` of the test key of `2^9` blocks. No claim of this file
  is for `crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks: the
  claims for that file are in `Claims.AkitaKey30`. The claims that end in `_with_akita` of
  `Claims.Final` do not use them and keep the premises of `[AKITA]`.
- Assumptions: `[MSIS]` and `[KECCAK]`. Each bound has the term `breakChance`, or it bounds only
  the event that the extractor returns no output. Lean gives no bound on the probability that an
  output is a solution of Module-SIS. `key20_instances` and `key20_instance_list` give the
  instances of Module-SIS that `breakChance` is about, and Lean does not prove their hardness.
- Open: the operator bound `Γ` of the filtered challenge families is a hypothesis (`bounded₃` to
  `boundedL`). No Lean result compares the Lean decoders or the Lean verifier with the code. -/

namespace NightstreamFPrime.Claims.Akita

open Spec Spec.Compression Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped BigOperators Spec.FieldExtensionRing Spec.Compression.Production
attribute [local instance low] Classical.propDecidable
attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK

/-- Akita Theorem 10.17 in the form (161), for a fixed environment. The model is one uniform table
of stage answers with independent slots, and a limit `budget` on the attempts of each retry loop
of the extractor. For each prover with at most `Q` queries to the table and each structure with
the conditions of Akita Definition 10.16 (`lawful`), the probability that the verifier accepts
and the extractor returns no output is at most `(Q + 1)·ε_ck + (Q + 1)·L/budget`. `ε_ck` is the
sum of the stage errors and `L` is the number of retry loops. The term `(Q + 1)·L/budget` is not
in the paper, whose extractor has an expected run time. -/
theorem checkpoint_extraction
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    (structure_ : Checkpoint.Structure Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Point (Coord → Slot) Proof)
    (lawful : structure_.Lawful)
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        𝔼 tape,
          if
              structure_.protocol.accept (structure_.protocol.final prover table) ∧
                Checkpoint.extract structure_ prover budget table tape = none then
            1
          else 0) ≤
      ((↑queries : ℝ) + 1) * structure_.checkError +
        ((↑queries : ℝ) + 1) * (↑(structure_.loopsFrom 0) : ℝ) / (↑budget : ℝ) :=
  @NightstreamFPrime.Spec.Checkpoint.checkpoint_extraction_le Point Coord Slot Proof Cert Break Cand Key
    Witness fintypePoint decidableEqPoint fintypeCoord decidableEqCoord fintypeSlot nonemptySlot structure_
    prover lawful queries bounded budget positive

/-- Akita Theorem 10.17, the number of runs. In the model of `checkpoint_extraction`, the extractor
runs the prover at most `K̂ + (Q + 1)·(K̂ − 1)` times in expectation, for the tree size `K̂` of the
structure and for each limit `budget`. The claim gives no bound on the run time. -/
theorem checkpoint_runs
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    (structure_ : Checkpoint.Structure Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Point (Coord → Slot) Proof)
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (budget : ℕ) :
    𝔼 table, 𝔼 tape, (↑(Checkpoint.extractRuns structure_ prover budget table tape) : ℝ) ≤
      (↑structure_.treeSize : ℝ) + ((↑queries : ℝ) + 1) * ((↑structure_.treeSize : ℝ) - 1) :=
  @NightstreamFPrime.Spec.Checkpoint.extract_runs_le Point Coord Slot Proof Cert Break Cand Key Witness
    fintypePoint decidableEqPoint fintypeCoord decidableEqCoord fintypeSlot nonemptySlot structure_ prover
    queries bounded budget

/-- The output of the extractor of Akita Theorem 10.17 (Akita Definition 10.15). For a structure
with the conditions of Akita Definition 10.16 (`lawful`), each output of the extractor is with an
accepted transcript of the prover, and it is a witness of the relation for the statement of the
prover or a witness of `R_bind`. The claim gives no bound on the probability of a witness of
`R_bind`. -/
theorem checkpoint_sound
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [decidableEqPoint : DecidableEq Point]
    [decidableEqCoord : DecidableEq Coord]
    (structure_ : Checkpoint.Structure Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Point (Coord → Slot) Proof)
    [fintypeCoord : Fintype Coord]
    [fintypeSlot : Fintype Slot]
    (lawful : structure_.Lawful)
    (budget : ℕ)
    (table : Point → Coord → Slot)
    (tape : Checkpoint.Tape Point Coord Slot budget structure_.protocol.stages)
    (out : Cert ⊕ Break)
    (result : Checkpoint.extract structure_ prover budget table tape = some out) :
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          structure_.relation (structure_.protocol.input prover table 0) (structure_.witness certificate))
        structure_.breaks out :=
  @NightstreamFPrime.Spec.Checkpoint.extract_sound Point Coord Slot Proof Cert Break Cand Key Witness
    decidableEqPoint decidableEqCoord structure_ prover fintypeCoord fintypeSlot lawful budget table tape
    out result

/-- Akita Lemma 10.23 (weak binding of one fold) with the bound (176), for coefficient packing.
Two weak openings of one statement (Akita Definition 10.22) with the same bounds `κ̄_1` and `β̄_∞`
that differ on a source block or on a carried digit vector give a solution of Module-SIS: for
`A` with the norm `2·κ̄_1·β̄_∞`, for `B` or `D` with the norm `b⋆ − 1`, or for a compression map
with the norm 1. The claim is the reduction. It gives no bound on the probability of a
solution (`[MSIS]`). -/
theorem fold_weak_binding
    {shape : Akita.Shape}
    {opening : Akita.OpeningShape shape}
    {E : Type}
    [commRingE : CommRing E]
    [moduleZqE : Module Akita.Zq E]
    {β : Module.Basis (Fin opening.planes) Akita.Zq E}
    {st : Akita.Statement opening E}
    {ω : Akita.Weak opening}
    {ω' : Akita.Weak opening}
    {factorBound : ℕ}
    {sourceBound : ℕ}
    {table : Akita.Table}
    {table' : Akita.Table}
    (opens : Akita.Weak.Opens β factorBound sourceBound ω st table)
    (opens' : Akita.Weak.Opens β factorBound sourceBound ω' st table')
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (Akita.weakCollision st (2 * factorBound * sourceBound) ω ω').isSome = true :=
  @NightstreamFPrime.Spec.Compression.Akita.weakCollision_isSome_of_bounds shape opening E commRingE moduleZqE β st ω
    ω' factorBound sourceBound table table' opens opens' different

/-- Akita Corollary 10.32 for one claim with coefficient packing, on the coefficient-norm route.
`fork` has two children for each block that satisfy their exact branch relations, with challenges
of a family with units as differences (`units`) and `‖c‖_1 = ω` (`sparse`). Then a child with
other carried digits gives a solution of Module-SIS for `B`, `D` or a compression map, or the
extracted opening is a weak opening with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`. -/
theorem fold_extracts
    {shape : Akita.Shape}
    {opening : Akita.OpeningShape shape}
    {fold : Akita.Descriptor}
    {E : Type}
    [commRingE : CommRing E]
    [moduleZqE : Module Akita.Zq E]
    {β : Module.Basis (Fin opening.planes) Akita.Zq E}
    {st : Akita.Statement opening E}
    {fork : Akita.Fork opening fold}
    {family : Set (Akita.Rq opening.carrierDegree)}
    {digits : Akita.Carried opening}
    {mass : ℕ}
    {peak : ℕ}
    (units : Akita.UnitDifferences family)
    (sparse : ∀ c ∈ family, Akita.Sparse mass peak c)
    (base : 2 ≤ fold.responseBase)
    (accepts : Akita.Fork.Accepts β family fork st)
    (opens : Akita.Carried.Opens β digits st) :
    (Akita.forkCollision st fork digits).isSome = true ∨
      Akita.Weak.Opens β (2 * mass) (fold.sourceBound shape) (Akita.extract fork digits) st
        (shape.decode (Akita.extract fork digits).source) :=
  @NightstreamFPrime.Spec.Compression.Akita.extract_opens_or_collision shape opening fold E commRingE moduleZqE β st
    fork family digits mass peak units sparse base accepts opens

/-- Akita Lemma 10.23 at a level with evaluation trace, with the bound (174). Two weak openings of
one statement that differ on a source block or on a carried digit vector give a solution of
Module-SIS: for `A` with each norm `η_A ≥ κ̄'_1·β̄_∞ + κ̄_1·β̄'_∞` (`radius`), for `B` or `D` with
the norm `b⋆ − 1`, or for a compression map with the norm 1. The claim is the reduction. It
gives no bound on the probability of a solution (`[MSIS]`). -/
theorem trace_weak_binding
    {shape : Akita.Shape}
    {opening : Akita.OpeningShape shape}
    {E : Type}
    [commRingE : CommRing E]
    [moduleZqE : Module Akita.Zq E]
    {st : Akita.TraceStatement opening E}
    {ω : Akita.Weak opening}
    {ω' : Akita.Weak opening}
    {factorBound : ℕ}
    {sourceBound : ℕ}
    {factorBound' : ℕ}
    {sourceBound' : ℕ}
    {table : Akita.Table}
    {table' : Akita.Table}
    {innerBound : ℕ}
    (opens : Akita.Weak.TraceOpens factorBound sourceBound ω st table)
    (opens' : Akita.Weak.TraceOpens factorBound' sourceBound' ω' st table')
    (radius : factorBound' * sourceBound + factorBound * sourceBound' ≤ innerBound)
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (Akita.traceCollision st innerBound ω ω').isSome = true :=
  @NightstreamFPrime.Spec.Compression.Akita.traceCollision_isSome shape opening E commRingE moduleZqE st ω ω'
    factorBound sourceBound factorBound' sourceBound' table table' innerBound opens opens' radius different

/-- Akita Corollary 10.32 for one claim at a level with evaluation trace, on the coefficient-norm
route. The hypotheses and the result are those of `fold_extracts` for the fold of a trace level:
a child with other carried digits gives a solution of Module-SIS, or the extracted opening is a
weak opening with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`. -/
theorem trace_extracts
    {shape : Akita.Shape}
    {opening : Akita.OpeningShape shape}
    {fold : Akita.Descriptor}
    {E : Type}
    [commRingE : CommRing E]
    [moduleZqE : Module Akita.Zq E]
    {st : Akita.TraceStatement opening E}
    {fork : Akita.TraceFork opening fold}
    {family : Set (Akita.Rq shape.innerDegree)}
    {digits : Akita.Carried opening}
    {mass : ℕ}
    {peak : ℕ}
    (units : Akita.UnitDifferences family)
    (sparse : ∀ c ∈ family, Akita.Sparse mass peak c)
    (base : 2 ≤ fold.responseBase)
    (accepts : Akita.TraceFork.Accepts family fork st)
    (opens : digits.TraceOpens st) :
    (Akita.traceForkCollision st fork digits).isSome = true ∨
      Akita.Weak.TraceOpens (2 * mass) (fold.sourceBound shape) (Akita.traceExtract fork digits) st
        (shape.decode (Akita.traceExtract fork digits).source) :=
  @NightstreamFPrime.Spec.Compression.Akita.traceExtract_opens_or_collision shape opening fold E commRingE moduleZqE
    st fork family digits mass peak units sparse base accepts opens

/-- Each `x ≠ 0` of `R_q = F_q[X]/(X^d + 1)` with `2·‖x‖_∞² < q` is a unit, for each degree
`d = 2^{n+1}`. This is the fact of Lyubashevsky and Seiler (ePrint 2017/523) that Akita cites for
the condition E1 of Akita Definition 9.2: the differences of the fold challenges are units. Lean
proves it for `q = 2^64 − 59`. It is not a premise. -/
theorem short_units
    {d : ℕ}
    [neZeroNatD : NeZero d]
    (n : ℕ)
    (power : 2 ^ (n + 1) = d)
    {x : Akita.Rq d}
    (nonzero : x ≠ 0)
    (short : 2 * Akita.normInf x ^ 2 < fieldModulus) :
    IsUnit x :=
  @NightstreamFPrime.Spec.Compression.Akita.isUnit_of_short d neZeroNatD n power x nonzero short

/-- The output of the extractor for the opening of `crates/ns-compress/artifacts/key20.aks`, the
schedule of the test key of `2^9` blocks (Akita Theorem 10.17 with Akita Proposition 10.44 (i)).
It does not cover `key30.aks`. The verifier is the Lean verifier `Key20.Tower.accepts` with the
Lean decoders `draw`, which differs from the verifier of the pinned Akita code. Each output is
with an accepted transcript. It is a certified decommitment of the commitment of each group of
the root to a table with the claimed value (`OpensAt`), or a solution of Module-SIS for a matrix
of a group, of a level, or for `A_term`. `drawn.bounded₃` to `drawn.boundedL`: the operator bound
`Γ` of the filtered families, which Lean does not prove (Akita Theorem H.1). -/
theorem key20_opening_sound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {o : Akita.Key20.Tower E Wire Coord total}
    {keepL : Akita.Rq Akita.Key20.terminal.innerDegree → Prop}
    (drawn : o.Drawn keepL)
    (slots : 22 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
    (budget : ℕ)
    (table : Akita.Extract.Prefix Wire (Option Coord → E) total → Option Coord → E)
    (tape : Checkpoint.Tape (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord) E budget
          o.body.protocol.stages)
    (out : Akita.Extract.Opened Akita.Key20.root.Weak
            (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₁.opening)
              (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₂.opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₅.opening)
                      (Akita.Extract.Prefix Wire (Option Coord → E) total ×
                        Option (Akita.TerminalFork Akita.Key20.terminal))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * Akita.Key20.rootMass g) *
                (Akita.Key20.root.fold g).sourceBound (Akita.Key20.root.shape g)) ⊕
            Akita.Extract.LevelBreak o.key₁ o.openKey₁
                (2 * (2 * 51) * Akita.Key20.level₁.fold.sourceBound Akita.Key20.level₁.shape) ⊕
              Akita.Extract.Route.Break o.key₂ o.openKey₂ 31
                  (Akita.Key20.level₂.fold.sourceBound Akita.Key20.level₂.shape)
                  Akita.Extract.Route.coefficient ⊕
                Akita.Extract.Route.Break o.key₃ o.openKey₃ 31
                    (Akita.Key20.level₃.fold.sourceBound Akita.Key20.level₃.shape)
                    (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ 53
                      (Akita.Key20.level₄.fold.sourceBound Akita.Key20.level₄.shape)
                      (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit) ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ 53
                        (Akita.Key20.level₅.fold.sourceBound Akita.Key20.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit) ⊕
                      Option
                        ((Akita.Extract.Route.euclidean Akita.Key20.normTerminal.operator
                              Akita.Key20.normTerminal.limit).Kernel
                          o.terminal.matrix o.terminal.weight o.terminal.bound))
    (result : out ∈ Checkpoint.extract o.body prover budget table tape) :
    o.accepts (o.body.protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          Akita.Extract.OpensAt 0 o.commit o.value o.place o.key Akita.Key20.rootMass
            (o.body.protocol.input prover table 0) (o.body.witness certificate))
        o.body.breaks out :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extract_sound E fieldE fintypeE decidableEqE
    algebraZqE degree2EAlgebraZqE Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total o keepL
    drawn slots prover budget table tape out result

/-- Akita Theorem 10.17 in the form (161) for the opening of
`crates/ns-compress/artifacts/key20.aks` (Akita Theorem 10.46 for this schedule). It does not
cover `key30.aks`. The model is one uniform table of stage answers with independent slots, and
the verifier is the Lean verifier `Key20.Tower.accepts`, which differs from the verifier of the
pinned Akita code. For each prover with at most `Q` queries, the probability that the verifier
accepts and the extractor returns no output is at most
`(Q + 1)·(317735/(1024·q²) + 49/(8·q²)) + (Q + 1)·142/budget`: `317735/(1024·q²)` for the
draws of the checks, `49/(8·q²)` for the 135 fold coordinates, and 142 retry loops. The draw of
each site has the factor `2^{−g}` for the work bits `g` of the site in the Akita code (Akita
Remark 10.18, `Extract.workBits`, `Extract.foldWorkBits`). The bound does not contain the
probability that the output is a solution of Module-SIS (`[MSIS]`), and the operator bound of
the filtered families is a hypothesis. -/
theorem key20_opening_bound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {o : Akita.Key20.Tower E Wire Coord total}
    {keepL : Akita.Rq Akita.Key20.terminal.innerDegree → Prop}
    [finiteWire : Finite Wire]
    (drawn : o.Drawn keepL)
    (slots : 22 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → E) total) (Option Coord → E) (ℕ → Wire))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        𝔼 tape,
          if
              o.accepts (o.body.protocol.final prover table) ∧
                Checkpoint.extract o.body prover budget table tape = none then
            1
          else 0) ≤
      ((↑queries : ℝ) + 1) * (317735 / (1024 * (↑(Fintype.card E) : ℝ)) + 49 / 8 * (1 / (↑fieldModulus : ℝ) ^ 2)) +
        ((↑queries : ℝ) + 1) * (↑142 : ℝ) / (↑budget : ℝ) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extraction_le E fieldE fintypeE decidableEqE
    algebraZqE degree2EAlgebraZqE Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total o keepL
    finiteWire drawn slots prover queries bounded budget positive

/-- The Lean verifier of the opening of `crates/ns-compress/artifacts/key20.aks` accepts a proof
(Akita Theorem 10.9 and Akita Section 8.2, for the zero tables). For each table of stage answers
whose slots decode to fold challenges and whose work slots have 57 zero work bits (`worked`: 57 is
the largest count of work bits of a site, and Lean does not model the search of the honest prover
for the work bits of Akita Remark 10.18), a proof exists whose transcript the verifier
`towerLinks` accepts: the proof of the honest prover for the zero tables of the key polynomial, `W` and `V`.
So the event of `key20_opening_bound` is not empty. The claim is not a completeness bound: Lean
has no bound on the probability that a response of another table fits, and the Lean verifier has
no rejection at a zero tensor factor. It does not cover `key30.aks`. -/
theorem key20_opening_accepts
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    (β : (g : Fin 3) → Module.Basis (Fin (Akita.Key20.root.planes g)) Akita.Zq E)
    (key : (g : Fin 3) → Akita.Key (Akita.Key20.root.shape g))
    (openKey : Akita.Key20.root.OpenKey)
    (block : Akita.Key20.root.Block → Coord)
    (decode : (g : Fin 3) → E → Option (Akita.Rq (Akita.Key20.root.carrierDegree g)))
    (mass : Fin 3 → ℕ)
    (β₁ : Module.Basis (Fin Akita.Key20.level₁.opening.planes) Akita.Zq E)
    (key₁ : Akita.Key Akita.Key20.level₁.shape)
    (openKey₁ : Akita.OpeningKey Akita.Key20.level₁.opening)
    (block₁ : Fin Akita.Key20.level₁.shape.blocks → Coord)
    (decode₁ : E → Option (Akita.Rq Akita.Key20.level₁.opening.carrierDegree))
    (weight₁ : ℕ)
    (key₂ : Akita.Key Akita.Key20.level₂.shape)
    (openKey₂ : Akita.OpeningKey Akita.Key20.level₂.opening)
    (block₂ : Fin Akita.Key20.level₂.shape.blocks → Coord)
    (decode₂ : E → Option (Akita.Rq Akita.Key20.level₂.shape.innerDegree))
    (weight₂ : ℕ)
    (key₃ : Akita.Key Akita.Key20.level₃.shape)
    (openKey₃ : Akita.OpeningKey Akita.Key20.level₃.opening)
    (block₃ : Fin Akita.Key20.level₃.shape.blocks → Coord)
    (decode₃ : E → Option (Akita.Rq Akita.Key20.level₃.shape.innerDegree))
    (weight₃ : ℕ)
    (key₄ : Akita.Key Akita.Key20.level₄.shape)
    (openKey₄ : Akita.OpeningKey Akita.Key20.level₄.opening)
    (block₄ : Fin Akita.Key20.level₄.shape.blocks → Coord)
    (decode₄ : E → Option (Akita.Rq Akita.Key20.level₄.shape.innerDegree))
    (weight₄ : ℕ)
    (key₅ : Akita.Key Akita.Key20.level₅.shape)
    (openKey₅ : Akita.OpeningKey Akita.Key20.level₅.opening)
    (block₅ : Fin Akita.Key20.level₅.shape.blocks → Coord)
    (decode₅ : E → Option (Akita.Rq Akita.Key20.level₅.shape.innerDegree))
    (weight₅ : ℕ)
    (matrix : Matrix (Fin Akita.Key20.terminal.innerRank)
          (Fin Akita.Key20.terminal.blockLength × Fin Akita.Key20.terminal.sourceDigits)
          (Akita.Rq Akita.Key20.terminal.innerDegree))
    (blockL : Fin Akita.Key20.terminal.blocks → Coord)
    (decodeL : E → Option (Akita.Rq Akita.Key20.terminal.innerDegree))
    {total : ℕ}
    (inside : ((Akita.Extract.committedTraceSq key₅ openKey₅
                  (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₅.vars
                  Akita.Key20.sized₅.left Akita.Key20.level₅.batch Akita.Key20.sized₅.right.left
                  block₅ decode₅ Akita.Extract.Page.read weight₅
                  Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
                  ((Akita.Extract.committedTraceSq key₄ openKey₄
                          (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                          Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                          block₄ decode₄ Akita.Extract.Page.read weight₄
                          Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                          ((Akita.Extract.committedTraceSq key₃ openKey₃
                                  (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                  Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
                                  Akita.Key20.sized₃.right.left block₃ decode₃
                                  Akita.Extract.Page.read weight₃ Akita.Key20.norm₃.operator
                                  Akita.Key20.norm₃.limit
                                  ((Akita.Extract.committedTrace key₂ openKey₂
                                          (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                          Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                                          decode₂ Akita.Extract.Page.read weight₂
                                          ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                                  (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                                  Akita.Key20.quotientDigits
                                                  Akita.Key20.level₁_sized.right.left
                                                  Akita.Key20.level₁.batch
                                                  Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                                  Akita.Extract.Page.read weight₁
                                                  (Akita.Extract.liftBatch β key openKey
                                                      (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                      Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                      Akita.Key20.quotientDigits Akita.Key20.root_room
                                                      Akita.Key20.rootBatch
                                                      Akita.Key20.root_sized.right.left block decode
                                                      Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                              Akita.Key20.level₁.tail).finish).tensored
                                      Akita.Key20.level₂.tail).finish).tensored
                              Akita.Key20.level₃.tail).finish).tensored
                      Akita.Key20.level₄.tail).finish).tensored
              Akita.Key20.level₅.tail).finish <
          total)
    (accept : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total → Prop)
    (table : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total → Option Coord → E)
    (decodes : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (b : Akita.Key20.root.Block), (decode b.fst (table input (some (block b)))).isSome = true)
    (decodes₁ : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.level₁.shape.blocks), (decode₁ (table input (some (block₁ i)))).isSome = true)
    (decodes₂ : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.level₂.shape.blocks), (decode₂ (table input (some (block₂ i)))).isSome = true)
    (decodes₃ : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.level₃.shape.blocks), (decode₃ (table input (some (block₃ i)))).isSome = true)
    (decodes₄ : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.level₄.shape.blocks), (decode₄ (table input (some (block₄ i)))).isSome = true)
    (decodes₅ : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.level₅.shape.blocks), (decode₅ (table input (some (block₅ i)))).isSome = true)
    (decodesL : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total)
          (i : Fin Akita.Key20.terminal.blocks), (decodeL (table input (some (blockL i)))).isSome = true)
    (worked : ∀ (input : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → E) total),
          table input none ∈ Akita.Extract.workZeros E 57) :
    ∃ proof,
      Akita.Extract.towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) Akita.Extract.Route.coefficient
        (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit)
        (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit)
        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit)
        (Akita.Extract.liftBatch β key openKey (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
          Akita.Key20.rootVars Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
          Akita.Key20.root_room Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
          Akita.Extract.Page.read mass (0 + 1))
        ((Akita.Extract.liftFold β₁ key₁ openKey₁ (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
              Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
              Akita.Key20.level₁_sized.right.right.left block₁ decode₁ Akita.Extract.Page.read weight₁
              (Akita.Extract.liftBatch β key openKey (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                  Akita.Key20.rootVars Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
                  Akita.Key20.root_room Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
                  Akita.Extract.Page.read mass (0 + 1)).finish).tensored
          Akita.Key20.level₁.tail)
        ((Akita.Extract.committedTrace key₂ openKey₂ (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₂.vars Akita.Key20.sized₂.left Akita.Key20.level₂.batch
              Akita.Key20.sized₂.right.left block₂ decode₂ Akita.Extract.Page.read weight₂
              ((Akita.Extract.liftFold β₁ key₁ openKey₁ (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                      Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
                      Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
                      Akita.Key20.level₁_sized.right.right.left block₁ decode₁ Akita.Extract.Page.read
                      weight₁
                      (Akita.Extract.liftBatch β key openKey
                          (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.rootVars
                          Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits Akita.Key20.root_room
                          Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
                          Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                  Akita.Key20.level₁.tail).finish).tensored
          Akita.Key20.level₂.tail)
        ((Akita.Extract.committedTraceSq key₃ openKey₃ (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
              Akita.Key20.sized₃.right.left block₃ decode₃ Akita.Extract.Page.read
              weight₃ Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
              ((Akita.Extract.committedTrace key₂ openKey₂
                      (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₂.vars
                      Akita.Key20.sized₂.left Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                      decode₂ Akita.Extract.Page.read weight₂
                      ((Akita.Extract.liftFold β₁ key₁ openKey₁
                              (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₁.vars
                              Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
                              Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
                              Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                              Akita.Extract.Page.read weight₁
                              (Akita.Extract.liftBatch β key openKey
                                  (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.rootVars
                                  Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
                                  Akita.Key20.root_room Akita.Key20.rootBatch
                                  Akita.Key20.root_sized.right.left block decode Akita.Extract.Page.read
                                  mass (0 + 1)).finish).tensored
                          Akita.Key20.level₁.tail).finish).tensored
                  Akita.Key20.level₂.tail).finish).tensored
          Akita.Key20.level₃.tail)
        ((Akita.Extract.committedTraceSq key₄ openKey₄ (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₄.vars Akita.Key20.sized₄.left Akita.Key20.level₄.batch
              Akita.Key20.sized₄.right.left block₄ decode₄ Akita.Extract.Page.read
              weight₄ Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
              ((Akita.Extract.committedTraceSq key₃ openKey₃
                      (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₃.vars
                      Akita.Key20.sized₃.left Akita.Key20.level₃.batch Akita.Key20.sized₃.right.left
                      block₃ decode₃ Akita.Extract.Page.read weight₃
                      Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
                      ((Akita.Extract.committedTrace key₂ openKey₂
                              (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₂.vars
                              Akita.Key20.sized₂.left Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left
                              block₂ decode₂ Akita.Extract.Page.read weight₂
                              ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                      (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                      Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                      Akita.Key20.quotientDigits Akita.Key20.level₁_sized.right.left
                                      Akita.Key20.level₁.batch Akita.Key20.level₁_sized.right.right.left
                                      block₁ decode₁ Akita.Extract.Page.read weight₁
                                      (Akita.Extract.liftBatch β key openKey
                                          (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                          Akita.Key20.quotientDigits Akita.Key20.root_room
                                          Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block
                                          decode Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                  Akita.Key20.level₁.tail).finish).tensored
                          Akita.Key20.level₂.tail).finish).tensored
                  Akita.Key20.level₃.tail).finish).tensored
          Akita.Key20.level₄.tail)
        ((Akita.Extract.committedTraceSq key₅ openKey₅ (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₅.vars Akita.Key20.sized₅.left Akita.Key20.level₅.batch
              Akita.Key20.sized₅.right.left block₅ decode₅ Akita.Extract.Page.read
              weight₅ Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
              ((Akita.Extract.committedTraceSq key₄ openKey₄
                      (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                      Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                      block₄ decode₄ Akita.Extract.Page.read weight₄
                      Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                      ((Akita.Extract.committedTraceSq key₃ openKey₃
                              (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₃.vars
                              Akita.Key20.sized₃.left Akita.Key20.level₃.batch Akita.Key20.sized₃.right.left
                              block₃ decode₃ Akita.Extract.Page.read weight₃
                              Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
                              ((Akita.Extract.committedTrace key₂ openKey₂
                                      (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                      Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                      Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂ decode₂
                                      Akita.Extract.Page.read weight₂
                                      ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                              (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                              Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                              Akita.Key20.quotientDigits Akita.Key20.level₁_sized.right.left
                                              Akita.Key20.level₁.batch
                                              Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                              Akita.Extract.Page.read weight₁
                                              (Akita.Extract.liftBatch β key openKey
                                                  (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                  Akita.Key20.quotientDigits Akita.Key20.root_room
                                                  Akita.Key20.rootBatch Akita.Key20.root_sized.right.left
                                                  block decode Akita.Extract.Page.read mass
                                                  (0 + 1)).finish).tensored
                                          Akita.Key20.level₁.tail).finish).tensored
                                  Akita.Key20.level₂.tail).finish).tensored
                          Akita.Key20.level₃.tail).finish).tensored
                  Akita.Key20.level₄.tail).finish).tensored
          Akita.Key20.level₅.tail)
        (Akita.Key20.terminalOf total matrix blockL decodeL
          ((Akita.Extract.committedTraceSq key₅ openKey₅ (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                  Akita.Key20.level₅.vars Akita.Key20.sized₅.left Akita.Key20.level₅.batch
                  Akita.Key20.sized₅.right.left block₅ decode₅
                  Akita.Extract.Page.read weight₅ Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
                  ((Akita.Extract.committedTraceSq key₄ openKey₄
                          (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                          Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                          block₄ decode₄ Akita.Extract.Page.read weight₄
                          Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                          ((Akita.Extract.committedTraceSq key₃ openKey₃
                                  (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                  Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
                                  Akita.Key20.sized₃.right.left block₃ decode₃
                                  Akita.Extract.Page.read weight₃ Akita.Key20.norm₃.operator
                                  Akita.Key20.norm₃.limit
                                  ((Akita.Extract.committedTrace key₂ openKey₂
                                          (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                          Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                                          decode₂ Akita.Extract.Page.read weight₂
                                          ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                                  (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                                  Akita.Key20.quotientDigits
                                                  Akita.Key20.level₁_sized.right.left
                                                  Akita.Key20.level₁.batch
                                                  Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                                  Akita.Extract.Page.read weight₁
                                                  (Akita.Extract.liftBatch β key openKey
                                                      (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                      Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                      Akita.Key20.quotientDigits Akita.Key20.root_room
                                                      Akita.Key20.rootBatch
                                                      Akita.Key20.root_sized.right.left block decode
                                                      Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                              Akita.Key20.level₁.tail).finish).tensored
                                      Akita.Key20.level₂.tail).finish).tensored
                              Akita.Key20.level₃.tail).finish).tensored
                      Akita.Key20.level₄.tail).finish).tensored
              Akita.Key20.level₅.tail).finish)
        0 Akita.Extract.Page.read (Akita.Key20.Note.commit ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.value ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.place ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.quarters.left
        Akita.Key20.quarters.right.left Akita.Key20.quarters.right.right.left
        Akita.Key20.quarters.right.right.right.left Akita.Key20.quarters.right.right.right.right
        (Akita.Key20.Note.commit₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.state ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        ((Akita.Extract.transcript total accept).final (RandomOracle.OracleComp.done proof) table) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.zero_accepts E fieldE fintypeE decidableEqE algebraZqE
    degree2EAlgebraZqE Coord fintypeCoord decidableEqCoord β key openKey block decode mass β₁ key₁ openKey₁
    block₁ decode₁ weight₁ key₂ openKey₂ block₂ decode₂ weight₂ key₃ openKey₃ block₃ decode₃ weight₃ key₄
    openKey₄ block₄ decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ matrix blockL decodeL total inside
    accept table decodes decodes₁ decodes₂ decodes₃ decodes₄ decodes₅ decodesL worked

/-- Soundness of the final verifier of `SPEC.md` Algorithm 10 with the Akita openings of
`crates/ns-compress/artifacts/key20.aks` (Akita Theorem 10.17 for the compression argument over
the opening). It does not cover `key30.aks`. The model is one uniform table of stage answers
with independent slots, and the verifier is the Lean verifier `FinalKey20.finalAccepts` with the
Lean decoders `draw`, which differs from the pinned Akita code. For each prover with at most `Q`
queries, the probability that the verifier accepts and no proof with witnesses satisfies
`Verifies` for the statement is at most `FinalKey20.shapesBound` at the slot probability `1/q²`:
for each claim shape, `(Q + 1)·(runCost + 317735/(1024·q²) + 49/(8·q²)) + (Q + 1)·144/budget + breakChance`.
Lean gives no bound on `breakChance`, the probability of a solution of Module-SIS (`[MSIS]`).
`drawn`: the conditions `FinalKey20.Opening.Drawn` of each opening, with the operator bound of
the filtered families. -/
theorem key20_verifier_sound
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
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 512)
    (coords : 28 ≤ Fintype.card Coord)
    {keep₃ : ι → Akita.Rq Akita.Key20.level₃.shape.innerDegree → Prop}
    {keep₄ : ι → Akita.Rq Akita.Key20.level₄.shape.innerDegree → Prop}
    {keep₅ : ι → Akita.Rq Akita.Key20.level₅.shape.innerDegree → Prop}
    {keepL : ι → Akita.Rq Akita.Key20.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (keep₃ index) (keep₄ index) (keep₅ index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        if
            FinalKey20.finalAccepts setup complete.packageWidth step cap control read head kind family
                (FinalKey20.played prover table) ∧
              ¬∃ proof,
                  RunCompleteness.Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
                    control (head (FinalKey20.played prover table).first).statement proof then
          1
        else 0) ≤
      FinalKey20.shapesBound setup complete.packageWidth step cap control read head family prover
        (1 / (↑fieldModulus : ℝ) ^ 2) queries budget :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.verifier_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family prover finiteWire relation cubeFits exports complete rows blocks coords keep₃ keep₄ keep₅ keepL
    drawn queries bounded budget positive

/-- The final verifier with the Akita openings of `crates/ns-compress/artifacts/key20.aks` against a
false history: `Claims.Final.history_sound` in the game with Akita openings (Akita Theorem
10.17). It does not cover `key30.aks`. The model and the verifier are those of
`key20_verifier_sound`. For each prover with at most `Q` queries, the probability that the
verifier accepts `(i, z0, zi)` and no `i` valid steps lead from `z0` to `zi` is at most
`FinalKey20.shapesBound` at the slot probability `1/q²` plus `IvcSecurity.walkBound` under
`[HN-A1]`. Lean gives no bound on the term `breakChance` of `shapesBound` (`[MSIS]`). -/
theorem key20_history_sound
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
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
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
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 512)
    (coords : 28 ≤ Fintype.card Coord)
    {keep₃ : ι → Akita.Rq Akita.Key20.level₃.shape.innerDegree → Prop}
    {keep₄ : ι → Akita.Rq Akita.Key20.level₄.shape.innerDegree → Prop}
    {keep₅ : ι → Akita.Rq Akita.Key20.level₅.shape.innerDegree → Prop}
    {keepL : ι → Akita.Rq Akita.Key20.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (keep₃ index) (keep₄ index) (keep₅ index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey20.adversary setup complete.packageWidth step cap control read head kind family
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Prefix Wire (Option Coord → K) total → Option Coord → K),
          FinalKey20.finalAccepts setup complete.packageWidth step cap control read head kind family
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((FinalKey20.adversary setup complete.packageWidth step cap control read head kind family prover
                  budget).tape.toOuterMeasure :
            Set
                (FinalKey20.adversary setup complete.packageWidth step cap control read head kind family
                    prover budget).Tape →
              ENNReal)
          {sample |
            FinalKey20.FalseHistory setup complete.packageWidth step cap control read head kind family
              prover exports.program sample.1}).toReal ≤
      FinalKey20.shapesBound setup complete.packageWidth step cap control read head family prover
          (1 / (↑fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey20.adversary setup complete.packageWidth step cap control read head kind family prover
            budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family prover finiteWire budget relation cubeFits exports Admitted StageAdmitted Efficient error
    assumption closed complete rows keysShaped blocks coords keep₃ keep₄ keep₅ keepL drawn queries bounded
    positive admitted depth depthBound

/-- The knowledge form of `key20_history_sound`: `Claims.Final.history_knowledge` in the game with
the Akita openings of `crates/ns-compress/artifacts/key20.aks` (Akita Theorem 10.17). It does
not cover `key30.aks`. The probability that the verifier accepts and the witnesses of the
reverse extractor are not a history of the accepted statement is at most `FinalKey20.shapesBound`
at the slot probability `1/q²` plus `IvcSecurity.walkBound`. The model, the verifier and the open
terms are those of `key20_history_sound`. -/
theorem key20_history_knowledge
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
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Prefix Wire (Option Coord → K) total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
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
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 512)
    (coords : 28 ≤ Fintype.card Coord)
    {keep₃ : ι → Akita.Rq Akita.Key20.level₃.shape.innerDegree → Prop}
    {keep₄ : ι → Akita.Rq Akita.Key20.level₄.shape.innerDegree → Prop}
    {keep₅ : ι → Akita.Rq Akita.Key20.level₅.shape.innerDegree → Prop}
    {keepL : ι → Akita.Rq Akita.Key20.terminal.innerDegree → Prop}
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (keep₃ index) (keep₄ index) (keep₅ index) (keepL index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey20.adversary setup complete.packageWidth step cap control read head kind family
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Prefix Wire (Option Coord → K) total → Option Coord → K),
          FinalKey20.finalAccepts setup complete.packageWidth step cap control read head kind family
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((↑(HyperNova.ReverseStages.reverseStages assumption closed
                      (IvcSecurity.walkAdversary
                        (FinalKey20.adversary setup complete.packageWidth step cap control read head kind
                          family prover budget))
                      admitted depth) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalKey20.adversary setup complete.packageWidth step cap control read head kind
                            family prover budget))
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalKey20.AdviceFailure setup step cap control read head kind family prover budget exports
              assumption closed complete admitted depth tape}).toReal ≤
      FinalKey20.shapesBound setup complete.packageWidth step cap control read head family prover
          (1 / (↑fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey20.adversary setup complete.packageWidth step cap control read head kind family prover
            budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_knowledge_drawn logicalWidth publicFits mode
    setup step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι
    fintypeι kind family prover finiteWire budget relation cubeFits exports Admitted StageAdmitted
    Efficient error assumption closed complete rows keysShaped blocks coords keep₃ keep₄ keep₅ keepL drawn
    queries bounded positive admitted depth depthBound

/-- The numbers of the three opening claims of the compression argument under the key of 512
blocks, which is the key of `crates/ns-compress/artifacts/key20.aks`. For a setup whose claims
pass the layout of `SPEC.md` Algorithm 10, step 5 (`use`), the point of the key polynomial has
20 coordinates, the point of `W` has 16, and the point of `V` has 16. These are the table sizes
`2^20`, `2^16` and `2^16` of the three groups of the Lean schedule `Key20.root`. -/
theorem key20_points
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : FinalVerifier.Claims logicalWidth publicFits)
    (use : FinalKey20.Use setup shape) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars + (setup.params shape).rowVars = 20 ∧
      (setup.params shape).witnessVars = 16 ∧
        (setup.params shape).spanVars + (setup.verifier shape).columnVars + (setup.verifier shape).zeros =
            16 ∧
          (setup.params shape).spanVars ≤ 14 :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.key20_points logicalWidth publicFits mode setup shape
    use

/-- The block counts of the Lean schedule of `crates/ns-compress/artifacts/key20.aks` agree with
the points of its links (Akita Section 4.5, (40) to (42)). The live blocks of the levels 1 to 5
and of the terminal are at most `2^{r_blk}`, and the segments of the point of each link have the
sum of the coordinates of the final claim of its sender. `tests/AkitaSchedules.lean` compares
the numbers of the Lean schedule with the file. -/
theorem key20_blocks :
    (Akita.Key20.level₁.shape.blocks ≤ 2 ^ 5 ∧ 4 + (6 + (7 + 5)) = Akita.Key20.rootVars) ∧
      (Akita.Key20.level₂.shape.blocks ≤ 2 ^ 5 ∧ 6 + (9 + 5) = Akita.Key20.level₁.tail) ∧
        (Akita.Key20.level₃.shape.blocks ≤ 2 ^ 4 ∧ 6 + (8 + 4) = Akita.Key20.level₂.tail) ∧
          (Akita.Key20.level₄.shape.blocks ≤ 2 ^ 3 ∧ 5 + (9 + 3) = Akita.Key20.level₃.tail) ∧
            (Akita.Key20.level₅.shape.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = Akita.Key20.level₄.tail) ∧
              Akita.Key20.terminal.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = Akita.Key20.level₅.tail :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.blocks_live

/-- Ring descent for Module-SIS in the norm `‖·‖_∞`, for `q = 2^64 − 59` and each pair of ring
degrees with `stride · d_car = d` (Akita (38) and (39)). `solve` is a solver for a matrix of `n`
rows and `m` columns over `R_{q,d}` at the bound `β`. `ShortKernel.lower` is a solver for a matrix
of `n` rows and `stride · m` columns over `R_{q,d_car}` at the same bound: it runs `solve` on the
matrix whose entries have the columns of its input as coordinates in the basis
`1, X, …, X^{stride−1}`, and it returns the coordinates of the solution. The two solvers have one
probability on uniform matrices. So an instance at the degree `d` is not easier than the instance
at the degree `d_car` with `stride` times the width. The claim does not give the other direction,
and it gives no number for the hardness of an instance. -/
theorem module_sis_descent
    {d : ℕ}
    {dCar : ℕ}
    {stride : ℕ}
    [neZeroNatD : NeZero d]
    [neZeroNatDCar : NeZero dCar]
    {rows : Type}
    {cols : Type}
    [fintypeRows : Fintype rows]
    [decidableEqRows : DecidableEq rows]
    [fintypeCols : Fintype cols]
    [decidableEqCols : DecidableEq cols]
    (tower : stride * dCar = d)
    {bound : ℕ}
    (solve : (M : Matrix rows cols (Akita.Rq d)) → Option (Akita.ShortKernel M bound)) :
    (𝔼 N, if (Akita.ShortKernel.lower tower solve N).isSome then (1 : ℝ) else 0) =
      𝔼 M, if (solve M).isSome then (1 : ℝ) else 0 :=
  @NightstreamFPrime.Spec.Compression.Akita.ShortKernel.lower_chance d dCar stride neZeroNatD
    neZeroNatDCar rows cols fintypeRows decidableEqRows fintypeCols decidableEqCols tower bound solve

/-- The instances of Module-SIS of the Akita opening of `crates/ns-compress/artifacts/key20.aks`
(Akita Lemma 10.20 and Appendix G). `Key20.instances` has 13 instances `(d, n, m, p, η)`: the ring
degree, the rank, the width, the norm (`true` for `‖·‖_2`, with `η` a bound on the square) and
the bound that the file prices. `Key20.views` has the instance of each of the 43 matrices of the
Lean tower at the bound of its witness of `R_bind`. Each view is below an instance of the list:
one degree, one rank, one norm, a width that is not larger and a bound that is not larger. Each
instance has the width of a view below it, with the bound of this view in the norm `‖·‖_∞` and
with a bound below two times the bound of this view in the norm `‖·‖_2`. No instance is below
another instance. `tests/AkitaSchedules.lean` compares the list with the file. Lean does not
prove the hardness of an instance, and the list is not for `key30.aks`. -/
theorem key20_instance_list :
    Akita.Covered Akita.Key20.instances Akita.Key20.views ∧
      Akita.Key20.instances.map Akita.Setup.Instance.key =
          [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
            (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
            (128, 1, 1936, false, 7), (128, 1, 1092, false, 31), (128, 4, 512, false, 126852),
            (128, 4, 256, true, 2 ^ 44), (512, 2, 56, false, 53477172),
            (1024, 1, 224, false, 32505732), (1024, 1, 56, false, 53477172)] ∧
        Akita.Key20.views.length = 43 ∧
          (∀ I ∈ Akita.Key20.instances,
              ∃ v ∈ Akita.Key20.views,
                v.Below I ∧ v.cols = I.cols ∧ cond I.euclidean (I.bound < 2 * v.bound) (v.bound = I.bound)) ∧
            List.Pairwise (fun I J => ¬I.Below J ∧ ¬J.Below I) Akita.Key20.instances :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.instances_exact

/-- The list `Key20.instances` is complete for the term `breakChance` of the claims
`key20_verifier_sound`, `key20_history_sound` and `key20_history_knowledge` (Akita Lemma 10.20).
`solved`: `out` is an output of the extractor that is a solution of Module-SIS, which is the
event of `FinalAkita.breakChance`: a solution for the key commitment, or a witness of `R_bind`
for a matrix of the root, of a level or of the terminal. Then a matrix `M` of the opening has a
solution that gives a solution of an instance of the list (`Key20.Reduced`, `Reduces`). The
instance has the degree, the rank and the norm of `M`. Each matrix of the instance that has the
columns of `M` at some positions has a solution at the bound of the instance, which is the
vector for `M` with zeros in the other columns. So these 13 instances are the Module-SIS
instances that the `key20_` claims rest on. `mass`, `weight₁` and `weight₂`: the opening has the
norms `‖c‖_1` of the fold challenges that `FinalKey20.Opening.Drawn` gives, 31, 51 and 51 for the
groups of the root, 51 for the level 1 and 31 for the level 2. `inner`: the bound of the
reduction for the key commitment is at most `η_A` of the key group, 32505732. Lean does not prove the hardness of an
instance, and it has no bound on `breakChance`: the matrices of the claims are fixed, and no Lean
result joins this claim with the sum over the instances of Akita (168) for a uniform setup. -/
theorem key20_instances
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    {setup : FinalVerifier.Setup mode logicalWidth publicFits}
    {packageWidths : Fin 2 → ℕ}
    {step : AppState → AppWitness → AppState}
    {cap : ℕ}
    {control : AppState → AppWitness → Fin 2}
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    {read : Akita.Extract.Reader Wire}
    {head : Wire → FinalAkita.First logicalWidth publicFits}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey20.Opening logicalWidth publicFits Wire Coord total}
    (mass : o.mass = Akita.Key20.rootMass)
    (weight₁ : o.weight₁ = 51)
    (weight₂ : o.weight₂ = 31)
    (inner : o.innerBound ≤ (Akita.Key20.root.fold 0).innerBound (Akita.Key20.root.shape 0))
    {out : Akita.Extract.Opened Akita.Key20.root.Weak
            (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₁.opening)
              (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₂.opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₅.opening)
                      (Akita.Extract.Prefix Wire (Option Coord → K) total ×
                        Option (Akita.TerminalFork Akita.Key20.terminal))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * o.mass g) * (Akita.Key20.root.fold g).sourceBound (Akita.Key20.root.shape g)) ⊕
            Akita.Extract.LevelBreak o.key₁ o.openKey₁
                (2 * (2 * o.weight₁) * Akita.Key20.level₁.fold.sourceBound Akita.Key20.level₁.shape) ⊕
              Akita.Extract.Route.Break o.key₂ o.openKey₂ o.weight₂
                  (Akita.Key20.level₂.fold.sourceBound Akita.Key20.level₂.shape)
                  Akita.Extract.Route.coefficient ⊕
                Akita.Extract.Route.Break o.key₃ o.openKey₃ o.weight₃
                    (Akita.Key20.level₃.fold.sourceBound Akita.Key20.level₃.shape)
                    (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ o.weight₄
                      (Akita.Key20.level₄.fold.sourceBound Akita.Key20.level₄.shape)
                      (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit) ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ o.weight₅
                        (Akita.Key20.level₅.fold.sourceBound Akita.Key20.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit) ⊕
                      Option
                        ((Akita.Extract.Route.euclidean Akita.Key20.normTerminal.operator
                              Akita.Key20.normTerminal.limit).Kernel
                          o.terminal.matrix o.terminal.weight o.terminal.bound)}
    (solved : FinalAkita.Solved (FinalKey20.Opening.tower setup packageWidths step cap control read head o)
          o.key o.honest o.innerBound out) :
    Akita.Key20.Reduced Akita.Key20.instances o.key o.openKey o.key₁ o.openKey₁ o.key₂ o.openKey₂ o.key₃
      o.openKey₃ o.key₄ o.openKey₄ o.key₅ o.openKey₅ o.terminal.matrix :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.Opening.reduced logicalWidth publicFits mode setup
    packageWidths step cap control Wire Coord total read head fintypeCoord decidableEqCoord o mass weight₁
    weight₂ inner out solved

end NightstreamFPrime.Claims.Akita
