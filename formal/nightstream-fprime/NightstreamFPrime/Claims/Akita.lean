import NightstreamFPrime.Export.BoundedDepth.FinalKey20
import NightstreamFPrime.Spec.Checkpoint.Extraction
import NightstreamFPrime.Spec.Checkpoint.Runs
import NightstreamFPrime.Spec.Compression.Akita.Instance
import NightstreamFPrime.Spec.Compression.Akita.ShortUnit
import NightstreamFPrime.Spec.Compression.Akita.TraceFold

/-! # Claims: the Akita polynomial commitment scheme

The results of Akita (ePrint 2026/1983, revision of 2026-11-23) that hold for each schedule: the
extractor of Akita Theorem 10.17, the weak binding and the extraction of one fold, the units of
short ring elements, and the ring descent for Module-SIS (`SPEC.md` section 9; `SECURITY_MODEL.md`
adversary 6 and trust item 7). The claims for the schedule file
`crates/ns-compress/artifacts/key20.aks` are in `Claims.AkitaKey20`, and the claims for
`crates/ns-compress/artifacts/key30.aks` are in `Claims.AkitaKey30`.

- Model: the answers of the stages are one uniform table on the keys of the stage inputs (their
  messages) with independent slots, and an answer of one stage has one slot for each fold
  coordinate. One query returns all the slots of a key. The paper counts a query for each
  indexed coordinate stream (Akita Section 10.3), and no Lean result relates the two counts, so
  the bounds of the claims are Nightstream results and not the bounds of the paper.
- Assumptions: none. When an extractor of a claim can return a solution of Module-SIS, the claim
  gives no bound on the probability of that output. -/

namespace NightstreamFPrime.Claims.Akita

open Spec Spec.Compression Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped BigOperators Spec.FieldExtensionRing Spec.Compression.Production
attribute [local instance low] Classical.propDecidable
attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK


/-- Akita Theorem 10.17 in the form (161), for a fixed environment, as a Nightstream result in the
query count of the model. The model is one uniform table of answers on keys with independent
slots, and a limit `budget` on the attempts of each retry loop of the extractor. For each prover
with at most `Q` queries to the table and each structure with the conditions of Akita Definition
10.16 (`lawful`), the probability that the verifier accepts and the extractor returns no output
is at most `(Q + 1)·ε_ck + (Q + 1)·L/budget`. `ε_ck` is the
sum of the stage errors and `L` is the number of retry loops. The term `(Q + 1)·L/budget` is not
in the paper, whose extractor has an expected run time. Akita Definition 10.16 has
`e_r = Σ_u 1/|S_{r,u}|` at an indexed stage, and Akita Remark 10.18 gives the work credit
`2^{−g_r}` to a bad answer at a checkpoint or plain stage. The Lean form gives this credit
(`chance`) to the indexed stages too, and `weight` is a bound on the largest point mass of a
decoder (`lawful.key_mass`), not `1/|S_{r,u}|`; the `key20_` and `key30_` claims use no credit at
the indexed stages. -/
theorem checkpoint_extraction
    {Addr : Type}
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [fintypeAddr : Fintype Addr]
    [decidableEqAddr : DecidableEq Addr]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    (structure_ : Checkpoint.Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Addr (Coord → Slot) Proof)
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
  @NightstreamFPrime.Spec.Checkpoint.checkpoint_extraction_le Addr Point Coord Slot Proof Cert Break Cand
    Key Witness fintypeAddr decidableEqAddr fintypePoint decidableEqPoint fintypeCoord decidableEqCoord fintypeSlot nonemptySlot structure_
    prover lawful queries bounded budget positive

/-- Akita Theorem 10.17, the number of runs. In the model of `checkpoint_extraction`, the extractor
runs the prover at most `K̂ + (Q + 1)·(K̂ − 1)` times in expectation, for the tree size `K̂` of the
structure and for each limit `budget`. The claim gives no bound on the run time. -/
theorem checkpoint_runs
    {Addr : Type}
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [fintypeAddr : Fintype Addr]
    [decidableEqAddr : DecidableEq Addr]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    (structure_ : Checkpoint.Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Addr (Coord → Slot) Proof)
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (budget : ℕ) :
    𝔼 table, 𝔼 tape, (↑(Checkpoint.extractRuns structure_ prover budget table tape) : ℝ) ≤
      (↑structure_.treeSize : ℝ) + ((↑queries : ℝ) + 1) * ((↑structure_.treeSize : ℝ) - 1) :=
  @NightstreamFPrime.Spec.Checkpoint.extract_runs_le Addr Point Coord Slot Proof Cert Break Cand Key
    Witness fintypeAddr decidableEqAddr fintypePoint decidableEqPoint fintypeCoord decidableEqCoord fintypeSlot nonemptySlot structure_ prover
    queries bounded budget

/-- The output of the extractor of Akita Theorem 10.17 (Akita Definition 10.15). For a structure
with the conditions of Akita Definition 10.16 (`lawful`), each output of the extractor is with an
accepted transcript of the prover, and it is a witness of the relation for the statement of the
prover or a witness of `R_bind`. The claim gives no bound on the probability of a witness of
`R_bind`. -/
theorem checkpoint_sound
    {Addr : Type}
    {Point : Type}
    {Coord : Type}
    {Slot : Type}
    {Proof : Type}
    {Cert : Type}
    {Break : Type}
    {Cand : Type}
    {Key : Type}
    {Witness : Type}
    [decidableEqAddr : DecidableEq Addr]
    [decidableEqPoint : DecidableEq Point]
    [decidableEqCoord : DecidableEq Coord]
    (structure_ : Checkpoint.Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
    (prover : RandomOracle.OracleComp Addr (Coord → Slot) Proof)
    [fintypeCoord : Fintype Coord]
    [fintypeSlot : Fintype Slot]
    (lawful : structure_.Lawful)
    (budget : ℕ)
    (table : Addr → Coord → Slot)
    (tape : Checkpoint.Tape Addr Coord Slot budget structure_.protocol.stages)
    (out : Cert ⊕ Break)
    (result : Checkpoint.extract structure_ prover budget table tape = some out) :
    structure_.protocol.accept (structure_.protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          structure_.relation (structure_.protocol.input prover table 0) (structure_.witness certificate))
        structure_.breaks out :=
  @NightstreamFPrime.Spec.Checkpoint.extract_sound Addr Point Coord Slot Proof Cert Break Cand Key
    Witness decidableEqAddr decidableEqPoint decidableEqCoord structure_ prover fintypeCoord fintypeSlot lawful budget table tape
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

end NightstreamFPrime.Claims.Akita
