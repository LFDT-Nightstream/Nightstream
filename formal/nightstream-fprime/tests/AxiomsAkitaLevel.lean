import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelIn
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! Axiom audits for the levels of the replay of Akita openings against the games of the
specification: the statements of the words of the replay (`Replay/LevelStatement.lean`), the
fused sum-check (`Replay/LevelFused.lean`), and the fold on the coefficient route, on the Euclidean
route and with the tensor reduction of the next level (`Replay/LevelFold.lean`), and the fold of a
packing level, of a level with opening groups and of an offloaded level
(`Replay/LevelPack.lean`), and these folds on the recorded values of a state of the replay
(`Replay/LevelIn.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceStatementFor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.trace_fused_for
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pointOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packStatementFor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packing_fused_for
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchStatementFor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batch_fused_for
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FusedEnd.withClaim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedRun_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.holds_iff_final
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFinish
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFinish_weighs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFinish_weighsSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFinish_supports
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofFn_listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangePoint_listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeRun_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeRun_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.normRun_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.binPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs_drawBin
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batched_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FoldWords
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.NormWords
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FoldWords.point
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldRunSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldPathSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldRunSq_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.TraceValues.words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedClaim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedClaimSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.TraceValues.norms
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFoldRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFoldRunSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedClaim_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedClaimSq_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFoldRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceFoldRunSq_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensoredRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensoredPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensored_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceTensored_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceTensoredSq_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packFinish
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PackingValues.words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packClaim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packFoldRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.supports_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packFoldRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchFoldRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batch_fused_over
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchOffloadRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.dataIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rowsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.claimsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage2In
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage2Of_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.wordsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.normsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.finishIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.finishIn_trace
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.finishIn_packing
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.finishIn_end
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.setupClaimIn_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.setupClaimIn_some
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.productsIn_arities
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.toList_firstIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage2In_plain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage2In_euclidean
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.offloadRunIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rangeCheckOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rangeCheckOf_plain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rangeCheckOf_euclidean
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stageValue_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.wordsIn_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.fusedRun_passing
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRun_if
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunSq_if
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.offloadRun_if
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.offloadRunIn_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldPathIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.offloadPathIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_trace_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_traceSq_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_pack_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_batch_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.offloadRunIn_runs

namespace AkitaLevelGuards

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The check `check` of the fold `i` of a layout. -/
def atFold (layout : Layout) (i : ℕ) (check : Fold → Bool) : Bool :=
  ((layout.folds[i]?).map check).getD false

/-- The numbers of a fold of the layout that the hypotheses of the level theorems read, against
the numbers of the level of the claims: `μ`, the batch width, `low`, the base and the stages of
the range proof (`arities`, `lengths`, `positive`), the compression digits (`compressed`) and
the route with its cap (`plain`, `euclidean`, `small`). -/
def foldFits (F : Fold) (vars batch low base : ℕ) (compressed : Bool) (limit : Option ℕ) : Bool :=
  F.vars == vars && F.batch == batch && F.low == low && 2 ^ F.logBasis == base &&
    (productStages F.logBasis).map (·.1) == rangeArities base &&
    leafDegree F.logBasis == rangeLeaf base && decide (1 ≤ rangeLeaf base) &&
    F.compressed == compressed && F.euclidean == limit.isSome && F.limit == limit.getD 0 &&
    decide (F.limit < NightstreamFPrime.Spec.fieldModulus)

/-- `foldFits` for a level with one group of the claims, and at a trace level `quarter`,
`digits` and the rounds `ν = vars − 1` of the tensor reduction of the next level. -/
def levelFits (layout : Layout) (i : ℕ) (L : Key20.Level) (limit : Option ℕ) (trace : Bool) :
    Bool :=
  atFold layout i (fun F => foldFits F L.vars L.batch L.opening.low L.shape.certifiedBase
    (L.opening.compressed L.fold) limit) &&
  (!trace || (4 * (L.shape.innerDegree / 4) == L.shape.innerDegree &&
    decide (0 < L.fold.responseDigits) &&
    (match layout.folds[i + 1]? with
      | some N => N.trace.map (·.1) == some (L.vars - 1)
      | none => layout.terminal.rounds == L.vars - 1)))

#guard atFold (layout20 0 0) 0 fun F =>
  foldFits F Key20.rootVars Key20.rootBatch Key20.root.low 8 Key20.root.compressed none &&
    F.stage3.isNone
#guard levelFits (layout20 0 0) 1 Key20.level₁ none false
#guard levelFits (layout20 0 0) 2 Key20.level₂ none true
#guard levelFits (layout20 0 0) 3 Key20.level₃ (some Key20.norm₃.limit) true
#guard levelFits (layout20 0 0) 4 Key20.level₄ (some Key20.norm₄.limit) true
#guard levelFits (layout20 0 0) 5 Key20.level₅ (some Key20.norm₅.limit) true
#guard Key30.rows.all fun r =>
  atFold (layout30 r 0 0) 0 (fun F => foldFits F Key30.rootVars Key30.rootBatch (Key30.root r).low 8
    (Key30.root r).compressed none && F.stage3 == some Key30.setupVars) &&
  atFold (layout30 r 0 0) 1 (fun F => foldFits F Key30.vars₁ Key30.batch₁ (Key30.level₁ r).low 16
    (Key30.level₁ r).compressed none) &&
  levelFits (layout30 r 0 0) 2 (Key30.level₂ r) none false &&
  levelFits (layout30 r 0 0) 3 (Key30.level₃ r) none false &&
  levelFits (layout30 r 0 0) 4 Key30.level₄ none true &&
  levelFits (layout30 r 0 0) 5 Key30.level₅ (some (Key30.norm₅ r).limit) true &&
  levelFits (layout30 r 0 0) 6 Key30.level₆ (some Key30.norm₆.limit) true

/-- The numbers of a fold against the shapes of its end check (`Fold.fused`), for the hypotheses
`based`, `low`, `compressed`, `leaf`, `arities`, `quarter` and `digits` of the theorems of
`Replay/LevelIn.lean`: at a trace level for its `TraceShape`, at a packing level with one group for
its slot, and with groups for the base `2^logBasis`. -/
def shapesFit (F : Fold) : Bool :=
  match F.fused with
  | .trace T => 2 ^ F.logBasis == T.shape.certifiedBase && F.low == T.opening.low &&
      F.compressed == T.opening.compressed T.fold &&
      leafDegree F.logBasis == rangeLeaf T.shape.certifiedBase &&
      (productStages F.logBasis).map (·.1) == rangeArities T.shape.certifiedBase &&
      4 * (T.shape.innerDegree / 4) == T.shape.innerDegree && decide (0 < T.fold.responseDigits)
  | .packing P => match P.slots with
      | [g] => 2 ^ F.logBasis == g.shape.certifiedBase && F.low == g.opening.low &&
          F.compressed == g.opening.compressed g.fold &&
          (productStages F.logBasis).map (·.1) == rangeArities g.shape.certifiedBase
      | _ => (productStages F.logBasis).map (·.1) == rangeArities (2 ^ F.logBasis)

/-- The kind of the end check of each fold (`shaped`: trace or packing) and whether it has
Stage 3 (`direct`, `offloaded`). -/
def kinds (layout : Layout) : List (Bool × Bool) :=
  layout.folds.map fun F => (match F.fused with | .trace _ => true | .packing _ => false,
    F.stage3.isSome)

#guard (layout20 0 0).folds.all shapesFit
#guard kinds (layout20 0 0) ==
  [(false, false), (false, false), (true, false), (true, false), (true, false), (true, false)]
#guard Key30.rows.all fun r => (layout30 r 0 0).folds.all shapesFit &&
  kinds (layout30 r 0 0) == [(false, true), (false, false), (false, false), (false, false),
    (true, false), (true, false), (true, false)]

/-- The hypothesis `two` of the trace theorems holds at the trace levels of both layouts. -/
example : [Key20.level₂, Key20.level₃, Key20.level₄, Key20.level₅, Key30.level₄, Key30.level₅,
    Key30.level₆].all fun L => decide (((2 * (L.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0) := by
  decide

end AkitaLevelGuards
