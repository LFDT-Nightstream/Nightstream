import NightstreamFPrime.Spec.Compression.Akita.Replay.RunMessages

/-! The games of the specification levels on the state of a run of the operations of a fold
(Akita revision `d50f700`, `Build.fold` of `Replay/Script.lean`; the game theorems of
`Replay/LevelIn.lean`).

- `gameShape` is the shape of the operations of a fold for the state hypotheses of its game: the
  challenges of `τ_0` and `τ_1`, the sum-checks of the product stages, of the leaf or norm stage
  and of `fused` (`roundsAt`), and one virtual evaluation on the Euclidean route. `LevelState` is
  these hypotheses on a state; `level_state`: a run of aligned operations of this shape without a
  stop gives them on its last state.
- `foldRunIn_trace_game`, `foldRunIn_traceSq_game`, `foldRunIn_batch_game` and
  `foldRunIn_pack_game`: when the run of the operations of a fold and of its `fused.end` has no
  stop, the game of the specification level (`foldCheck` of `traceReduction`,
  `Batch.liftedReduction` or `liftedReduction`, `foldCheckSq` on the Euclidean route) runs on the
  path of the recorded values to acceptance with the recorded `(fused.point,
  witness_evaluation)`. `offloadRunIn_game`: the same for `offloadCheck` at the offloaded root,
  with the result of Stage 3. The hypotheses that remain are facts of the layout, the
  weights of track P (`weighs`, `supported`) and the alignment of the messages. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- The shape of the operations `pre` of the fold `F` at `level` for the state hypotheses of its
game. -/
def gameShape (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) : Bool :=
  drawnCount ⟨level, "tau0", none⟩ pre == some F.vars &&
    drawnCount ⟨level, "tau1", none⟩ pre == some F.batch &&
    roundsAt pre ⟨level, "fused", none⟩ F.vars 3 &&
    ((productStages F.logBasis).map (·.1)).zipIdx.all (fun x =>
      roundsAt pre ⟨level, s!"range.{x.2}", none⟩ F.vars x.1) &&
    (if F.euclidean then
      roundsAt pre ⟨level, "norm.sumcheck", none⟩ F.vars (leafDegree F.logBasis + 1) &&
        extsCount ⟨level, "norm.virtual_evaluations", none⟩ pre == some 1
    else roundsAt pre ⟨level, "range.leaf", none⟩ F.vars (leafDegree F.logBasis))

namespace Replayer

/-- The state hypotheses of the games of the fold `F` at `level` on the state `s`. -/
structure LevelState (s : Replayer) (level : ℕ) (F : Fold) : Prop where
  vars : (recordedIn s ⟨level, "tau0", none⟩).size = F.vars
  tau : (recordedIn s ⟨level, "tau1", none⟩).size = F.batch
  stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
    Q.rounds.length = F.vars ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity
  fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = F.vars
  fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3
  leaf : F.euclidean = false →
    (recordedIn s ⟨level, "range.leaf.point", none⟩).toList =
        (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2) ∧
      (roundsIn s ⟨level, "range.leaf", none⟩).length = F.vars ∧
      ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = leafDegree F.logBasis
  norm : F.euclidean = true →
    (recordedIn s ⟨level, "norm.virtual_evaluations", none⟩).size = 1 ∧
      (recordedIn s ⟨level, "norm.sumcheck.point", none⟩).toList =
        (roundsIn s ⟨level, "norm.sumcheck", none⟩).map (·.2) ∧
      (roundsIn s ⟨level, "norm.sumcheck", none⟩).length = F.vars ∧
      ∀ x ∈ roundsIn s ⟨level, "norm.sumcheck", none⟩, x.1.length = leafDegree F.logBasis + 1

section State

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A run of aligned operations of the shape `gameShape` without a stop, from a state with no
recorded value and no rounds, gives the state hypotheses of the game on its last state. -/
theorem level_state (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (level : ℕ) (F : Fold)
    (shape : gameShape pre level F = true) (aligned : ReadsAligned messages pre s₀.next)
    (ok : (runFrom pub proof messages pre s₀).1 = .ok ()) :
    LevelState (runFrom pub proof messages pre s₀).2 level F := by
  simp only [gameShape, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at shape
  obtain ⟨⟨⟨⟨tau0, tau1⟩, fused⟩, stages⟩, route⟩ := shape
  have facts := fun (key : Key) (n d : ℕ) (at_ : roundsAt pre key n d = true) =>
    let ⟨e, shaped⟩ := roundsAt_sound at_
    rounds_facts pub proof messages s₀ fresh empty pre key n d e shaped aligned ok
  obtain ⟨fusedCount, fusedLengths, _⟩ := facts _ _ _ fused
  refine ⟨?_, ?_, ?_, fusedCount, fusedLengths, ?_, ?_⟩
  · rw [size_drawnCount pub proof messages _ _ _ tau0 ok, fresh]
    simp
  · rw [size_drawnCount pub proof messages _ _ _ tau1 ok, fresh]
    simp
  · intro Q member
    unfold productsIn at member
    obtain ⟨⟨arity, k⟩, inside, rfl⟩ := List.mem_map.mp member
    obtain ⟨count, lengths, _⟩ := facts _ _ _ (stages (arity, k) inside)
    exact ⟨count, lengths⟩
  · intro plain
    rw [plain] at route
    simp only [Bool.false_eq_true, ↓reduceIte] at route
    obtain ⟨count, lengths, point⟩ := facts _ _ _ route
    exact ⟨point.symm, count, lengths⟩
  · intro euclidean
    rw [euclidean] at route
    simp only [↓reduceIte, Bool.and_eq_true, beq_iff_eq] at route
    obtain ⟨norm, single⟩ := route
    obtain ⟨count, lengths, point⟩ := facts _ _ _ norm
    refine ⟨?_, point.symm, count, lengths⟩
    rw [size_extsCount pub proof messages _ _ _ single aligned ok, fresh]
    simp

end State

section Games

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- A trace level on the coefficient route: when the run of the operations `pre` of a fold and of
its `fused.end` has no stop, `foldCheck` of `traceReduction` runs on the path of the recorded values
of the state `s` at `fused.end` to acceptance with the recorded `(fused.point, witness_evaluation)`.
-/
theorem foldRunIn_trace_game (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true) (game : gameShape pre level F = true)
    (aligned : ReadsAligned messages pre s₀.next)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ())
    {s : Replayer} (state : (runFrom pub proof messages pre s₀).2 = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (based : 2 ^ F.logBasis = T.shape.certifiedBase) (low : F.low = T.opening.low)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase) :
    (foldCheck (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  subst state
  have st := level_state pub proof messages s₀ fresh empty pre level F game aligned
    ((runFrom_append_fst pub proof messages _ _ s₀).mp ran).1
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  have runs := foldRunIn_trace_runs (E := E) pub _ level F shaped expanded rows based low quarter
    two challenge challenged fits rowsFit w plain compressed virtual st.vars st.tau arities
    st.stages count (fun x member => (lengths x member).trans leaf) st.fusedCount st.fusedLengths
  rw [foldRunIn_of_run pub proof messages s₀ fresh empty pre key c level F shape ran] at runs
  exact runs

/-- A trace level on the Euclidean route (`foldCheckSq` with the norm stage). -/
theorem foldRunIn_traceSq_game (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true) (game : gameShape pre level F = true)
    (aligned : ReadsAligned messages pre s₀.next)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ())
    {s : Replayer} (state : (runFrom pub proof messages pre s₀).2 = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (low : F.low = T.opening.low) (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (positive : 1 ≤ rangeLeaf T.shape.certifiedBase) (small : F.limit < fieldModulus)
    (digits : 0 < T.fold.responseDigits) (euclidean : F.euclidean = true)
    (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase) :
    (foldCheckSq (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) F.limit
        (T.shape.normKernel T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  subst state
  have st := level_state pub proof messages s₀ fresh empty pre level F game aligned
    ((runFrom_append_fst pub proof messages _ _ s₀).mp ran).1
  obtain ⟨single, virtual, count, lengths⟩ := st.norm euclidean
  have runs := foldRunIn_traceSq_runs (E := E) pub _ level F shaped expanded rows low leaf
    quarter two challenge challenged fits rowsFit w positive small digits euclidean compressed
    single virtual st.vars st.tau arities st.stages count
    (fun x member => by rw [lengths x member, leaf]) st.fusedCount st.fusedLengths
  rw [foldRunIn_of_run pub proof messages s₀ fresh empty pre key c level F shape ran] at runs
  exact runs

/-- A fold with opening groups without Stage 3 (`foldCheck` of `Batch.liftedReduction`); `weighs`
and `supported` are the weight hypotheses of track P. -/
theorem foldRunIn_batch_game (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true) (game : gameShape pre level F = true)
    (aligned : ReadsAligned messages pre s₀.next)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ())
    {s : Replayer} (state : (runFrom pub proof messages pre s₀).2 = s)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (challenge : B.Challenge)
    (fits : B.cells ≤ 2 ^ F.vars) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ F.vars)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ F.batch) (w : Table) (plain : F.euclidean = false)
    (compressed : F.compressed = B.compressed)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (leaf : leafDegree F.logBasis = rangeLeaf base)
    (weighs : ∀ p : List Ext, p.length = F.vars →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) none p).weight.into : E) =
      mle (p.map Ext.into) (((Batch.liftedReduction β (batchStatementFor B setup
        (claimPointIn s level) (dataIn pub s level) source (rowsIn s level F)
          (claimsIn s level F)) challenge F.vars fits base digits room F.batch rowsFit).rows
            (firstIn s ⟨level, "alpha", none⟩).into).batchWeight
              ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = F.vars → B.compressed = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (B.gaps digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0) :
    (foldCheck (Batch.liftedReduction β (batchStatementFor B setup (claimPointIn s level)
        (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits
        base digits room F.batch rowsFit) base (B.gaps digits).support B.compressed w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  subst state
  have st := level_state pub proof messages s₀ fresh empty pre level F game aligned
    ((runFrom_append_fst pub proof messages _ _ s₀).mp ran).1
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  have runs := foldRunIn_batch_runs (E := E) pub _ level F shaped direct expanded source rows
    base based low β challenge fits room rowsFit w plain compressed virtual st.vars st.tau
    arities st.stages count (fun x member => (lengths x member).trans leaf) st.fusedCount
    st.fusedLengths weighs supported
  rw [foldRunIn_of_run pub proof messages s₀ fresh empty pre key c level F shape ran] at runs
  exact runs

/-- A packing level with one group without Stage 3 (`foldCheck` of `liftedReduction`); `weighs`
and `supported` are the weight hypotheses of track P. -/
theorem foldRunIn_pack_game (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true) (game : gameShape pre level F = true)
    (aligned : ReadsAligned messages pre s₀.next)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ())
    {s : Replayer} (state : (runFrom pub proof messages pre s₀).2 = s)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {S : Shape}
    {O : OpeningShape S} {fold : Descriptor}
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
      fun _ => .witness)
    (based : 2 ^ F.logBasis = S.certifiedBase) (low : F.low = O.low)
    (β : Module.Basis (Fin O.planes) Zq E) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (fits : O.cells fold ≤ 2 ^ F.vars) {digits : ℕ}
    (room : O.quotientRoom fold digits ≤ 2 ^ F.vars) (rowsFit : O.rowCount + 1 ≤ 2 ^ F.batch)
    (w : Table) (plain : F.euclidean = false) (compressed : F.compressed = O.compressed fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities S.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf S.certifiedBase)
    (weighs : ∀ p : List Ext, p.length = F.vars →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) none p).weight.into : E) =
      mle (p.map Ext.into) (((liftedReduction β (packStatementFor O setup (claimPointIn s level 0)
        (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)) challenge F.vars
          fits digits room F.batch rowsFit).rows (firstIn s ⟨level, "alpha", none⟩).into).batchWeight
            ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = F.vars → O.compressed fold = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (O.gaps fold digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0) :
    (foldCheck (liftedReduction β (packStatementFor O setup (claimPointIn s level 0)
        (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits
        digits room F.batch rowsFit) S.certifiedBase (O.gaps fold digits).support
        (O.compressed fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  subst state
  have st := level_state pub proof messages s₀ fresh empty pre level F game aligned
    ((runFrom_append_fst pub proof messages _ _ s₀).mp ran).1
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  have runs := foldRunIn_pack_runs (E := E) pub _ level F shaped direct expanded rows based low β
    challenge fits room rowsFit w plain compressed virtual st.vars st.tau arities st.stages count
    (fun x member => (lengths x member).trans leaf) st.fusedCount st.fusedLengths weighs supported
  rw [foldRunIn_of_run pub proof messages s₀ fresh empty pre key c level F shape ran] at runs
  exact runs

/-- The offloaded root of `key30.aks` (`offloadCheck` of `Batch.liftedReduction` over the setup
prefix `flat`): when the run
of the operations `pre` of the shape `offloadShape` and of its `fused.end` has no stop, Stage 3
gives `some o` and `offloadCheck` runs on the path of the recorded values to acceptance with the
recorded `(fused.point, witness_evaluation)` and `o`. `weighs`, `supported` and `setupWeighs` are
the weight hypotheses of track P. -/
theorem offloadRunIn_game (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    (plan : SetupPlan) {r a e b t : ℕ} (shape : offloadShape pre level F plan r a e b t = true)
    (game : gameShape pre level F = true) {ν : ℕ}
    (setupShape : roundsAt pre ⟨level, "stage3", none⟩ ν 2 = true)
    (aligned : ReadsAligned messages pre s₀.next)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ())
    {s : Replayer} (state : (runFrom pub proof messages pre s₀).2 = s)
    {P : PackShape} (shaped : F.fused = .packing P) (offloaded : F.stage3 = some ν)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (challenge : B.Challenge)
    (fits : B.cells ≤ 2 ^ F.vars) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ F.vars)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ F.batch) (w : Table) (flat : Table)
    (scan : SetupScan (Batch.liftedReduction β ((batchStatementFor B setup (claimPointIn s level)
      (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)).over
        fun p : Fin (2 ^ ν) => flat p) challenge F.vars fits base digits room F.batch rowsFit) ν
      flat)
    (plain : F.euclidean = false) (compressed : F.compressed = B.compressed)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (leaf : leafDegree F.logBasis = rangeLeaf base)
    (weighs : ∀ p : List Ext, p.length = F.vars →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) (some (firstIn s ⟨level, "stage3.setup_claim", none⟩))
          p).weight.into : E) =
      scan.part (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) +
        scan.scale (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) *
          (firstIn s ⟨level, "stage3.setup_claim", none⟩).into)
    (supported : ∀ p : List Ext, p.length = F.vars → B.compressed = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (B.gaps digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0)
    (setupWeighs : ∀ r2 ρ : List Ext, r2.length = F.vars → ρ.length = ν →
      ((stage3Weight plan (firstIn s ⟨level, "alpha", none⟩)
        (recordedIn s ⟨level, "tau1", none⟩).toList r2 ρ).into : E) =
      mle (ρ.map Ext.into) (scan.weight (firstIn s ⟨level, "alpha", none⟩).into
        ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (r2.map Ext.into))) :
    ∃ o, stage3In s level plan = some o ∧
      (offloadCheck (Batch.liftedReduction β ((batchStatementFor B setup (claimPointIn s level)
          (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)).over
            fun p : Fin (2 ^ ν) => flat p) challenge F.vars fits base digits room F.batch rowsFit)
          base (B.gaps digits).support B.compressed w ν flat scan).Runs
        (offloadPathIn E pub s level F setup F.vars F.batch)
        (some (((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
            (firstIn s ⟨level, "witness_evaluation", none⟩).into),
          (o.1.map Ext.into, o.2.into))) := by
  subst state
  have okPre := ((runFrom_append_fst pub proof messages _ _ s₀).mp ran).1
  have st := level_state pub proof messages s₀ fresh empty pre level F game aligned okPre
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  obtain ⟨e', shaped'⟩ := roundsAt_sound setupShape
  obtain ⟨setupCount, setupLengths, _⟩ := rounds_facts pub proof messages s₀ fresh empty pre _ ν 2
    e' shaped' aligned okPre
  have runs := offloadRunIn_runs (E := E) pub _ level F plan shaped offloaded expanded source rows
    base based low β challenge fits room rowsFit w flat scan plain compressed virtual st.vars
    st.tau arities st.stages count (fun x member => (lengths x member).trans leaf) st.fusedCount
    st.fusedLengths setupCount setupLengths weighs supported setupWeighs
  obtain ⟨result, some_⟩ := offloadRunIn_of_run pub proof messages s₀ fresh empty pre key c level F
    plan shape ran
  obtain ⟨o, found⟩ := Option.isSome_iff_exists.mp some_
  rw [result, found] at runs
  exact ⟨o, found, runs⟩

end Games

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
