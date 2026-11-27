import NightstreamFPrime.Spec.Compression.Akita.Replay.Rows
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightTraceSpec

/-! The statements of one fold for the words that the replay reads (Akita revision `d50f700`,
`verify_fold` of `fold/mod.rs`; the words `data` of `Replayer.stage2Of`: the commitment rows of a
root group, the setup rows of the registry, the next witness of the level before and the opening
payload).

- `traceStatementFor`, `packStatementFor` and `batchStatementFor` are the statements of a trace
  level, of a packing level with one group and of a fold with opening groups: the payloads are
  `payloadOf` of the words of their runs, the value is `Σ_i ρ_i·c_i` of the row coefficients and
  the claims, the matrices are the views of the setup array (`flatOf`), and the points are the
  points of the replay (`pointOf`, with the segments of `PackGroup.segments`).
- `trace_fused_for`, `packing_fused_for` and `batch_fused_for` are `trace_fused`,
  `packing_fused` and `batch_fused` for these statements, with no hypothesis on the payloads and
  the value. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor}

section Trace

variable (O) in
/-- The statement of a trace level for the setup array, the values `V` of the replay and the
words `data` of its runs: `traceStatementOf` with the payloads of the run of the group (from
`source`) and of the run of the opening, and the value `Σ_i ρ_i·c_i`. -/
noncomputable def traceStatementFor (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (V : TraceValues) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) : TraceStatement O E :=
  traceStatementOf O quarter setup V
    (payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)))
    (payloadOf O.chain _ _ (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)))
    (dot coefficients claims).into

/-- `trace_fused` for the statement `traceStatementFor`. -/
theorem trace_fused_for [Fintype E] (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (V : TraceValues) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (rest : Ext) (challenge : Fin S.blocks → Rq S.innerDegree)
    (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (batch : ℕ) (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (τ : List Ext) (α : Ext) (length : τ.length = batch) :
    ((stage2 τ α data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims rest).fused.into : E) =
      ((traceReduction (traceStatementFor O quarter setup V data source coefficients claims)
        challenge μ fits batch rowsFit).rows α.into).batchTarget (τ.map Ext.into) + rest.into :=
  trace_fused _ challenge μ fits batch rowsFit τ α data source coefficients claims rest length rfl
    rfl rfl

end Trace

section Packing

variable (O) in
/-- The weights of the opening point of a group of the replay: the segments `r_blk`, `r_pos`,
`r_pack` and `r_tail` of `PackGroup.segments`. -/
noncomputable def pointOf (point : List Ext) : Point O E :=
  let pack := Nat.log 2 O.stride
  let tail := Nat.log 2 O.carrierDegree
  let position := Nat.log 2 S.blockLength
  Point.ofSegments O ((point.drop (pack + tail + position)).map Ext.into)
    (((point.drop (pack + tail)).take position).map Ext.into) ((point.take pack).map Ext.into)
    (((point.drop pack).take tail).map Ext.into)

variable (O) in
/-- The statement of a packing level with one group for the setup array, the opening point of
the replay and the words `data` of its runs. -/
noncomputable def packStatementFor (setup : Array Fq) (point : List Ext)
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext) :
    Akita.Statement O E :=
  ⟨Key.ofSetup (flatOf setup) S, OpeningKey.ofSetup (flatOf setup) O,
    payloadOf S.chain _ _
      (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)),
    payloadOf O.chain _ _ (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)),
    pointOf O point, (dot coefficients claims).into⟩

/-- `packing_fused` for the statement `packStatementFor`. -/
theorem packing_fused_for [Fintype E] (β : Module.Basis (Fin O.planes) Zq E) (setup : Array Fq)
    (point : List Ext) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (rest : Ext) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (digits : ℕ) (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (τ : List Ext) (α : Ext)
    (length : τ.length = batch) :
    ((stage2 τ α data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims rest).fused.into : E) =
      ((liftedReduction β (packStatementFor O setup point data source coefficients claims)
        challenge μ fits digits room batch rowsFit).rows α.into).batchTarget (τ.map Ext.into) +
        rest.into :=
  packing_fused β _ challenge μ fits digits room batch rowsFit τ α data source coefficients claims
    rest length rfl rfl rfl

end Packing

section Groups

variable {n : ℕ}

variable (B : Batch (Fin n)) in
/-- The statement of a fold with opening groups for the setup array, the opening point
`points g` and the row coefficient `ρ_g` of each group `g`, and the words `data` of the runs,
with the groups in the order `groupOrder` (`Batch.Statement.over` for the matrices `A`, `B` and
`D`, and the compression maps of `PathKey.ofSetup`). -/
noncomputable def batchStatementFor (setup : Array Fq) (points : ℕ → List Ext)
    (data : RhsRun → Array Fq) (source : Fin n → RhsSource) (coefficients claims : Array Ext) :
    B.Statement E where
  key g := Key.ofSetup (flatOf setup) (B.shape g)
  openKey := ⟨Setup.shared (flatOf setup) B.openDegree B.openRank _ B.openSetupCode,
    PathKey.ofSetup (flatOf setup) B.chain⟩
  payload g := payloadOf (B.shape g).chain _ _ (data (groupRun (fun k => B.shape (groupOrder n k))
    B.openRank B.chain source ((groupOrder n).symm g)))
  openPayload := payloadOf B.chain _ _
    (data (openRun (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain))
  point g := pointOf (B.opening g) (points g)
  weight g := (coefficients.getD g Ext.zero).into
  value := (dot coefficients claims).into

variable {B : Batch (Fin n)}

/-- `batch_fused` for the statement `batchStatementFor`. -/
theorem batch_fused_for [Fintype E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (setup : Array Fq) (points : ℕ → List Ext) (data : RhsRun → Array Fq)
    (source : Fin n → RhsSource) (coefficients claims : Array Ext) (rest : Ext)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (τ : List Ext) (α : Ext) (length : τ.length = batch) :
    ((stage2 τ α data
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
        coefficients claims rest).fused.into : E) =
      ((Batch.liftedReduction β (batchStatementFor B setup points data source coefficients claims)
        challenge μ fits base digits room batch rowsFit).rows α.into).batchTarget
        (τ.map Ext.into) + rest.into :=
  batch_fused β _ challenge μ fits base digits room batch rowsFit τ α data source coefficients
    claims rest length (fun k => by simp only [batchStatementFor, Equiv.symm_apply_apply]) rfl rfl

end Groups

end NightstreamFPrime.Spec.Compression.Akita.Replay
