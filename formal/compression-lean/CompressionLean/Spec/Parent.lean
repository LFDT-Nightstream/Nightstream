import CompressionLean.Ports.Setup
import CompressionLean.Terminal.Soundness
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.OutputBinding

/-!
Owns what the claims say in F′'s terms.

- `StateWords`, `TerminalFor`: the statement of a final proof, and a
  terminal assignment for it.
- `ParentClaim`: what a final proof claims. Some terminal assignment holds a
  PiCCS proof that F′'s verifier accepts, and the PiRLC parent that the
  verifier computes has a witness of F′'s CE(B) relation.
- `ParentShape`, `HonestSetup`: the layer-1 sizes of F′'s PiRLC parent, and
  the layer-1 setup of F′'s key and matrices (`runEntry`).
- `RunsGive`: the runs that Rust pushes give F′'s matrices (`pushedEntry`).
-/

namespace CompressionLean.Spec

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Setup

/-! ## Maps from F′ -/

/-- Vertex `i mod 2^n` of the Boolean cube, in F′'s little-endian order. -/
def vertexAt (n i : ℕ) : BooleanVertex n :=
  NumericBooleanDomain.vertex n ⟨i % 2 ^ n, Nat.mod_lt _ (by positivity)⟩

/-- An F′ field value as a `Gl` value. `Spec.F` and `Gl` are the same type;
`gl` fixes the type that the arithmetic around it uses. -/
abbrev gl (x : F) : Gl := x

/-! ## The statement and the claim -/

/-- The statement of a final proof: the iteration, `z0` and the current
state, the nine words that the final program makes public
(`lifecycle/terminal/mod.rs:46-52`). -/
abbrev StateWords := Fin 9 → Spec.F

/-- The statement words of the prior state preimage, after the four context
words (`STATE_CONTEXT_START`, `lifecycle/inputs.rs:41-43`). -/
def priorStatement (env : Circuit.Env) : StateWords := fun k =>
  (Layout.Stage1.PiCCSInputs.priorStateWord
    (Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart + 4 + k.val)).eval
    (Layout.Stage1.Spartan.pullback env)

/-- A terminal assignment for the statement `x`, as the final program wires
it: every terminal row holds, the public context columns hold the circuit's
context digest, the statement words of the prior state preimage are `x`
(`terminal/package.rs:186-213`), and the iteration is not zero
(`terminal/mod.rs:60-68`). -/
def TerminalFor (authority : Lifecycle.VerifierContext.Authority) (x : StateWords)
    (env : Circuit.Env) : Prop :=
  (Terminal.circuitPackage ()).RowsHold env ∧
    Export.Stage1.Package.SelectedVerifierContext authority env ∧
    priorStatement env = x ∧ x 0 ≠ 0

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- The running claims of a terminal assignment, as F′'s accumulator reads
them from the package columns. -/
abbrev runningOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.running lw pf (Layout.Stage1.Spartan.pullback env)

/-- The fresh claim of a terminal assignment. -/
abbrev freshOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.fresh lw pf (Layout.Stage1.Spartan.pullback env)

/-- The PiCCS proof of a terminal assignment. -/
abbrev proofOf (relation : ProductionKey.LogicalRelation lw pf) (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.proof relation (Layout.Stage1.Spartan.pullback env)

/-- **What a final proof for `x` claims, in F′'s terms.** Some terminal
assignment for `x` (`TerminalFor`) holds a PiCCS proof that F′'s verifier
accepts for the key of `relation` and `ajtai`, and the PiRLC parent that the
verifier computes from it has a witness of F′'s CE(B) relation (SuperNeo
v1.2 Definition 21). -/
def ParentClaim (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧
    Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) (runningOf env) (freshOf env)
      (proofOf relation env) = true ∧
    (ProductionKey.key relation ajtai).parent (runningOf env) (freshOf env) (proofOf relation env) =
      some (Terminal.parent relation env) ∧
    ∃ a, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams (Terminal.parent relation env) a

/-! ## The layer-1 relation of the parent -/

/-- Entry `(i, c)` of matrix `j` as the runs of `S` give it: the sum of the
values `m_k·ratio^(c - start)` of the runs of matrix `j` and row `i` that
cover column `c` (`matrix.rs`, one `GeometricRowRun` per run). -/
noncomputable def runEntry (S : Layer1.Setup Gl) (D j i c : ℕ) : Gl :=
  ∑ k ∈ ((Finset.range S.runs).filter (fun k => S.runMatrix k = j ∧
      D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
      c < D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k))).filter
      (fun k => S.runRow k = i),
    S.runCoef k * S.slotRatio (S.runSlot k) ^ (c - (D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k)))

/-- The layer-1 shape of the CE(B) relation of F′'s PiRLC parent at logical
width `lw`: what `CompressionKey::relation` (`finish.rs`) passes to
`Relation::new`. -/
structure ParentShape (s : Layer1.Shape) (lw : ℕ) : Prop where
  D : s.D = ringDegree
  laneVars : s.laneVars = 6
  kappa : s.kappa = productionProfile.commitmentWidth
  matrices : s.matrices = productionProfile.ccsMatrices
  publicBlocks : s.publicBlocks = PaperAlgebra.publicRingColumns
  blocks : s.blocks = Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)
  /-- The checked bound `H` is below `B`. -/
  norm : s.normBound < productionGlobalParams.bigB

/-- **The honest setup.** `S` is the layer-1 setup of the F′ circuit with the
logical relation `relation` and the Ajtai key `ajtai`. Rust builds it in
`neo_spartan::Setup::build` and `Key::derive` (`lib.rs`, `key` and `oracle`)
from `Tables::from_rows` (`matrix.rs`) of the circuit's matrix rows. -/
structure HonestSetup (s : Layer1.Shape) (S : Layer1.Setup Gl) (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)) : Prop where
  /-- Key row `i` holds Ajtai row `i`: block `b`, lane `l` at cube index
  `2^laneVars·b + l` (`oracle`). -/
  key : ∀ (i : Fin productionProfile.commitmentWidth)
    (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw))) (l : Fin ringDegree),
    S.key i (2 ^ s.laneVars * b + l) = gl (ajtai i b l)
  /-- Every run names a row of the cube. -/
  row_lt : ∀ k, k < S.runs → S.runRow k < 2 ^ cubeVariables
  /-- The runs give every entry of every completed CCS matrix, row `i` in
  F′'s little-endian order (`Tables::from_rows`). -/
  entry : ∀ (j : Fin productionProfile.ccsMatrices) (i : ℕ) (c : Fin (Phi81CarrierLayout.carrierWidth lw)),
    i < 2 ^ cubeVariables →
      runEntry S s.D j i c = gl ((PaperAlgebra.matrixSource relation.system).matrices j (vertexAt cubeVariables i) c)

/-! ## The runs that Rust pushes -/

/-- The value that the run gives column `c`. -/
def _root_.CompressionLean.Setup.Push.value (p : Push) (c : ℕ) : Gl :=
  if p.start ≤ c ∧ c < p.start + p.len then p.initial * p.ratio ^ (c - p.start) else 0

/-- Entry `(i, c)` of matrix `j` as the pushed runs give it. -/
def pushedEntry (pushes : List Push) (j i c : ℕ) : Gl :=
  (pushes.map fun p => if p.matrix = j ∧ p.row = i then p.value c else 0).sum

/-- **The source premise.** The pushed runs name rows of the cube, and they
give every entry of every completed CCS matrix of `relation`, row `i` in F′'s
little-endian order. -/
structure RunsGive (pushes : List Push) (relation : ProductionKey.LogicalRelation lw pf) : Prop where
  row_lt : ∀ p ∈ pushes, p.row < 2 ^ cubeVariables
  entry : ∀ (j : Fin productionProfile.ccsMatrices) (i : ℕ) (c : Fin (Phi81CarrierLayout.carrierWidth lw)),
    i < 2 ^ cubeVariables →
      pushedEntry pushes j i c = gl ((PaperAlgebra.matrixSource relation.system).matrices j (vertexAt cubeVariables i) c)

end CompressionLean.Spec
