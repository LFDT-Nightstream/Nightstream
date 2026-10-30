import CompressionLean.Ports.Setup
import CompressionLean.Terminal.Soundness
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.OutputBinding

namespace CompressionLean.Spec

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Setup

def vertexAt (n i : ℕ) : BooleanVertex n :=
  NumericBooleanDomain.vertex n ⟨i % 2 ^ n, Nat.mod_lt _ (by positivity)⟩

abbrev gl (x : F) : Gl := x

abbrev StateWords := Fin 9 → Spec.F

def priorStatement (env : Circuit.Env) : StateWords := fun k =>
  (Layout.Stage1.PiCCSInputs.priorStateWord
    (Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart + 4 + k.val)).eval
    (Layout.Stage1.Spartan.pullback env)

def TerminalFor (authority : Lifecycle.VerifierContext.Authority) (x : StateWords)
    (env : Circuit.Env) : Prop :=
  (Terminal.circuitPackage ()).RowsHold env ∧
    Export.Stage1.Package.SelectedVerifierContext authority env ∧
    priorStatement env = x ∧ x 0 ≠ 0

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

abbrev runningOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.running lw pf (Layout.Stage1.Spartan.pullback env)

abbrev freshOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.fresh lw pf (Layout.Stage1.Spartan.pullback env)

abbrev proofOf (relation : ProductionKey.LogicalRelation lw pf) (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.proof relation (Layout.Stage1.Spartan.pullback env)

/-- SuperNeo v1.2 Definition 21. -/
def ParentClaim (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧
    Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) (runningOf env) (freshOf env)
      (proofOf relation env) = true ∧
    (ProductionKey.key relation ajtai).parent (runningOf env) (freshOf env) (proofOf relation env) =
      some (Terminal.parent relation env) ∧
    ∃ a, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams (Terminal.parent relation env) a

noncomputable def runEntry (S : Layer1.Setup Gl) (D j i c : ℕ) : Gl :=
  ∑ k ∈ ((Finset.range S.runs).filter (fun k => S.runMatrix k = j ∧
      D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) ≤ c ∧
      c < D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) + S.slotLength (S.runSlot k))).filter
      (fun k => S.runRow k = i),
    S.runCoef k * S.slotRatio (S.runSlot k) ^ (c - (D * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k)))

structure ParentShape (s : Layer1.Shape) (lw : ℕ) : Prop where
  D : s.D = ringDegree
  laneVars : s.laneVars = 6
  kappa : s.kappa = productionProfile.commitmentWidth
  matrices : s.matrices = productionProfile.ccsMatrices
  publicBlocks : s.publicBlocks = PaperAlgebra.publicRingColumns
  blocks : s.blocks = Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)
  norm : s.normBound < productionGlobalParams.bigB

structure HonestSetup (s : Layer1.Shape) (S : Layer1.Setup Gl) (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)) : Prop where
  key : ∀ (i : Fin productionProfile.commitmentWidth)
    (b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw))) (l : Fin ringDegree),
    S.key i (2 ^ s.laneVars * b + l) = gl (ajtai i b l)
  row_lt : ∀ k, k < S.runs → S.runRow k < 2 ^ cubeVariables
  entry : ∀ (j : Fin productionProfile.ccsMatrices) (i : ℕ) (c : Fin (Phi81CarrierLayout.carrierWidth lw)),
    i < 2 ^ cubeVariables →
      runEntry S s.D j i c = gl ((PaperAlgebra.matrixSource relation.system).matrices j (vertexAt cubeVariables i) c)

def _root_.CompressionLean.Setup.Push.value (p : Push) (c : ℕ) : Gl :=
  if p.start ≤ c ∧ c < p.start + p.len then p.initial * p.ratio ^ (c - p.start) else 0

def pushedEntry (pushes : List Push) (j i c : ℕ) : Gl :=
  (pushes.map fun p => if p.matrix = j ∧ p.row = i then p.value c else 0).sum

structure RunsGive (pushes : List Push) (relation : ProductionKey.LogicalRelation lw pf) : Prop where
  row_lt : ∀ p ∈ pushes, p.row < 2 ^ cubeVariables
  entry : ∀ (j : Fin productionProfile.ccsMatrices) (i : ℕ) (c : Fin (Phi81CarrierLayout.carrierWidth lw)),
    i < 2 ^ cubeVariables →
      pushedEntry pushes j i c = gl ((PaperAlgebra.matrixSource relation.system).matrices j (vertexAt cubeVariables i) c)

end CompressionLean.Spec
