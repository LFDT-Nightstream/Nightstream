import Nightstream.Implementation.Rust.NifsProductionGolden.Decode
import Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcReplay
import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Independent replay of the production `Pi_RLC` receipt.

The challenge is recomputed from the recorded post-`Pi_CCS` transcript state.
The checker then applies the exact typed Phi81 paper equations.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcChecker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex

def arity : BatchArity productionGlobalParams :=
  BatchArity.bootstrap productionGlobalParams 1 (by decide) (by decide)

def only : Fin arity.total :=
  ⟨0, by simp [arity, BatchArity.total, BatchArity.bootstrap,
    RunningMode.count]⟩

theorem index_eq_only (index : Fin arity.total) : index = only := by
  apply Fin.ext
  have bound : index.val < 1 := by
    simpa [arity, BatchArity.total, BatchArity.bootstrap,
      RunningMode.count] using index.isLt
  change index.val = 0
  omega

private def finFunctionDecidableEq
    {count : Nat} {Value : Type} [DecidableEq Value] :
    DecidableEq (Fin count -> Value) :=
  fun left right =>
    if equal : forall index, left index = right index then
      isTrue (funext equal)
    else
      isFalse fun functionEqual => equal fun index => congrFun functionEqual index

local instance {count : Nat} {Value : Type} [DecidableEq Value] :
    DecidableEq (Fin count -> Value) :=
  finFunctionDecidableEq

private def cubePointDecidableEq
    {Field : Type} {dimension : Nat} [DecidableEq Field] :
    DecidableEq (CubePoint Field dimension) :=
  fun left right =>
    if equal : left.coordinates = right.coordinates then
      isTrue (by cases left; cases right; simp_all)
    else
      isFalse fun pointEqual => equal (congrArg CubePoint.coordinates pointEqual)

local instance {Field : Type} {dimension : Nat} [DecidableEq Field] :
    DecidableEq (CubePoint Field dimension) :=
  cubePointDecidableEq

/-- Exact bounded 54-of-64 sampler at the sole bootstrap coordinate. -/
def sampleScalar? (receipt : ProductionReceipt) : Option Scalar :=
  match PiRlcReplay.sampleFromReceipt? receipt with
  | none => none
  | some result => some result.1

theorem sampleScalar?_production_exact (receipt : ProductionReceipt)
    (scalar : Scalar) (accepted : sampleScalar? receipt = some scalar) :
    exists finalTranscript,
      PiRlcReplay.referenceSample? (decodeSnapshot receipt.rhoStart) =
        some (scalar, finalTranscript) := by
  unfold sampleScalar? PiRlcReplay.sampleFromReceipt? at accepted
  cases sampledEq : PiRlcReplay.sample? receipt
      (CertifiedDuplex.initial
        (decodeSnapshot receipt.rhoStart)
        receipt.rhoStartPermutationCount) with
  | none => simp [sampledEq] at accepted
  | some sampled =>
    have scalarEq : sampled.1 = scalar := Option.some.inj
      (by simpa [sampledEq] using accepted)
    subst scalar
    have sound := PiRlcReplay.sample?_sound receipt
      (CertifiedDuplex.initial
        (decodeSnapshot receipt.rhoStart)
        receipt.rhoStartPermutationCount)
      sampled.2 sampled.1 (by simpa using sampledEq)
    exact ⟨sampled.2.transcript, by
      simpa [CertifiedDuplex.initial] using sound⟩

structure Decoded where
  inputRaw : RawClaim
  outputRaw : RawClaim
  scalar : Scalar

def attempt (decoded : Decoded) : PiRLC.Attempt
    (Structure FixedRelation.shape)
    (PublicInput FixedRelation.shape)
    (Point FixedRelation.shape)
    Evaluation GoldenCommitment RingF productionGlobalParams arity where
  inputs := fun _ => decodeClaim .fresh decoded.inputRaw
  challenges := fun _ => Phi81StrongSet.embedScalar decoded.scalar
  output := decodeClaim .combined decoded.outputRaw

def decode (receipt : ProductionReceipt) : Option Decoded :=
  if relationShapeCheck receipt && piRlcShapeCheck receipt &&
      poseidonTraceShapeCheck receipt then
    match receipt.piRlcInputs, sampleScalar? receipt with
    | [input], some scalar =>
        some {
          inputRaw := input
          outputRaw := receipt.piRlcCombined
          scalar := scalar }
    | _, _ => none
  else
    none

def algebra := PiRLCAlgebra.Algebra.concrete FixedRelation.zeroKey

def EquationsHold (decoded : Decoded) : Prop :=
  ((attempt decoded).inputs only).point = (attempt decoded).output.point /\
    (attempt decoded).output.commitment =
      algebra.combineCommitment (attempt decoded).challenges
        (fun index => ((attempt decoded).inputs index).commitment) /\
    (attempt decoded).output.publicInput =
      algebra.combinePublicInput (attempt decoded).challenges
        (fun index => ((attempt decoded).inputs index).publicInput) /\
    (attempt decoded).output.evaluations =
      algebra.combineEvaluations (attempt decoded).challenges
        (fun index => ((attempt decoded).inputs index).evaluations)

instance (decoded : Decoded) : Decidable (EquationsHold decoded) :=
  by
    unfold EquationsHold
    infer_instance

def checkReceipt (receipt : ProductionReceipt) : Bool :=
  match decode receipt with
  | none => false
  | some decoded => decide (EquationsHold decoded)

namespace PaperPiRLC

def Accepts (receipt : ProductionReceipt) : Prop :=
  exists decoded : Decoded,
    decode receipt = some decoded /\
      PiRLC.Accepted algebra (attempt decoded)

end PaperPiRLC

theorem checkReceipt_sound (receipt : ProductionReceipt) :
    checkReceipt receipt = true -> PaperPiRLC.Accepts receipt := by
  intro checked
  unfold checkReceipt at checked
  cases decodedEq : decode receipt with
  | none => simp [decodedEq] at checked
  | some decoded =>
      have equations : EquationsHold decoded := by
        exact of_decide_eq_true (by simpa [decodedEq] using checked)
      refine ⟨decoded, decodedEq, ?_⟩
      exact {
        inputFresh := by
          intro index
          rw [index_eq_only index]
          rfl
        sameStructure := by
          intro index
          rw [index_eq_only index]
          rfl
        samePoint := by
          intro index
          rw [index_eq_only index]
          exact equations.1
        outputCombined := rfl
        commitmentEquation := equations.2.1
        publicInputEquation := equations.2.2.1
        evaluationEquation := equations.2.2.2
        challengesValid := by
          intro index
          rw [index_eq_only index]
          exact PiRLCAlgebra.Challenge.embedScalar_valid decoded.scalar }

end Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcChecker
