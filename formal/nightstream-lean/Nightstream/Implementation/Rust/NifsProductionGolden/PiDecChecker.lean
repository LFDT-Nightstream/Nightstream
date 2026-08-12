import Nightstream.Implementation.Rust.NifsProductionGolden.Decode

/-! Independent replay of the exact Section-7.5 `Pi_DEC` paper verifier. -/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.NifsProductionGolden.PiDecChecker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Rust.NifsProductionGolden

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
    {Field : Type} {variableCount : Nat} [DecidableEq Field] :
    DecidableEq (CubePoint Field variableCount) :=
  fun left right =>
    if equal : left.coordinates = right.coordinates then
      isTrue (by cases left; cases right; simp_all)
    else
      isFalse fun pointEqual => equal (congrArg CubePoint.coordinates pointEqual)

local instance {Field : Type} {variableCount : Nat} [DecidableEq Field] :
    DecidableEq (CubePoint Field variableCount) :=
  cubePointDecidableEq

abbrev ChildIndex := Fin productionGlobalParams.k

def childrenOfList? (children : List RawClaim) :
    Option (ChildIndex -> RawClaim) :=
  if length : children.length = productionGlobalParams.k then
    some fun child => children.get ⟨child.val, by simpa [length] using child.isLt⟩
  else
    none

structure Decoded where
  parentRaw : RawClaim
  childRaw : ChildIndex -> RawClaim

def parent (decoded : Decoded) : GoldenInstance :=
  decodeClaim .combined decoded.parentRaw

def output (decoded : Decoded) : ChildIndex -> GoldenInstance :=
  fun child => decodeClaim .fresh (decoded.childRaw child)

def decode (receipt : ProductionReceipt) : Option Decoded :=
  if relationShapeCheck receipt && piDecShapeCheck receipt then
    match childrenOfList? receipt.piDecChildren with
    | none => none
    | some children => some {
        parentRaw := receipt.piRlcCombined
        childRaw := children }
  else
    none

def algebra := PiDECAlgebra.Algebra.concrete FixedRelation.zeroKey

def publicSplit :=
  PiDECAlgebra.PaperVerifier.publicInputSplit FixedRelation.zeroKey

def evaluationArity :=
  PiDECAlgebra.PaperVerifier.evaluationArity FixedRelation.zeroKey

def attempt (decoded : Decoded) :=
  PiDEC.PaperVerifier.attemptForOutput (parent decoded) (output decoded)

def EquationsHold (decoded : Decoded) : Prop :=
  (forall child,
      (output decoded child).publicInput =
        publicSplit.split (parent decoded).publicInput child) /\
    (forall child,
      (output decoded child).point = (parent decoded).point) /\
    (parent decoded).commitment =
      algebra.recomposeCommitment (fun child =>
        (output decoded child).commitment) /\
    (parent decoded).evaluations =
      algebra.recomposeEvaluations (fun child =>
        (output decoded child).evaluations)

instance (decoded : Decoded) : Decidable (EquationsHold decoded) := by
  unfold EquationsHold
  infer_instance

def checkReceipt (receipt : ProductionReceipt) : Bool :=
  match decode receipt with
  | none => false
  | some decoded => decide (EquationsHold decoded)

namespace PaperPiDEC

def Accepts (receipt : ProductionReceipt) : Prop :=
  exists decoded : Decoded,
    decode receipt = some decoded /\
      PiDEC.PaperVerifier.OutputAccepted algebra publicSplit evaluationArity
        (parent decoded) (output decoded)

end PaperPiDEC

theorem checkReceipt_sound (receipt : ProductionReceipt) :
    checkReceipt receipt = true -> PaperPiDEC.Accepts receipt := by
  intro checked
  unfold checkReceipt at checked
  cases decodedEq : decode receipt with
  | none => simp [decodedEq] at checked
  | some decoded =>
      have equations : EquationsHold decoded := by
        exact of_decide_eq_true (by simpa [decodedEq] using checked)
      refine ⟨decoded, decodedEq, ?_⟩
      refine {
        outputComputed := ?_
        checks := {
          parentCombined := rfl
          parentEvaluationSize := by
            change (decodeEvaluations decoded.parentRaw).size = 4
            simp [decodeEvaluations]
          messageEvaluationSize := by
            intro child
            change (decodeEvaluations (decoded.childRaw child)).size = 4
            simp [decodeEvaluations]
          commitmentEquation := equations.2.2.1
          evaluationEquation := equations.2.2.2 } }
      funext child
      unfold PiDEC.PaperVerifier.children
      unfold PiDEC.PaperVerifier.attemptForOutput
      unfold PiDEC.PaperVerifier.messagesOf
      simp only
      rw [← equations.1 child, ← equations.2.1 child]
      simp [parent, output, decodeClaim]

end Nightstream.Implementation.Rust.NifsProductionGolden.PiDecChecker
