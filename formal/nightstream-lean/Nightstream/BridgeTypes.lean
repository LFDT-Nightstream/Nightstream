import SuperNeo.FoldingProtocol.PiCCSInterface
import SuperNeo.FoldingProtocol.PiRLCInterface
import SuperNeo.FoldingProtocol.PiDECInterface
import TwistShout.ShoutCoreInterface
import TwistShout.TwistValueEvalInterface

namespace Nightstream

abbrev SuperNeoPiCCSStrongStatement := SuperNeo.PiCCSInterface.piCCSStrongStatement
abbrev SuperNeoPiRLCWeakStatement := SuperNeo.PiRLCInterface.piRLCWeakStatement
abbrev SuperNeoPiDECKnowledgeStatement := SuperNeo.PiDECInterface.piDECKnowledgeStatement

abbrev ShoutReadOnlyMemoryRelation := @TwistShout.ShoutCoreInterface.ReadOnlyMemoryRelation
abbrev TwistValEvaluationExpression := @TwistShout.TwistValueEvalInterface.valEvaluationExpression
abbrev ShoutPoint := @TwistShout.ShoutCoreInterface.Point
abbrev TwistPoint := @TwistShout.TwistValueEvalInterface.Point

inductive RelationKind where
  | ce
  | shoutReadEval
  | twistValEval
  | opening
  | final
deriving DecidableEq, Repr

inductive SourceKind where
  | mainLane
  | twistShout
deriving DecidableEq, Repr

structure ShoutReadPoint (K : Type*) [Field K] where
  t : Nat
  cycle : ShoutPoint (K := K) t

structure TwistValPoint (K : Type*) [Field K] where
  d : Nat
  m : Nat
  t : Nat
  address : Fin d → TwistPoint (K := K) m
  cycle : TwistPoint (K := K) t

structure Obligation (Family Point : Type*) where
  family : Family
  relation : RelationKind
  point : Point
  source : SourceKind

inductive FamilyDecision where
  | mergeMain
  | foldSeparate
  | exportFinal
deriving DecidableEq, Repr

def FoldableAt
  {Family Point : Type*}
  (family : Family)
  (relation : RelationKind)
  (point : Point)
  (claims : List (Obligation Family Point)) : Prop :=
  claims ≠ [] ∧
    ∀ claim ∈ claims,
      claim.family = family ∧ claim.relation = relation ∧ claim.point = point

def Homogeneous {Family Point : Type*} (claims : List (Obligation Family Point)) : Prop :=
  ∃ family relation point, FoldableAt family relation point claims

def MainLaneAdmissible
  {Family Point : Type*}
  (mainFamily : Family)
  (mainPoint : Point)
  (claims : List (Obligation Family Point)) : Prop :=
  FoldableAt mainFamily .ce mainPoint claims

def SeparateFoldSupported
  {Family Point : Type*}
  (supports : Family → RelationKind → Point → Prop)
  (claims : List (Obligation Family Point)) : Prop :=
  ∃ family relation point, supports family relation point ∧ FoldableAt family relation point claims

noncomputable def classifyFamily
  {Family Point : Type*}
  (mainFamily : Family)
  (mainPoint : Point)
  (supports : Family → RelationKind → Point → Prop)
  (claims : List (Obligation Family Point)) : FamilyDecision := by
  classical
  exact
    if MainLaneAdmissible mainFamily mainPoint claims then .mergeMain
    else if SeparateFoldSupported supports claims then .foldSeparate
    else .exportFinal

end Nightstream
