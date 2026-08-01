import Nightstream.Protocol.Nebula.Fingerprint

/-!
Contract: Lean-owned sequential memory semantics for one Nebula segment.

Assurance tier: model-level.

Owns an exact access relation, execution over a current cell snapshot, and
the telescoping theorem that every honest execution satisfies the Nebula
initial/write versus read/final product equation.

Does not own ROM/RAM namespace policy, stack rules, transcript challenges,
commitments, CCS rows, F-prime carry, Rust layouts, or collision probability.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Memory

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Protocol.Nebula.Fingerprint

/-- One memory access as the multiset checker sees it. -/
structure Access where
  read : MemTuple
  write : MemTuple
deriving DecidableEq, Repr

/-- One access replaces exactly the tuple that is currently stored at its
global cell index.  The write timestamp is the next global timestamp. -/
structure Applies
    (before : List MemTuple) (timestampIn : Nat)
    (access : Access) (after : List MemTuple) (timestampOut : Nat) where
  left : List MemTuple
  right : List MemTuple
  beforeExact : before = left ++ (access.read :: right)
  afterExact : after = left ++ (access.write :: right)
  sameCell : access.write.globalIndex = access.read.globalIndex
  previousTimestamp : access.read.timestamp < timestampIn + 1
  writeTimestamp : access.write.timestamp = timestampIn + 1
  timestampExact : timestampOut = timestampIn + 1

/-- Exact sequential execution of a list of accesses. -/
inductive Executes :
    List MemTuple → Nat → List Access → List MemTuple → Nat → Prop
  | nil (snapshot : List MemTuple) (timestamp : Nat) :
      Executes snapshot timestamp [] snapshot timestamp
  | cons
      {before middle after : List MemTuple}
      {timestampIn timestampMiddle timestampOut : Nat}
      {access : Access} {rest : List Access}
      (applies : Applies before timestampIn access middle timestampMiddle)
      (tail : Executes middle timestampMiddle rest after timestampOut) :
      Executes before timestampIn (access :: rest) after timestampOut

def readTuples (accesses : List Access) : List MemTuple :=
  accesses.map Access.read

def writeTuples (accesses : List Access) : List MemTuple :=
  accesses.map Access.write

private theorem k_mul_assoc (left middle right : K) :
    K.mul (K.mul left middle) right = K.mul left (K.mul middle right) :=
  extensionLaws.mul_assoc left middle right

private theorem k_mul_comm (left right : K) :
    K.mul left right = K.mul right left :=
  extensionLaws.mul_comm left right

private theorem k_one_mul (value : K) : K.mul K.one value = value :=
  extensionLaws.one_mul value

private theorem k_mul_one (value : K) : K.mul value K.one = value :=
  extensionLaws.mul_one value

local instance : Std.Associative K.mul := ⟨k_mul_assoc⟩
local instance : Std.Commutative K.mul := ⟨k_mul_comm⟩

theorem applies_product
    (challenges : Challenges)
    {before after : List MemTuple}
    {timestampIn timestampOut : Nat}
    {access : Access}
    (applies : Applies before timestampIn access after timestampOut) :
    K.mul (product challenges before)
        (fingerprint challenges access.write) =
      K.mul (fingerprint challenges access.read)
        (product challenges after) := by
  obtain ⟨left, right, beforeExact, afterExact, _, _, _, _⟩ := applies
  subst before
  subst after
  rw [product_append, product_append]
  simp only [product]
  ac_rfl

theorem executes_product
    (challenges : Challenges)
    {initial final : List MemTuple}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (execution : Executes initial timestampIn accesses final timestampOut) :
    K.mul (product challenges initial)
        (product challenges (writeTuples accesses)) =
      K.mul (product challenges (readTuples accesses))
        (product challenges final) := by
  induction execution with
  | nil snapshot timestamp =>
      simp only [writeTuples, readTuples, List.map, product]
      rw [k_mul_one, k_one_mul]
  | cons applies tail inductionHypothesis =>
      simp only [writeTuples, readTuples, List.map, product] at inductionHypothesis ⊢
      rw [← k_mul_assoc]
      rw [applies_product challenges applies]
      rw [k_mul_assoc]
      rw [inductionHypothesis]
      rw [← k_mul_assoc]

/-- Product order used by the selected Nebula lane:
`[read, write, initial, final]`. -/
def products
    (challenges : Challenges)
    (initial : List MemTuple) (accesses : List Access)
    (final : List MemTuple) : Fin 4 → K
  | ⟨0, _⟩ => product challenges (readTuples accesses)
  | ⟨1, _⟩ => product challenges (writeTuples accesses)
  | ⟨2, _⟩ => product challenges initial
  | _ => product challenges final

/-- Nebula's segment-close product equation. -/
def Balanced (values : Fin 4 → K) : Prop :=
  K.mul (values ⟨2, by decide⟩) (values ⟨1, by decide⟩) =
    K.mul (values ⟨0, by decide⟩) (values ⟨3, by decide⟩)

theorem executes_balanced
    (challenges : Challenges)
    {initial final : List MemTuple}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (execution : Executes initial timestampIn accesses final timestampOut) :
    Balanced (products challenges initial accesses final) := by
  exact executes_product challenges execution

end Nightstream.Protocol.Nebula.Memory
