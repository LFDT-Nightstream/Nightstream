import NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.Composition

/-!
Owns the one physical R1CS lowering plan for the exact 17-sampler chain.

The plan consumes the certified logical constraint list from `Composition`.
Its projection theorems form the opaque boundary used by preservation and by
the later PiRLC phase assembler.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def plan (interface : Logical.Interface) (offset : Nat) :
    R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + Logical.logicalPrivateCount

def physicalRows (interface : Logical.Interface) (offset : Nat) :
    List R1CS.Row :=
  (plan interface offset).rows

@[simp] theorem plan_constraints (interface : Logical.Interface)
    (offset : Nat) :
    (plan interface offset).constraints =
      logicalConstraints interface offset := by
  rfl

@[simp] theorem plan_firstFresh (interface : Logical.Interface)
    (offset : Nat) :
    (plan interface offset).firstFresh =
      offset + Logical.logicalPrivateCount := by
  rfl

@[simp] theorem physicalRows_eq (interface : Logical.Interface)
    (offset : Nat) :
    physicalRows interface offset = (plan interface offset).rows := by
  rfl

theorem freshColumnCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    (plan interface offset).freshColumnCount = 743631 := by
  change R1CS.totalFreshCount (logicalConstraints interface offset) = 743631
  exact totalFreshCount_eq interface offset inputs

theorem rowCount_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    (plan interface offset).rowCount = 1008848 := by
  rw [R1CS.LoweringPlan.rowCount_eq, plan_constraints,
    totalRowCount_eq interface offset inputs]

theorem next_eq (interface : Logical.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    (plan interface offset).next = offset + 1007199 := by
  rw [R1CS.LoweringPlan.next_eq, plan_firstFresh,
    freshColumnCount_eq interface offset inputs]
  change offset + 263568 + 743631 = offset + 1007199
  omega

end NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain
