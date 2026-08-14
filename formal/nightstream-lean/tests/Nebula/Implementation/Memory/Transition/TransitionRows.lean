import Nightstream.Implementation.Nebula.Memory.Transition.TransitionSound

/-! Focused schema and semantic gates for the exact local transition. -/

set_option autoImplicit false

namespace tests.NebulaMemoryTransitionRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.SuperNeo.Concrete

theorem exact_row_count (layout : MemoryTransitionRows.Layout) :
    (MemoryTransitionRows.rows layout).length = 225 :=
  MemoryTransitionRows.rows_length_exact layout

theorem satisfying_rows_select_one_semantic_transition
    {layout : MemoryTransitionRows.Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {before after : MemoryCarryCodec.Value}
    {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (beforeParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.before
      assignment headers before)
    (claimParsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment
      claim)
    (afterParsed : MemoryCarryPublicRows.ParsedColumnsMatch layout.after
      assignment headers after)
    (holds : Satisfies (MemoryTransitionRows.rows layout) assignment)
    (balancedOnClose : after.phase = .closed →
      MemoryProductBalanceRows.ConcreteBalanced claim.productsAfter) :
    Consumes MemoryProductBalanceRows.ConcreteBalanced
      (MemoryCarryParser.semanticCarry before
        beforeParsed.parserCanonical.stepIndex)
      claim
      (MemoryCarryParser.semanticCarry after
        afterParsed.parserCanonical.stepIndex) :=
  MemoryTransitionSound.consumes_of_rows canonical one beforeParsed
    claimParsed afterParsed holds balancedOnClose

end tests.NebulaMemoryTransitionRows
