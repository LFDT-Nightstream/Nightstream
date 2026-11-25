import tests.AxiomAudit
import NightstreamFPrime.Export.Generic.ConstantBlocks
import NightstreamFPrime.Export.Generic.Packable

#audit_axioms NightstreamFPrime.Export.Generic.CircuitRelation.packable_eq_packableFast
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.replace_eval
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.permutation_get
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.restore_of_lt
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.outputs_iff
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.sound
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.complete
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.replace_eval_of_holds
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.holds_of_constraints
#audit_axioms NightstreamFPrime.Export.Generic.ConstantBlocks.constraints_complete

namespace NightstreamFPrime.Tests.ConstantBlocks

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Generic.ConstantBlocks
open NightstreamFPrime.Export.Generic.Pieces

private def testPieces : List Piece :=
  [.permutation 320 (List.replicate 16 (.const 0)),
   .permutation 1416 (List.ofFn fun i : Fin 16 => Expr.var (outputColumn 320 i)),
   .permutation 2512 (Expr.var 1 :: List.replicate 15 (.const 0)),
   .constraint (.var (outputColumn 1416 0)),
   .constraint (.var (outputColumn 2512 0))]

#guard (specialize testPieces).pieces.length == 3

#guard (lookup (specialize testPieces).constants (outputColumn 1416 0)) ==
  some ((permutation (permutation (List.replicate 16 0))).getD 0 0)

#guard (lookup (specialize testPieces).constants (outputColumn 2512 0)).isNone

end NightstreamFPrime.Tests.ConstantBlocks
