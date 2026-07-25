import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

/-!
Focused elaboration boundary for the semantics-preserving numeric-to-typed
Goldilocks row translation.
-/

namespace NightstreamTests.NumericRowBridge

open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

#check residue_mod
#check residue_add
#check residue_mul
#check numericAssignment_canonical
#check terms_eval_eq_residue_lcEval
#check row_columnIds
#check row_holds_iff
#check ownedRowsFrom_length
#check ownedRowsFrom_rows
#check ownedRowsFrom_ids_exact
#check ownedRowsFrom_owned
#check ownedRowsFrom_ids_nodup
#check ownedRowsFrom_satisfies_iff

end NightstreamTests.NumericRowBridge
