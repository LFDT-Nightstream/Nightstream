(* The logical catalog of each binding export, which Lean `checkWasmBinding` checks, is
   `MemoryCatalog.memories`. *)
Require Import MemoryTrace MemoryCatalog tests.BindingCatalogTrace.

Example micro_catalog : microLogical = memories := eq_refl.
Example small_catalog : smallLogical = memories := eq_refl.
Example medium_catalog : mediumLogical = memories := eq_refl.
Example large_catalog : largeLogical = memories := eq_refl.
