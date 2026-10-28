import CompressionLean
import CompressionLeanTests.Inventory

/-! The `status` executable: prints `STATUS.md`. Run it in `formal/compression-lean`.

The page is computed when this file compiles, where the library's notation is
loaded, so the executable needs no run-time import. -/

open Lean CompressionLeanTests.Inventory

elab "status_page%" : term => do
  return mkStrLit (← statusPage (← readPorts))

def main : IO Unit :=
  IO.print status_page%
