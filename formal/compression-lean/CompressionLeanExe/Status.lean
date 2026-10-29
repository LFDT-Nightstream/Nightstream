import CompressionLean
import CompressionLeanTests.Inventory

open Lean CompressionLeanTests.Inventory

elab "status_page%" : term => do
  return mkStrLit (← statusPage (← readPorts))

def main : IO Unit :=
  IO.print status_page%
