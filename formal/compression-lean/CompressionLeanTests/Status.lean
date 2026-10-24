import CompressionLeanTests.Inventory

/-! The `status` executable: prints `STATUS.md`. Run it in `formal/compression-lean`. -/

open Lean CompressionLeanTests.Inventory

unsafe def main : IO Unit := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let env ← importModules #[{ module := `CompressionLean }] {} (loadExts := true)
  let page ← (statusPage (← readPorts)).toIO' { fileName := "status", fileMap := default } { env }
  IO.print page
