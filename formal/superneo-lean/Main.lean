import Init

/-- Import prefixes the theorem-facing proof surface must never depend on. -/
private def forbiddenImportPrefixes : List String :=
  ["import SuperNeo.Golden"]

/-- Files and directories that form the theorem-facing proof surface. -/
private def proofSurfaceRoots : List System.FilePath :=
  [ "SuperNeo/ProofSystem"
  , "SuperNeo/FoldingProtocol"
  , "SuperNeo/SecurityModel"
  , "SuperNeo/EmbeddingTheory"
  , "SuperNeo/Primitives"
  , "SuperNeo/FoldingProtocol.lean"
  , "SuperNeo/SecurityModel.lean"
  , "SuperNeo/EmbeddingTheory.lean"
  , "SuperNeo/Primitives.lean"
  ]

private def leanFilesUnder (root : System.FilePath) : IO (Array System.FilePath) := do
  if (← root.isDir) then
    let entries ← root.walkDir
    pure <| entries.filter (·.extension == some "lean")
  else if (← root.pathExists) then
    pure #[root]
  else
    pure #[]

private def fileViolations (path : System.FilePath) : IO (Array String) := do
  let text ← IO.FS.readFile path
  let mut out := #[]
  let mut lineNo := 1
  for line in text.splitOn "\n" do
    for prefixStr in forbiddenImportPrefixes do
      if line.startsWith prefixStr then
        out := out.push s!"{path}:{lineNo}: {line}"
    lineNo := lineNo + 1
  pure out

/--
Pure-Lean theorem import wall: the proof surface must not import the
golden-value executable lane (or any future archived lane added to
`forbiddenImportPrefixes`). No external tools required.
-/
private def checkProofImportWall : IO Bool := do
  let mut violations := #[]
  for root in proofSurfaceRoots do
    for file in (← leanFilesUnder root) do
      violations := violations ++ (← fileViolations file)
  if violations.isEmpty then
    pure true
  else
    IO.println "proof_import_wall_violations:"
    for v in violations do
      IO.println v
    pure false

def main : IO UInt32 := do
  let okProofImportWall ← checkProofImportWall
  let allOk := okProofImportWall

  IO.println s!"proof_import_wall={okProofImportWall}"
  if allOk then
    IO.println "all_checks=true"
    pure 0
  else
    IO.println "all_checks=false"
    pure 1
