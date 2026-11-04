import NightstreamFPrime.Export.ParallelChunks

namespace NightstreamFPrime.Tests.ParallelChunks

open NightstreamFPrime.Export

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw (IO.userError message)

private def ordering : IO Unit := do
  let running ← IO.mkRef 0
  let peak ← IO.mkRef 0
  let runs ← IO.mkRef (Array.replicate 40 0)
  let tasks ← ParallelChunks.start 3 40 fun index => do
    let now ← running.modifyGet fun count => (count + 1, count + 1)
    peak.modify (max · now)
    IO.sleep 2
    runs.modify fun counts => counts.modify index (· + 1)
    running.modify (· - 1)
    return index * index
  let mut results := #[]
  for task in tasks do
    match ← IO.wait task with
    | .ok (value, _, _) => results := results.push value
    | .error error => throw error
  check (results == (Array.range 40).map fun index => index * index)
    "chunk results lost their order"
  check ((← runs.get).all (· == 1)) "a chunk did not run exactly once"
  check ((← peak.get) ≤ 3) "more chunks ran at once than threads"
  let empty ← ParallelChunks.start 3 0 fun index => pure index
  check empty.isEmpty "an empty chunk range gave tasks"

private def failure : IO Unit := do
  let running ← IO.mkRef 0
  let lateWrites ← IO.mkRef 0
  let result ← (ParallelChunks.withScratch 4 32
    (fun scratch index => do
      running.modify (· + 1)
      try
        if index == 0 then
          IO.sleep 10
          throw (IO.userError "chunk 0 failed")
        IO.sleep 50
        IO.FS.writeFile (scratch / s!"{index}.txt") "chunk"
      catch error =>
        unless index == 0 do lateWrites.modify (· + 1)
        running.modify (· - 1)
        throw error
      running.modify (· - 1)
      return index)
    fun tasks => do
      for task in tasks do
        match ← IO.wait task with
        | .ok _ => pure ()
        | .error error => throw error).toBaseIO
  match result with
  | .error error => check (toString error == "chunk 0 failed") s!"unexpected error: {error}"
  | .ok _ => throw (IO.userError "the failing chunk was not reported")
  check ((← running.get) == 0) "a chunk was still running after the directory was removed"
  check ((← lateWrites.get) == 0) "a chunk wrote into the removed directory"

#eval ordering
#eval failure

end NightstreamFPrime.Tests.ParallelChunks
