/-!
Run independent chunks on a fixed number of dedicated threads. Each thread takes
the next unstarted chunk, so the thread count is the configured worker count and
the chunk count only sets the size of one work item. Results keep chunk order.
After a chunk fails, the chunks not yet started are skipped with an error. This
is scheduling only; callers combine the results in their canonical order.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.ParallelChunks

/-- The worker count from `LEAN_NUM_THREADS`, at least one. -/
def workers : IO Nat := do
  return max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)

/-- Start chunks `0` to `count - 1` on at most `threads` dedicated threads. The task
of each chunk gives its result with its start and finish times, or its error. -/
def start {Alpha : Type} (threads count : Nat) (run : Nat → IO Alpha) :
    IO (Array (Task (Except IO.Error (Alpha × Nat × Nat)))) := do
  let promises : Array (IO.Promise (Except IO.Error (Alpha × Nat × Nat))) ←
    (Array.range count).mapM fun _ => IO.Promise.new
  let next ← IO.mkRef 0
  let failed ← IO.mkRef false
  for _ in [:min threads count] do
    let _ ← IO.asTask (prio := Task.Priority.dedicated) do
      repeat
        let index ← next.modifyGet fun index => (index, index + 1)
        if within : index < promises.size then
          if ← failed.get then
            promises[index].resolve
              (.error (IO.userError s!"chunk {index} skipped after an earlier chunk failed"))
          else
            let started ← IO.monoNanosNow
            let result ← (run index).toBaseIO
            let finished ← IO.monoNanosNow
            if result matches .error _ then failed.set true
            promises[index].resolve (result.map fun value => (value, started, finished))
        else break
  return promises.map (·.result!)

/-- Wait until every chunk has finished or been skipped. -/
def waitAll {Beta : Type} (tasks : Array (Task (Except IO.Error Beta))) : BaseIO Unit := do
  for task in tasks do
    let _ ← IO.wait task

/-- Run chunks that write into a new temporary directory, and consume their results.
Every chunk finishes or is skipped before the directory is removed, also when
`consume` fails; the original error is kept. -/
def withScratch {Alpha Beta : Type} (threads count : Nat)
    (run : System.FilePath → Nat → IO Alpha)
    (consume : Array (Task (Except IO.Error (Alpha × Nat × Nat))) → IO Beta) : IO Beta :=
  IO.FS.withTempDir fun scratch => do
    let tasks ← start threads count (run scratch)
    try consume tasks finally waitAll tasks

end NightstreamFPrime.Export.ParallelChunks
