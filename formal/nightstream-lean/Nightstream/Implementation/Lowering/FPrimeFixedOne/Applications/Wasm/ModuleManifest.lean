import Lean.Data.Json
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.Module

/-!
Contract: the compact artifact boundary for a Lean-owned WASM application
module.

Assurance tier: model-level.

Owns: schema version 1 and deterministic JSON for the exact module bytes,
entrypoint bytes, and parser-checkable memory initialization facts.

Does not own: file I/O, Rust parsing, execution traces, F-prime, Spartan,
WHIR, or cryptographic soundness. The JSON contains no claimed result; a
consumer must execute or prove the exact module bytes.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ModuleManifest

open Lean
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

def schemaVersion : Nat := 1

def formatName : String := "nightstream/wasm-application-module"

/-- Proof-free deployment descriptor derived from one certified module. The
module bytes remain the authority; the other fields let a consumer fail
closed if its parser sees different memory or export facts. -/
def toJson (identifier : String) (module : CertifiedModule) : Json :=
  Json.mkObj
    [ ("schema", schemaVersion)
    , ("format", formatName)
    , ("module_id", identifier)
    , ("module_hex", bytesHex module.bytes)
    , ("entrypoint_hex", bytesHex module.module.exportName)
    , ("memory_minimum_pages", module.module.memoryMinimumPages)
    , ("memory_maximum_pages", module.module.memoryMaximumPages)
    , ("data_offset", module.module.dataOffset)
    , ("data_hex", bytesHex module.module.data)
    ]

def render (identifier : String) (module : CertifiedModule) : String :=
  (toJson identifier module).compress ++ "\n"

theorem render_exact (identifier : String) (module : CertifiedModule) :
    render identifier module =
      (toJson identifier module).compress ++ "\n" :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.ModuleManifest
