import Nightstream.Implementation.NebulaV2.WasmStatementBytes

set_option autoImplicit false

namespace tests.NebulaV2WasmStatementBytes

open Nightstream.Implementation.NebulaV2.WasmStatementBytes
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

example (word : BitVec paddedBitCount) :
    joinBytes (splitBytes word) = word :=
  join_split word

example (image : PublicImage) :
    (encode image).length = 984 := by
  simpa [statementByteCount] using encode_length image

example (image : PublicImage) :
    (byteWords image ⟨983, by decide⟩).toNat < 16 := by
  simpa [statementByteCount] using final_byte_lt_16 image

end tests.NebulaV2WasmStatementBytes
