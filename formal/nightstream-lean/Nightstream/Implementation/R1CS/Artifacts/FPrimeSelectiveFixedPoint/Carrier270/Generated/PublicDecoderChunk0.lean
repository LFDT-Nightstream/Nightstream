import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Schema

/-! Generated file: fixed-point public-coordinate decoder chunk.

Owns: exact proof-free coordinate owners exported from the prepared selective
layout used by the bounded fixed-point projected emitter.

Does not own: source semantics, private coordinates, relation satisfaction,
commitment alignment, or row removal. Do not hand-edit.

Emits constraints: no.

| Artifact field | Exact source | Meaning |
|---|---|---|
| `totalColumns` | final projected-emitter width | bounded profile only |
| `rawCoordinates` | validated prepared-layout owners | public decoder data |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire

def totalColumns : Nat := 11725506
def rawCoordinates : List RawCoordinate := [
  { schemaVersion := 1, column := 0, source := .constantOne }
, { schemaVersion := 1, column := 1, source := .sourceField 1 }
, { schemaVersion := 1, column := 2, source := .sourceField 2 }
, { schemaVersion := 1, column := 3, source := .sourceField 3 }
, { schemaVersion := 1, column := 4, source := .sourceField 4 }
, { schemaVersion := 1, column := 5, source := .sourceField 5 }
, { schemaVersion := 1, column := 6, source := .sourceField 6 }
, { schemaVersion := 1, column := 7, source := .sourceField 7 }
, { schemaVersion := 1, column := 8, source := .sourceField 8 }
, { schemaVersion := 1, column := 9, source := .sourceField 9 }
, { schemaVersion := 1, column := 10, source := .sourceField 10 }
, { schemaVersion := 1, column := 11, source := .sourceField 11 }
, { schemaVersion := 1, column := 12, source := .sourceField 12 }
, { schemaVersion := 1, column := 13, source := .sourceField 13 }
, { schemaVersion := 1, column := 14, source := .sourceField 14 }
, { schemaVersion := 1, column := 15, source := .sourceField 15 }
, { schemaVersion := 1, column := 16, source := .sourceField 16 }
, { schemaVersion := 1, column := 17, source := .sourceField 17 }
, { schemaVersion := 1, column := 18, source := .sourceField 18 }
, { schemaVersion := 1, column := 19, source := .sourceField 19 }
, { schemaVersion := 1, column := 20, source := .sourceField 20 }
, { schemaVersion := 1, column := 21, source := .sourceField 21 }
, { schemaVersion := 1, column := 22, source := .sourceField 22 }
, { schemaVersion := 1, column := 23, source := .sourceField 23 }
, { schemaVersion := 1, column := 24, source := .sourceField 24 }
, { schemaVersion := 1, column := 25, source := .sourceField 25 }
, { schemaVersion := 1, column := 26, source := .sourceField 26 }
, { schemaVersion := 1, column := 27, source := .sourceField 27 }
, { schemaVersion := 1, column := 28, source := .sourceField 28 }
, { schemaVersion := 1, column := 29, source := .sourceField 29 }
, { schemaVersion := 1, column := 30, source := .sourceField 30 }
, { schemaVersion := 1, column := 31, source := .sourceField 31 }
, { schemaVersion := 1, column := 32, source := .sourceField 32 }
, { schemaVersion := 1, column := 33, source := .sourceField 33 }
, { schemaVersion := 1, column := 34, source := .sourceField 34 }
, { schemaVersion := 1, column := 35, source := .sourceField 35 }
, { schemaVersion := 1, column := 36, source := .sourceField 36 }
, { schemaVersion := 1, column := 37, source := .sourceField 37 }
, { schemaVersion := 1, column := 38, source := .sourceField 38 }
, { schemaVersion := 1, column := 39, source := .sourceField 39 }
, { schemaVersion := 1, column := 40, source := .sourceField 40 }
, { schemaVersion := 1, column := 41, source := .sourceField 41 }
, { schemaVersion := 1, column := 42, source := .sourceField 42 }
, { schemaVersion := 1, column := 43, source := .sourceField 43 }
, { schemaVersion := 1, column := 44, source := .sourceField 44 }
, { schemaVersion := 1, column := 45, source := .sourceField 45 }
, { schemaVersion := 1, column := 46, source := .sourceField 46 }
, { schemaVersion := 1, column := 47, source := .sourceField 47 }
, { schemaVersion := 1, column := 48, source := .sourceField 48 }
, { schemaVersion := 1, column := 49, source := .sourceField 49 }
, { schemaVersion := 1, column := 50, source := .sourceField 50 }
, { schemaVersion := 1, column := 51, source := .sourceField 51 }
, { schemaVersion := 1, column := 52, source := .sourceField 52 }
, { schemaVersion := 1, column := 53, source := .sourceField 53 }
, { schemaVersion := 1, column := 54, source := .sourceField 54 }
, { schemaVersion := 1, column := 55, source := .sourceField 55 }
, { schemaVersion := 1, column := 56, source := .sourceField 56 }
, { schemaVersion := 1, column := 57, source := .sourceField 57 }
, { schemaVersion := 1, column := 58, source := .sourceField 58 }
, { schemaVersion := 1, column := 59, source := .sourceField 59 }
, { schemaVersion := 1, column := 60, source := .sourceField 60 }
, { schemaVersion := 1, column := 61, source := .sourceField 61 }
, { schemaVersion := 1, column := 62, source := .sourceField 62 }
, { schemaVersion := 1, column := 63, source := .sourceField 63 }
, { schemaVersion := 1, column := 64, source := .sourceField 64 }
, { schemaVersion := 1, column := 65, source := .sourceField 65 }
, { schemaVersion := 1, column := 66, source := .sourceField 66 }
, { schemaVersion := 1, column := 67, source := .sourceField 67 }
, { schemaVersion := 1, column := 68, source := .sourceField 68 }
, { schemaVersion := 1, column := 69, source := .sourceField 69 }
, { schemaVersion := 1, column := 70, source := .sourceField 70 }
, { schemaVersion := 1, column := 71, source := .sourceField 71 }
, { schemaVersion := 1, column := 72, source := .sourceField 72 }
, { schemaVersion := 1, column := 73, source := .sourceField 73 }
, { schemaVersion := 1, column := 74, source := .sourceField 74 }
, { schemaVersion := 1, column := 75, source := .sourceField 75 }
, { schemaVersion := 1, column := 76, source := .sourceField 76 }
, { schemaVersion := 1, column := 77, source := .sourceField 77 }
, { schemaVersion := 1, column := 78, source := .sourceField 78 }
, { schemaVersion := 1, column := 79, source := .sourceField 79 }
, { schemaVersion := 1, column := 80, source := .sourceField 80 }
, { schemaVersion := 1, column := 81, source := .sourceField 81 }
, { schemaVersion := 1, column := 82, source := .sourceField 82 }
, { schemaVersion := 1, column := 83, source := .sourceField 83 }
, { schemaVersion := 1, column := 84, source := .sourceField 84 }
, { schemaVersion := 1, column := 85, source := .sourceField 85 }
, { schemaVersion := 1, column := 86, source := .sourceField 86 }
, { schemaVersion := 1, column := 87, source := .sourceField 87 }
, { schemaVersion := 1, column := 88, source := .sourceField 88 }
, { schemaVersion := 1, column := 89, source := .sourceField 89 }
, { schemaVersion := 1, column := 90, source := .sourceField 90 }
, { schemaVersion := 1, column := 91, source := .sourceField 91 }
, { schemaVersion := 1, column := 92, source := .sourceField 92 }
, { schemaVersion := 1, column := 93, source := .sourceField 93 }
, { schemaVersion := 1, column := 94, source := .sourceField 94 }
, { schemaVersion := 1, column := 95, source := .sourceField 95 }
, { schemaVersion := 1, column := 96, source := .sourceField 96 }
, { schemaVersion := 1, column := 97, source := .sourceField 97 }
, { schemaVersion := 1, column := 98, source := .sourceField 98 }
, { schemaVersion := 1, column := 99, source := .sourceField 99 }
, { schemaVersion := 1, column := 100, source := .sourceField 100 }
, { schemaVersion := 1, column := 101, source := .sourceField 101 }
, { schemaVersion := 1, column := 102, source := .sourceField 102 }
, { schemaVersion := 1, column := 103, source := .sourceField 103 }
, { schemaVersion := 1, column := 104, source := .sourceField 104 }
, { schemaVersion := 1, column := 105, source := .sourceField 105 }
, { schemaVersion := 1, column := 106, source := .sourceField 106 }
, { schemaVersion := 1, column := 107, source := .sourceField 107 }
, { schemaVersion := 1, column := 108, source := .sourceField 108 }
, { schemaVersion := 1, column := 109, source := .sourceField 109 }
, { schemaVersion := 1, column := 110, source := .sourceField 110 }
, { schemaVersion := 1, column := 111, source := .sourceField 111 }
, { schemaVersion := 1, column := 112, source := .sourceField 112 }
, { schemaVersion := 1, column := 113, source := .sourceField 113 }
, { schemaVersion := 1, column := 114, source := .sourceField 114 }
, { schemaVersion := 1, column := 115, source := .sourceField 115 }
, { schemaVersion := 1, column := 116, source := .sourceField 116 }
, { schemaVersion := 1, column := 117, source := .sourceField 117 }
, { schemaVersion := 1, column := 118, source := .sourceField 118 }
, { schemaVersion := 1, column := 119, source := .sourceField 119 }
, { schemaVersion := 1, column := 120, source := .sourceField 120 }
, { schemaVersion := 1, column := 121, source := .sourceField 121 }
, { schemaVersion := 1, column := 122, source := .sourceField 122 }
, { schemaVersion := 1, column := 123, source := .sourceField 123 }
, { schemaVersion := 1, column := 124, source := .sourceField 124 }
, { schemaVersion := 1, column := 125, source := .sourceField 125 }
, { schemaVersion := 1, column := 126, source := .sourceField 126 }
, { schemaVersion := 1, column := 127, source := .sourceField 127 }
, { schemaVersion := 1, column := 128, source := .sourceField 128 }
, { schemaVersion := 1, column := 129, source := .sourceField 129 }
, { schemaVersion := 1, column := 130, source := .sourceField 130 }
, { schemaVersion := 1, column := 131, source := .sourceField 131 }
, { schemaVersion := 1, column := 132, source := .sourceField 132 }
, { schemaVersion := 1, column := 133, source := .sourceField 133 }
, { schemaVersion := 1, column := 134, source := .sourceField 134 }
, { schemaVersion := 1, column := 135, source := .sourceField 135 }
, { schemaVersion := 1, column := 136, source := .sourceField 136 }
, { schemaVersion := 1, column := 137, source := .sourceField 137 }
, { schemaVersion := 1, column := 138, source := .sourceField 138 }
, { schemaVersion := 1, column := 139, source := .sourceField 139 }
, { schemaVersion := 1, column := 140, source := .sourceField 140 }
, { schemaVersion := 1, column := 141, source := .sourceField 141 }
, { schemaVersion := 1, column := 142, source := .sourceField 142 }
, { schemaVersion := 1, column := 143, source := .sourceField 143 }
, { schemaVersion := 1, column := 144, source := .sourceField 144 }
, { schemaVersion := 1, column := 145, source := .sourceField 145 }
, { schemaVersion := 1, column := 146, source := .sourceField 146 }
, { schemaVersion := 1, column := 147, source := .sourceField 147 }
, { schemaVersion := 1, column := 148, source := .sourceField 148 }
, { schemaVersion := 1, column := 149, source := .sourceField 149 }
, { schemaVersion := 1, column := 150, source := .sourceField 150 }
, { schemaVersion := 1, column := 151, source := .sourceField 151 }
, { schemaVersion := 1, column := 152, source := .sourceField 152 }
, { schemaVersion := 1, column := 153, source := .sourceField 153 }
, { schemaVersion := 1, column := 154, source := .sourceField 154 }
, { schemaVersion := 1, column := 155, source := .sourceField 155 }
, { schemaVersion := 1, column := 156, source := .sourceField 156 }
, { schemaVersion := 1, column := 157, source := .sourceField 157 }
, { schemaVersion := 1, column := 158, source := .sourceField 158 }
, { schemaVersion := 1, column := 159, source := .sourceField 159 }
, { schemaVersion := 1, column := 160, source := .sourceField 160 }
, { schemaVersion := 1, column := 161, source := .sourceField 161 }
, { schemaVersion := 1, column := 162, source := .sourceField 162 }
, { schemaVersion := 1, column := 163, source := .sourceField 163 }
, { schemaVersion := 1, column := 164, source := .sourceField 164 }
, { schemaVersion := 1, column := 165, source := .sourceField 165 }
, { schemaVersion := 1, column := 166, source := .sourceField 166 }
, { schemaVersion := 1, column := 167, source := .sourceField 167 }
, { schemaVersion := 1, column := 168, source := .sourceField 168 }
, { schemaVersion := 1, column := 169, source := .sourceField 169 }
, { schemaVersion := 1, column := 170, source := .sourceField 170 }
, { schemaVersion := 1, column := 171, source := .sourceField 171 }
, { schemaVersion := 1, column := 172, source := .sourceField 172 }
, { schemaVersion := 1, column := 173, source := .sourceField 173 }
, { schemaVersion := 1, column := 174, source := .sourceField 174 }
, { schemaVersion := 1, column := 175, source := .sourceField 175 }
, { schemaVersion := 1, column := 176, source := .sourceField 176 }
, { schemaVersion := 1, column := 177, source := .sourceField 177 }
, { schemaVersion := 1, column := 178, source := .sourceField 178 }
, { schemaVersion := 1, column := 179, source := .sourceField 179 }
, { schemaVersion := 1, column := 180, source := .sourceField 180 }
, { schemaVersion := 1, column := 181, source := .sourceField 181 }
, { schemaVersion := 1, column := 182, source := .sourceField 182 }
, { schemaVersion := 1, column := 183, source := .sourceField 183 }
, { schemaVersion := 1, column := 184, source := .sourceField 184 }
, { schemaVersion := 1, column := 185, source := .sourceField 185 }
, { schemaVersion := 1, column := 186, source := .sourceField 186 }
, { schemaVersion := 1, column := 187, source := .sourceField 187 }
, { schemaVersion := 1, column := 188, source := .sourceField 188 }
, { schemaVersion := 1, column := 189, source := .sourceField 189 }
, { schemaVersion := 1, column := 190, source := .sourceField 190 }
, { schemaVersion := 1, column := 191, source := .sourceField 191 }
, { schemaVersion := 1, column := 192, source := .sourceField 192 }
, { schemaVersion := 1, column := 193, source := .sourceField 193 }
, { schemaVersion := 1, column := 194, source := .sourceField 194 }
, { schemaVersion := 1, column := 195, source := .sourceField 195 }
, { schemaVersion := 1, column := 196, source := .sourceField 196 }
, { schemaVersion := 1, column := 197, source := .sourceField 197 }
, { schemaVersion := 1, column := 198, source := .sourceField 198 }
, { schemaVersion := 1, column := 199, source := .sourceField 199 }
, { schemaVersion := 1, column := 200, source := .sourceField 200 }
, { schemaVersion := 1, column := 201, source := .sourceField 201 }
, { schemaVersion := 1, column := 202, source := .sourceField 202 }
, { schemaVersion := 1, column := 203, source := .sourceField 203 }
, { schemaVersion := 1, column := 204, source := .sourceField 204 }
, { schemaVersion := 1, column := 205, source := .sourceField 205 }
, { schemaVersion := 1, column := 206, source := .sourceField 206 }
, { schemaVersion := 1, column := 207, source := .sourceField 207 }
, { schemaVersion := 1, column := 208, source := .sourceField 208 }
, { schemaVersion := 1, column := 209, source := .sourceField 209 }
, { schemaVersion := 1, column := 210, source := .sourceField 210 }
, { schemaVersion := 1, column := 211, source := .sourceField 211 }
, { schemaVersion := 1, column := 212, source := .sourceField 212 }
, { schemaVersion := 1, column := 213, source := .sourceField 213 }
, { schemaVersion := 1, column := 214, source := .sourceField 214 }
, { schemaVersion := 1, column := 215, source := .sourceField 215 }
, { schemaVersion := 1, column := 216, source := .sourceField 216 }
, { schemaVersion := 1, column := 217, source := .sourceField 217 }
, { schemaVersion := 1, column := 218, source := .sourceField 218 }
, { schemaVersion := 1, column := 219, source := .sourceField 219 }
, { schemaVersion := 1, column := 220, source := .sourceField 220 }
, { schemaVersion := 1, column := 221, source := .sourceField 221 }
, { schemaVersion := 1, column := 222, source := .sourceField 222 }
, { schemaVersion := 1, column := 223, source := .sourceField 223 }
, { schemaVersion := 1, column := 224, source := .sourceField 224 }
, { schemaVersion := 1, column := 225, source := .sourceField 225 }
, { schemaVersion := 1, column := 226, source := .sourceField 226 }
, { schemaVersion := 1, column := 227, source := .sourceField 227 }
, { schemaVersion := 1, column := 228, source := .sourceField 228 }
, { schemaVersion := 1, column := 229, source := .sourceField 229 }
, { schemaVersion := 1, column := 230, source := .sourceField 230 }
, { schemaVersion := 1, column := 231, source := .sourceField 231 }
, { schemaVersion := 1, column := 232, source := .sourceField 232 }
, { schemaVersion := 1, column := 233, source := .sourceField 233 }
, { schemaVersion := 1, column := 234, source := .sourceField 234 }
, { schemaVersion := 1, column := 235, source := .sourceField 235 }
, { schemaVersion := 1, column := 236, source := .sourceField 236 }
, { schemaVersion := 1, column := 237, source := .sourceField 237 }
, { schemaVersion := 1, column := 238, source := .sourceField 238 }
, { schemaVersion := 1, column := 239, source := .sourceField 239 }
, { schemaVersion := 1, column := 240, source := .sourceField 240 }
, { schemaVersion := 1, column := 241, source := .sourceField 241 }
, { schemaVersion := 1, column := 242, source := .sourceField 242 }
, { schemaVersion := 1, column := 243, source := .sourceField 243 }
, { schemaVersion := 1, column := 244, source := .sourceField 244 }
, { schemaVersion := 1, column := 245, source := .sourceField 245 }
, { schemaVersion := 1, column := 246, source := .sourceField 246 }
, { schemaVersion := 1, column := 247, source := .sourceField 247 }
, { schemaVersion := 1, column := 248, source := .sourceField 248 }
, { schemaVersion := 1, column := 249, source := .sourceField 249 }
, { schemaVersion := 1, column := 250, source := .sourceField 250 }
, { schemaVersion := 1, column := 251, source := .sourceField 251 }
, { schemaVersion := 1, column := 252, source := .sourceField 252 }
, { schemaVersion := 1, column := 253, source := .sourceField 253 }
, { schemaVersion := 1, column := 254, source := .sourceField 254 }
, { schemaVersion := 1, column := 255, source := .sourceField 255 }
]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk0
