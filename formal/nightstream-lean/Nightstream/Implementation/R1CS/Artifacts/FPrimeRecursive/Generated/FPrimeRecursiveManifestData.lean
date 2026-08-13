import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema

/-! Generated diagnostic direct-CCS bit-carrier data by `gadgets_f_prime_recursive_manifest`; do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

def schemaVersion : Nat := 3
def artifactKind : String := "r1cs/f-prime-recursive-program-manifest"
def profile : String := "diagnostic/direct-ccs-bit-carrier/plain/stateless/steady-recursive"
def piCcsSourceCount : Nat := 15
def piCcsMatrixCount : Nat := 4
def piCcsOutputFieldCount : Nat := 6683
def totalRows : Nat := 7169252
def totalColumns : Nat := 7100181
def nifsRowStart : Nat := 21855
def nifsRowEnd : Nat := 4687398
def nifsRowCount : Nat := 4665543
def totalNonzeroEntries : Nat := 32977959
def totalSha256 : String := "6a1af037066cc4203ce799aa83cda5f480b83590fd1c412a73cbe0746e16f838"

def topLevelFamilies : List RowRange :=
  [ { name := "fprime.recursive.prelude", rowStart := 0, rowEnd := 6782, nonzeroEntries := 50822, sha256 := "7c306594c1ea723f5239b6983ecff0af7c3851337a94328e9f5529ba8023bb10" }
  , { name := "fprime.recursive.transcript", rowStart := 6782, rowEnd := 21855, nonzeroEntries := 120169, sha256 := "785d6568371caccff51b4cebc5bf83b5ab60bc787d470f2443be4e1ed8208bd3" }
  , { name := "fprime.recursive.nifs", rowStart := 21855, rowEnd := 4687398, nonzeroEntries := 21663259, sha256 := "5864404e404795d5e2e6bf54aa73b3e03ac64c1539a2b59f36751afe0f7a93f0" }
  , { name := "fprime.recursive.prior_link", rowStart := 4687398, rowEnd := 4692643, nonzeroEntries := 38016, sha256 := "0af859541dce95c8fd96a43d8271a593b7036a36f2accc0891be2dba2cbe485c" }
  , { name := "fprime.recursive.nebula", rowStart := 4692643, rowEnd := 4692643, nonzeroEntries := 0, sha256 := "84a39c68a24131c8c8d551f5b97e172f633bf78f3e4d0d703615fc7032b314c0" }
  , { name := "fprime.recursive.accumulator", rowStart := 4692643, rowEnd := 7163962, nonzeroEntries := 11067430, sha256 := "adc4f1b51c27039a481678aca842a88ac2760b6c66817f3dee8fad404dd86a5c" }
  , { name := "fprime.recursive.counter", rowStart := 7163962, rowEnd := 7164484, nonzeroEntries := 2136, sha256 := "c5b553efa7de792f8534ac3bf370cee1ce2e218b32ab4cea224cd17dca270162" }
  , { name := "fprime.recursive.output", rowStart := 7164484, rowEnd := 7169252, nonzeroEntries := 36127, sha256 := "5b410cb84a4d6f1178459b5f74a98e0384ade8a502c5726247bf4aee744d3db8" }
  ]

def nifsFamilies : List RowRange :=
  [ { name := "nifs.pi_ccs", rowStart := 21855, rowEnd := 3881029, nonzeroEntries := 17459511, sha256 := "a1af4a85f79fc66106e8922e8b8035cbf0dcda80629e25599f0fbbc06dc00b7d" }
  , { name := "nifs.running_parent_pi_dec", rowStart := 3881029, rowEnd := 3888721, nonzeroEntries := 43908, sha256 := "9d5831dce1b496787ff26bd1e6c435babe9f4bebf23fbb8987f835aa86b61279" }
  , { name := "nifs.pi_rlc", rowStart := 3888721, rowEnd := 4679688, nonzeroEntries := 4115878, sha256 := "72bcc518d21462992c2072ac312acac022c7efe440faa016ced2c23b95588f56" }
  , { name := "nifs.pi_dec", rowStart := 4679688, rowEnd := 4687380, nonzeroEntries := 43908, sha256 := "b542d30f72b2ff6d07fb476773995a1b0e1a6ee82dbacec3096afa89b6df203b" }
  , { name := "nifs.point_binding", rowStart := 4687380, rowEnd := 4687398, nonzeroEntries := 54, sha256 := "606d33d5b363cd2b6fd9f355e6d8a70707509e513133e500e00be02b88d7e78c" }
  ]

def projectionShared : RowRange := { name := "nifs.pi_rlc.projection_shared", rowStart := 4617120, rowEnd := 4619012, nonzeroEntries := 7520, sha256 := "4ae965f129c68dcb2487b343813d1a3a319b6438fb479bb0624746c0e32ed0ba" }
def projectionIdentityCount : Nat := 31
def projectionIdentityRows : Nat := 59396
def projectionPairCounts : List Nat := [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
def projectionIdentityRanges : List RowRange :=
  [ { name := "nifs.pi_rlc.projection_identity", rowStart := 4619012, rowEnd := 4620928, nonzeroEntries := 7647, sha256 := "83bcec937dace60def19f016d4920f440687319974fa5a3b2ba236b55f87f352" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4620928, rowEnd := 4622844, nonzeroEntries := 7647, sha256 := "d8a992414defeab93d2b20f5fd36d33d37288544092e258290a92b2e4fdd2678" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4622844, rowEnd := 4624760, nonzeroEntries := 7647, sha256 := "66558d9e46b4f2fb72503a62dd2d2f8b545d13009d6274dc411ccda34deae48e" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4624760, rowEnd := 4626676, nonzeroEntries := 7647, sha256 := "eaa677e8add622e6f1b05bd5e0cc15e936db7ab44801cfdc02c0ea005c6fc83a" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4626676, rowEnd := 4628592, nonzeroEntries := 7647, sha256 := "06279d912e94bc3c8acf58a8ed786a74c3a314a69c700a1accefb12958f884cb" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4628592, rowEnd := 4630508, nonzeroEntries := 7647, sha256 := "756a32a41e18932ee8260af56c32a8e9e6dd6281f34deb2e3a770896b89b0e93" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4630508, rowEnd := 4632424, nonzeroEntries := 7647, sha256 := "a3be4e11c98759b02dc781967fbebc3e7f7388acecd63c9b098c80fd6da55778" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4632424, rowEnd := 4634340, nonzeroEntries := 7647, sha256 := "d0a2df22fa83911b206d3cc58b7f34349f26c84bd90646078091aa586d6d5cf6" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4634340, rowEnd := 4636256, nonzeroEntries := 7647, sha256 := "82f0ca11755df95a3dcb9f9c9f27989c32ab98210505826fb0767eeabfa688cf" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4636256, rowEnd := 4638172, nonzeroEntries := 7647, sha256 := "7477f7b043bcd1c1006d813fb53f423d68020e76fa6ec2ea5486f10378fd696f" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4638172, rowEnd := 4640088, nonzeroEntries := 7647, sha256 := "ad60d3170aecb55fae279a118b32106e818ba0a01007d900d85cb75e0a4338f7" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4640088, rowEnd := 4642004, nonzeroEntries := 7647, sha256 := "2c881ad07468e5a39c5aa19a18a08dce97fd37e386d55c197155248f429a072e" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4642004, rowEnd := 4643920, nonzeroEntries := 7647, sha256 := "a2133779c0cd10a8ff109d98f6aebdee417479337bfafdb14e4eabf07a8aea9c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4643920, rowEnd := 4645836, nonzeroEntries := 7647, sha256 := "8939ee996b71722b12bf806db32e9e901a40e52bc49cb9d66002f8fc05baecc9" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4645836, rowEnd := 4647752, nonzeroEntries := 7647, sha256 := "29d65b912e1ffc55a95e09518c561c4e9ae18d93ac15d9b0ee612c01e4ac7208" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4647752, rowEnd := 4649668, nonzeroEntries := 7647, sha256 := "0138896ba6c9c12a16d296b1322fa25a3670e0486d258f3e038f7efa84b6f2d1" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4649668, rowEnd := 4651584, nonzeroEntries := 7647, sha256 := "c92892e11f4057e25b50e9e062930fe9ded511c3c927014a907dbbeac3019ac5" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4651584, rowEnd := 4653500, nonzeroEntries := 7647, sha256 := "bea98843e0fda758bddbc4a4d18ca6ed5460fcdab8de3481d5d9fbd87d2d44c2" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4653500, rowEnd := 4655416, nonzeroEntries := 7647, sha256 := "82dc6d67cee76db40014cd52c7bad480e872456a5bd1196c74310ebf79b11761" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4655416, rowEnd := 4657332, nonzeroEntries := 7647, sha256 := "71ac808b34ea836be34d29a5b2ac6e3b06f05611f294ce499d7d1952469b0c5d" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4657332, rowEnd := 4659248, nonzeroEntries := 7647, sha256 := "d05216d4dd137668201aed86587f62d794e1ce71c03ac8180f7564812145db0c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4659248, rowEnd := 4661164, nonzeroEntries := 7647, sha256 := "84b45b0e86aa6b640c151287be5b3de038006a8dbd09837c94562f06ea9dee39" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4661164, rowEnd := 4663080, nonzeroEntries := 7647, sha256 := "9d22941bea83ae60209427a995cc4263d67fa8967c60c424687cc1b7035cd562" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4663080, rowEnd := 4664996, nonzeroEntries := 7647, sha256 := "44c36606e20ff1538264709d9dacd479a01eee80d8826ec37ef739e2b9f2fd8f" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4664996, rowEnd := 4666912, nonzeroEntries := 7647, sha256 := "2d513c8fc9ce87783149b331c6c09cf9b878b501d31813650d060e35c6fb5a05" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4667232, rowEnd := 4669148, nonzeroEntries := 7647, sha256 := "824e0023804b8930d89e01ca3e91bdb64f6276798d893590138c1dc2c20e675c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4669148, rowEnd := 4671064, nonzeroEntries := 7647, sha256 := "7335f558267f304b4c5c402e533594850c95747c1f2a4901bb2938d3a6b70c1e" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4671384, rowEnd := 4673300, nonzeroEntries := 7647, sha256 := "53027e385d8e71c257c48fddf98426df40d3126ea249c244f1bedafab582268b" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4673300, rowEnd := 4675216, nonzeroEntries := 7647, sha256 := "7292c62c958848a711750b7fac7a1cfe4353f8293fa388658088d8e73019ed08" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4675536, rowEnd := 4677452, nonzeroEntries := 7647, sha256 := "b6c0e5334be894846b8e6581f8d9ee314d5bfe6f7442aa98e31ade425ee9bc60" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4677452, rowEnd := 4679368, nonzeroEntries := 7647, sha256 := "dbd5ac19dbd826630281f4e331a377beb25520c6989097160a506d5b13df8b4a" }
  ]

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
