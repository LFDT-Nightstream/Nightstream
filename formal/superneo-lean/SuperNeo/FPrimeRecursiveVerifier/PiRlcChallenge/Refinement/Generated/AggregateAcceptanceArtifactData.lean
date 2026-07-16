import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.AggregateAcceptanceArtifactSchema

/-! Generated exact active aggregate-acceptance data; do not hand-edit.

Owns: the active singleton-input leaf's source equations, canonical inverse
decoder, global geometry, emitted rows, and sparse polynomial specialization.

Does not own: the fixed-F' `ChunkBitOuterImage` or generated 960-role census.
Those remain required before this leaf can authorize production decoded-LC inputs.

Emits constraints: no.

Authority boundary: every role, range, equation, and coefficient is listed
directly. No digest authorizes any artifact branch.

| Data branch | Mathematical obligation | Production check |
|---|---|---|
| `sourceRows` | four canonical source equations | all 64 traced chunks agree |
| `inverseDecoder` | exact projected inverse and rows 2/3 ownership | production lowering plan |
| `chunkGeometry` | exact global row/column role census | all singleton active leaves |
| `activeRows` | seven bit pairs, radix-3 ProductSum, root binding | materialized CCS matrices |
| `polynomialTerms` | exact arity-48 specialization | production sparse polynomial |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifactData

open AggregateAcceptanceArtifact

def schemaVersion : Nat := 1
def sourceInputOrder : List SourceRole :=
  [(.chunkBit 0), (.chunkBit 1), (.chunkBit 2), (.chunkBit 3), (.chunkBit 4), (.chunkBit 5), (.chunkBit 6), (.chunkBit 7), (.chunkBit 8), (.chunkBit 9), (.chunkBit 10), (.chunkBit 11), (.chunkBit 12), (.chunkBit 13), (.chunkBit 14), (.chunkBit 15)]
def sourceAllocatedOrder : List SourceRole :=
  [.accept, .inverse]
def coordinateOrder : List CoordinateRole :=
  [.accept, (.treeOutput 0), (.treeOutput 1), (.treeOutput 2), (.treeOutput 3), (.treeOutput 4), (.treeOutput 5), (.treeOutput 6), (.treeOutput 7), (.treeOutput 8), (.treeOutput 9), (.treeOutput 10), (.treeOutput 11), (.treeOutput 12), (.treeOutput 13)]
def sourceRows : List SourceRow :=
[
  ⟨[⟨.accept, 1⟩],
    [⟨.one, -1⟩, ⟨.accept, 1⟩],
    []⟩
, ⟨[⟨.one, 1⟩, ⟨.accept, -1⟩],
    [⟨.one, -65535⟩, ⟨(.chunkBit 0), 1⟩, ⟨(.chunkBit 1), 2⟩, ⟨(.chunkBit 2), 4⟩, ⟨(.chunkBit 3), 8⟩, ⟨(.chunkBit 4), 16⟩, ⟨(.chunkBit 5), 32⟩, ⟨(.chunkBit 6), 64⟩, ⟨(.chunkBit 7), 128⟩, ⟨(.chunkBit 8), 256⟩, ⟨(.chunkBit 9), 512⟩, ⟨(.chunkBit 10), 1024⟩, ⟨(.chunkBit 11), 2048⟩, ⟨(.chunkBit 12), 4096⟩, ⟨(.chunkBit 13), 8192⟩, ⟨(.chunkBit 14), 16384⟩, ⟨(.chunkBit 15), 32768⟩],
    []⟩
, ⟨[⟨.one, -65535⟩, ⟨(.chunkBit 0), 1⟩, ⟨(.chunkBit 1), 2⟩, ⟨(.chunkBit 2), 4⟩, ⟨(.chunkBit 3), 8⟩, ⟨(.chunkBit 4), 16⟩, ⟨(.chunkBit 5), 32⟩, ⟨(.chunkBit 6), 64⟩, ⟨(.chunkBit 7), 128⟩, ⟨(.chunkBit 8), 256⟩, ⟨(.chunkBit 9), 512⟩, ⟨(.chunkBit 10), 1024⟩, ⟨(.chunkBit 11), 2048⟩, ⟨(.chunkBit 12), 4096⟩, ⟨(.chunkBit 13), 8192⟩, ⟨(.chunkBit 14), 16384⟩, ⟨(.chunkBit 15), 32768⟩],
    [⟨.inverse, 1⟩],
    [⟨.accept, 1⟩]⟩
, ⟨[⟨.one, 1⟩, ⟨.accept, -1⟩],
    [⟨.inverse, 1⟩],
    []⟩
]
def inverseDecoder : CanonicalInverseDecoder :=
  { output := .inverse,
    difference := [⟨(.chunkBit 0), 1⟩, ⟨(.chunkBit 1), 2⟩, ⟨(.chunkBit 2), 4⟩, ⟨(.chunkBit 3), 8⟩, ⟨(.chunkBit 4), 16⟩, ⟨(.chunkBit 5), 32⟩, ⟨(.chunkBit 6), 64⟩, ⟨(.chunkBit 7), 128⟩, ⟨(.chunkBit 8), 256⟩, ⟨(.chunkBit 9), 512⟩, ⟨(.chunkBit 10), 1024⟩, ⟨(.chunkBit 11), 2048⟩, ⟨(.chunkBit 12), 4096⟩, ⟨(.chunkBit 13), 8192⟩, ⟨(.chunkBit 14), 16384⟩, ⟨(.chunkBit 15), 32768⟩, ⟨.one, -65535⟩],
    ownedRowOffsets := [2, 3] }
set_option maxRecDepth 65536 in
def chunkGeometry : List ChunkGeometry :=
[
  { sourceRowStart := 682, sourceRowEnd := 686,
    sourceColumnStart := 680, sourceColumnEnd := 682,
    sourceInputColumns := [614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629], sourceAcceptColumn := 680, sourceInverseColumn := 681,
    encodedInputColumns := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    encodedAcceptanceColumns := [9291, 41810, 41811, 41812, 41813, 41814, 41815, 41816, 41817, 41818, 41819, 41820, 41821, 41822, 41823],
    activeRowStart := 34373, activeRowEnd := 34382 }
, { sourceRowStart := 708, sourceRowEnd := 712,
    sourceColumnStart := 703, sourceColumnEnd := 705,
    sourceInputColumns := [630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645], sourceAcceptColumn := 703, sourceInverseColumn := 704,
    encodedInputColumns := [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    encodedAcceptanceColumns := [9305, 41824, 41825, 41826, 41827, 41828, 41829, 41830, 41831, 41832, 41833, 41834, 41835, 41836, 41837],
    activeRowStart := 34382, activeRowEnd := 34391 }
, { sourceRowStart := 734, sourceRowEnd := 738,
    sourceColumnStart := 726, sourceColumnEnd := 728,
    sourceInputColumns := [646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661], sourceAcceptColumn := 726, sourceInverseColumn := 727,
    encodedInputColumns := [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    encodedAcceptanceColumns := [9319, 41838, 41839, 41840, 41841, 41842, 41843, 41844, 41845, 41846, 41847, 41848, 41849, 41850, 41851],
    activeRowStart := 34391, activeRowEnd := 34400 }
, { sourceRowStart := 760, sourceRowEnd := 764,
    sourceColumnStart := 749, sourceColumnEnd := 751,
    sourceInputColumns := [662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677], sourceAcceptColumn := 749, sourceInverseColumn := 750,
    encodedInputColumns := [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    encodedAcceptanceColumns := [9333, 41852, 41853, 41854, 41855, 41856, 41857, 41858, 41859, 41860, 41861, 41862, 41863, 41864, 41865],
    activeRowStart := 34400, activeRowEnd := 34409 }
, { sourceRowStart := 855, sourceRowEnd := 859,
    sourceColumnStart := 838, sourceColumnEnd := 840,
    sourceInputColumns := [772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787], sourceAcceptColumn := 838, sourceInverseColumn := 839,
    encodedInputColumns := [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    encodedAcceptanceColumns := [9443, 41866, 41867, 41868, 41869, 41870, 41871, 41872, 41873, 41874, 41875, 41876, 41877, 41878, 41879],
    activeRowStart := 34409, activeRowEnd := 34418 }
, { sourceRowStart := 881, sourceRowEnd := 885,
    sourceColumnStart := 861, sourceColumnEnd := 863,
    sourceInputColumns := [788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803], sourceAcceptColumn := 861, sourceInverseColumn := 862,
    encodedInputColumns := [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
    encodedAcceptanceColumns := [9457, 41880, 41881, 41882, 41883, 41884, 41885, 41886, 41887, 41888, 41889, 41890, 41891, 41892, 41893],
    activeRowStart := 34418, activeRowEnd := 34427 }
, { sourceRowStart := 907, sourceRowEnd := 911,
    sourceColumnStart := 884, sourceColumnEnd := 886,
    sourceInputColumns := [804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819], sourceAcceptColumn := 884, sourceInverseColumn := 885,
    encodedInputColumns := [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    encodedAcceptanceColumns := [9471, 41894, 41895, 41896, 41897, 41898, 41899, 41900, 41901, 41902, 41903, 41904, 41905, 41906, 41907],
    activeRowStart := 34427, activeRowEnd := 34436 }
, { sourceRowStart := 933, sourceRowEnd := 937,
    sourceColumnStart := 907, sourceColumnEnd := 909,
    sourceInputColumns := [820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835], sourceAcceptColumn := 907, sourceInverseColumn := 908,
    encodedInputColumns := [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
    encodedAcceptanceColumns := [9485, 41908, 41909, 41910, 41911, 41912, 41913, 41914, 41915, 41916, 41917, 41918, 41919, 41920, 41921],
    activeRowStart := 34436, activeRowEnd := 34445 }
, { sourceRowStart := 1028, sourceRowEnd := 1032,
    sourceColumnStart := 996, sourceColumnEnd := 998,
    sourceInputColumns := [930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945], sourceAcceptColumn := 996, sourceInverseColumn := 997,
    encodedInputColumns := [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
    encodedAcceptanceColumns := [9595, 41922, 41923, 41924, 41925, 41926, 41927, 41928, 41929, 41930, 41931, 41932, 41933, 41934, 41935],
    activeRowStart := 34445, activeRowEnd := 34454 }
, { sourceRowStart := 1054, sourceRowEnd := 1058,
    sourceColumnStart := 1019, sourceColumnEnd := 1021,
    sourceInputColumns := [946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961], sourceAcceptColumn := 1019, sourceInverseColumn := 1020,
    encodedInputColumns := [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160],
    encodedAcceptanceColumns := [9609, 41936, 41937, 41938, 41939, 41940, 41941, 41942, 41943, 41944, 41945, 41946, 41947, 41948, 41949],
    activeRowStart := 34454, activeRowEnd := 34463 }
, { sourceRowStart := 1080, sourceRowEnd := 1084,
    sourceColumnStart := 1042, sourceColumnEnd := 1044,
    sourceInputColumns := [962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977], sourceAcceptColumn := 1042, sourceInverseColumn := 1043,
    encodedInputColumns := [161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176],
    encodedAcceptanceColumns := [9623, 41950, 41951, 41952, 41953, 41954, 41955, 41956, 41957, 41958, 41959, 41960, 41961, 41962, 41963],
    activeRowStart := 34463, activeRowEnd := 34472 }
, { sourceRowStart := 1106, sourceRowEnd := 1110,
    sourceColumnStart := 1065, sourceColumnEnd := 1067,
    sourceInputColumns := [978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993], sourceAcceptColumn := 1065, sourceInverseColumn := 1066,
    encodedInputColumns := [177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192],
    encodedAcceptanceColumns := [9637, 41964, 41965, 41966, 41967, 41968, 41969, 41970, 41971, 41972, 41973, 41974, 41975, 41976, 41977],
    activeRowStart := 34472, activeRowEnd := 34481 }
, { sourceRowStart := 1201, sourceRowEnd := 1205,
    sourceColumnStart := 1154, sourceColumnEnd := 1156,
    sourceInputColumns := [1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103], sourceAcceptColumn := 1154, sourceInverseColumn := 1155,
    encodedInputColumns := [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208],
    encodedAcceptanceColumns := [9747, 41978, 41979, 41980, 41981, 41982, 41983, 41984, 41985, 41986, 41987, 41988, 41989, 41990, 41991],
    activeRowStart := 34481, activeRowEnd := 34490 }
, { sourceRowStart := 1227, sourceRowEnd := 1231,
    sourceColumnStart := 1177, sourceColumnEnd := 1179,
    sourceInputColumns := [1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119], sourceAcceptColumn := 1177, sourceInverseColumn := 1178,
    encodedInputColumns := [209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224],
    encodedAcceptanceColumns := [9761, 41992, 41993, 41994, 41995, 41996, 41997, 41998, 41999, 42000, 42001, 42002, 42003, 42004, 42005],
    activeRowStart := 34490, activeRowEnd := 34499 }
, { sourceRowStart := 1253, sourceRowEnd := 1257,
    sourceColumnStart := 1200, sourceColumnEnd := 1202,
    sourceInputColumns := [1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135], sourceAcceptColumn := 1200, sourceInverseColumn := 1201,
    encodedInputColumns := [225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240],
    encodedAcceptanceColumns := [9775, 42006, 42007, 42008, 42009, 42010, 42011, 42012, 42013, 42014, 42015, 42016, 42017, 42018, 42019],
    activeRowStart := 34499, activeRowEnd := 34508 }
, { sourceRowStart := 1279, sourceRowEnd := 1283,
    sourceColumnStart := 1223, sourceColumnEnd := 1225,
    sourceInputColumns := [1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151], sourceAcceptColumn := 1223, sourceInverseColumn := 1224,
    encodedInputColumns := [241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256],
    encodedAcceptanceColumns := [9789, 42020, 42021, 42022, 42023, 42024, 42025, 42026, 42027, 42028, 42029, 42030, 42031, 42032, 42033],
    activeRowStart := 34508, activeRowEnd := 34517 }
, { sourceRowStart := 1978, sourceRowEnd := 1982,
    sourceColumnStart := 1916, sourceColumnEnd := 1918,
    sourceInputColumns := [1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865], sourceAcceptColumn := 1916, sourceInverseColumn := 1917,
    encodedInputColumns := [257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272],
    encodedAcceptanceColumns := [18069, 42034, 42035, 42036, 42037, 42038, 42039, 42040, 42041, 42042, 42043, 42044, 42045, 42046, 42047],
    activeRowStart := 34517, activeRowEnd := 34526 }
, { sourceRowStart := 2004, sourceRowEnd := 2008,
    sourceColumnStart := 1939, sourceColumnEnd := 1941,
    sourceInputColumns := [1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881], sourceAcceptColumn := 1939, sourceInverseColumn := 1940,
    encodedInputColumns := [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288],
    encodedAcceptanceColumns := [18083, 42048, 42049, 42050, 42051, 42052, 42053, 42054, 42055, 42056, 42057, 42058, 42059, 42060, 42061],
    activeRowStart := 34526, activeRowEnd := 34535 }
, { sourceRowStart := 2030, sourceRowEnd := 2034,
    sourceColumnStart := 1962, sourceColumnEnd := 1964,
    sourceInputColumns := [1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897], sourceAcceptColumn := 1962, sourceInverseColumn := 1963,
    encodedInputColumns := [289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],
    encodedAcceptanceColumns := [18097, 42062, 42063, 42064, 42065, 42066, 42067, 42068, 42069, 42070, 42071, 42072, 42073, 42074, 42075],
    activeRowStart := 34535, activeRowEnd := 34544 }
, { sourceRowStart := 2056, sourceRowEnd := 2060,
    sourceColumnStart := 1985, sourceColumnEnd := 1987,
    sourceInputColumns := [1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913], sourceAcceptColumn := 1985, sourceInverseColumn := 1986,
    encodedInputColumns := [305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320],
    encodedAcceptanceColumns := [18111, 42076, 42077, 42078, 42079, 42080, 42081, 42082, 42083, 42084, 42085, 42086, 42087, 42088, 42089],
    activeRowStart := 34544, activeRowEnd := 34553 }
, { sourceRowStart := 2151, sourceRowEnd := 2155,
    sourceColumnStart := 2074, sourceColumnEnd := 2076,
    sourceInputColumns := [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], sourceAcceptColumn := 2074, sourceInverseColumn := 2075,
    encodedInputColumns := [321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336],
    encodedAcceptanceColumns := [18221, 42090, 42091, 42092, 42093, 42094, 42095, 42096, 42097, 42098, 42099, 42100, 42101, 42102, 42103],
    activeRowStart := 34553, activeRowEnd := 34562 }
, { sourceRowStart := 2177, sourceRowEnd := 2181,
    sourceColumnStart := 2097, sourceColumnEnd := 2099,
    sourceInputColumns := [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039], sourceAcceptColumn := 2097, sourceInverseColumn := 2098,
    encodedInputColumns := [337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352],
    encodedAcceptanceColumns := [18235, 42104, 42105, 42106, 42107, 42108, 42109, 42110, 42111, 42112, 42113, 42114, 42115, 42116, 42117],
    activeRowStart := 34562, activeRowEnd := 34571 }
, { sourceRowStart := 2203, sourceRowEnd := 2207,
    sourceColumnStart := 2120, sourceColumnEnd := 2122,
    sourceInputColumns := [2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055], sourceAcceptColumn := 2120, sourceInverseColumn := 2121,
    encodedInputColumns := [353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368],
    encodedAcceptanceColumns := [18249, 42118, 42119, 42120, 42121, 42122, 42123, 42124, 42125, 42126, 42127, 42128, 42129, 42130, 42131],
    activeRowStart := 34571, activeRowEnd := 34580 }
, { sourceRowStart := 2229, sourceRowEnd := 2233,
    sourceColumnStart := 2143, sourceColumnEnd := 2145,
    sourceInputColumns := [2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071], sourceAcceptColumn := 2143, sourceInverseColumn := 2144,
    encodedInputColumns := [369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384],
    encodedAcceptanceColumns := [18263, 42132, 42133, 42134, 42135, 42136, 42137, 42138, 42139, 42140, 42141, 42142, 42143, 42144, 42145],
    activeRowStart := 34580, activeRowEnd := 34589 }
, { sourceRowStart := 2324, sourceRowEnd := 2328,
    sourceColumnStart := 2232, sourceColumnEnd := 2234,
    sourceInputColumns := [2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181], sourceAcceptColumn := 2232, sourceInverseColumn := 2233,
    encodedInputColumns := [385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400],
    encodedAcceptanceColumns := [18373, 42146, 42147, 42148, 42149, 42150, 42151, 42152, 42153, 42154, 42155, 42156, 42157, 42158, 42159],
    activeRowStart := 34589, activeRowEnd := 34598 }
, { sourceRowStart := 2350, sourceRowEnd := 2354,
    sourceColumnStart := 2255, sourceColumnEnd := 2257,
    sourceInputColumns := [2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197], sourceAcceptColumn := 2255, sourceInverseColumn := 2256,
    encodedInputColumns := [401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416],
    encodedAcceptanceColumns := [18387, 42160, 42161, 42162, 42163, 42164, 42165, 42166, 42167, 42168, 42169, 42170, 42171, 42172, 42173],
    activeRowStart := 34598, activeRowEnd := 34607 }
, { sourceRowStart := 2376, sourceRowEnd := 2380,
    sourceColumnStart := 2278, sourceColumnEnd := 2280,
    sourceInputColumns := [2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213], sourceAcceptColumn := 2278, sourceInverseColumn := 2279,
    encodedInputColumns := [417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432],
    encodedAcceptanceColumns := [18401, 42174, 42175, 42176, 42177, 42178, 42179, 42180, 42181, 42182, 42183, 42184, 42185, 42186, 42187],
    activeRowStart := 34607, activeRowEnd := 34616 }
, { sourceRowStart := 2402, sourceRowEnd := 2406,
    sourceColumnStart := 2301, sourceColumnEnd := 2303,
    sourceInputColumns := [2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229], sourceAcceptColumn := 2301, sourceInverseColumn := 2302,
    encodedInputColumns := [433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448],
    encodedAcceptanceColumns := [18415, 42188, 42189, 42190, 42191, 42192, 42193, 42194, 42195, 42196, 42197, 42198, 42199, 42200, 42201],
    activeRowStart := 34616, activeRowEnd := 34625 }
, { sourceRowStart := 2497, sourceRowEnd := 2501,
    sourceColumnStart := 2390, sourceColumnEnd := 2392,
    sourceInputColumns := [2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339], sourceAcceptColumn := 2390, sourceInverseColumn := 2391,
    encodedInputColumns := [449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464],
    encodedAcceptanceColumns := [18525, 42202, 42203, 42204, 42205, 42206, 42207, 42208, 42209, 42210, 42211, 42212, 42213, 42214, 42215],
    activeRowStart := 34625, activeRowEnd := 34634 }
, { sourceRowStart := 2523, sourceRowEnd := 2527,
    sourceColumnStart := 2413, sourceColumnEnd := 2415,
    sourceInputColumns := [2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355], sourceAcceptColumn := 2413, sourceInverseColumn := 2414,
    encodedInputColumns := [465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480],
    encodedAcceptanceColumns := [18539, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227, 42228, 42229],
    activeRowStart := 34634, activeRowEnd := 34643 }
, { sourceRowStart := 2549, sourceRowEnd := 2553,
    sourceColumnStart := 2436, sourceColumnEnd := 2438,
    sourceInputColumns := [2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371], sourceAcceptColumn := 2436, sourceInverseColumn := 2437,
    encodedInputColumns := [481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496],
    encodedAcceptanceColumns := [18553, 42230, 42231, 42232, 42233, 42234, 42235, 42236, 42237, 42238, 42239, 42240, 42241, 42242, 42243],
    activeRowStart := 34643, activeRowEnd := 34652 }
, { sourceRowStart := 2575, sourceRowEnd := 2579,
    sourceColumnStart := 2459, sourceColumnEnd := 2461,
    sourceInputColumns := [2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387], sourceAcceptColumn := 2459, sourceInverseColumn := 2460,
    encodedInputColumns := [497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512],
    encodedAcceptanceColumns := [18567, 42244, 42245, 42246, 42247, 42248, 42249, 42250, 42251, 42252, 42253, 42254, 42255, 42256, 42257],
    activeRowStart := 34652, activeRowEnd := 34661 }
, { sourceRowStart := 3274, sourceRowEnd := 3278,
    sourceColumnStart := 3152, sourceColumnEnd := 3154,
    sourceInputColumns := [3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101], sourceAcceptColumn := 3152, sourceInverseColumn := 3153,
    encodedInputColumns := [513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528],
    encodedAcceptanceColumns := [26847, 42258, 42259, 42260, 42261, 42262, 42263, 42264, 42265, 42266, 42267, 42268, 42269, 42270, 42271],
    activeRowStart := 34661, activeRowEnd := 34670 }
, { sourceRowStart := 3300, sourceRowEnd := 3304,
    sourceColumnStart := 3175, sourceColumnEnd := 3177,
    sourceInputColumns := [3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117], sourceAcceptColumn := 3175, sourceInverseColumn := 3176,
    encodedInputColumns := [529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544],
    encodedAcceptanceColumns := [26861, 42272, 42273, 42274, 42275, 42276, 42277, 42278, 42279, 42280, 42281, 42282, 42283, 42284, 42285],
    activeRowStart := 34670, activeRowEnd := 34679 }
, { sourceRowStart := 3326, sourceRowEnd := 3330,
    sourceColumnStart := 3198, sourceColumnEnd := 3200,
    sourceInputColumns := [3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133], sourceAcceptColumn := 3198, sourceInverseColumn := 3199,
    encodedInputColumns := [545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560],
    encodedAcceptanceColumns := [26875, 42286, 42287, 42288, 42289, 42290, 42291, 42292, 42293, 42294, 42295, 42296, 42297, 42298, 42299],
    activeRowStart := 34679, activeRowEnd := 34688 }
, { sourceRowStart := 3352, sourceRowEnd := 3356,
    sourceColumnStart := 3221, sourceColumnEnd := 3223,
    sourceInputColumns := [3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149], sourceAcceptColumn := 3221, sourceInverseColumn := 3222,
    encodedInputColumns := [561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576],
    encodedAcceptanceColumns := [26889, 42300, 42301, 42302, 42303, 42304, 42305, 42306, 42307, 42308, 42309, 42310, 42311, 42312, 42313],
    activeRowStart := 34688, activeRowEnd := 34697 }
, { sourceRowStart := 3447, sourceRowEnd := 3451,
    sourceColumnStart := 3310, sourceColumnEnd := 3312,
    sourceInputColumns := [3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259], sourceAcceptColumn := 3310, sourceInverseColumn := 3311,
    encodedInputColumns := [577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592],
    encodedAcceptanceColumns := [26999, 42314, 42315, 42316, 42317, 42318, 42319, 42320, 42321, 42322, 42323, 42324, 42325, 42326, 42327],
    activeRowStart := 34697, activeRowEnd := 34706 }
, { sourceRowStart := 3473, sourceRowEnd := 3477,
    sourceColumnStart := 3333, sourceColumnEnd := 3335,
    sourceInputColumns := [3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275], sourceAcceptColumn := 3333, sourceInverseColumn := 3334,
    encodedInputColumns := [593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608],
    encodedAcceptanceColumns := [27013, 42328, 42329, 42330, 42331, 42332, 42333, 42334, 42335, 42336, 42337, 42338, 42339, 42340, 42341],
    activeRowStart := 34706, activeRowEnd := 34715 }
, { sourceRowStart := 3499, sourceRowEnd := 3503,
    sourceColumnStart := 3356, sourceColumnEnd := 3358,
    sourceInputColumns := [3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291], sourceAcceptColumn := 3356, sourceInverseColumn := 3357,
    encodedInputColumns := [609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624],
    encodedAcceptanceColumns := [27027, 42342, 42343, 42344, 42345, 42346, 42347, 42348, 42349, 42350, 42351, 42352, 42353, 42354, 42355],
    activeRowStart := 34715, activeRowEnd := 34724 }
, { sourceRowStart := 3525, sourceRowEnd := 3529,
    sourceColumnStart := 3379, sourceColumnEnd := 3381,
    sourceInputColumns := [3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307], sourceAcceptColumn := 3379, sourceInverseColumn := 3380,
    encodedInputColumns := [625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640],
    encodedAcceptanceColumns := [27041, 42356, 42357, 42358, 42359, 42360, 42361, 42362, 42363, 42364, 42365, 42366, 42367, 42368, 42369],
    activeRowStart := 34724, activeRowEnd := 34733 }
, { sourceRowStart := 3620, sourceRowEnd := 3624,
    sourceColumnStart := 3468, sourceColumnEnd := 3470,
    sourceInputColumns := [3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417], sourceAcceptColumn := 3468, sourceInverseColumn := 3469,
    encodedInputColumns := [641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656],
    encodedAcceptanceColumns := [27151, 42370, 42371, 42372, 42373, 42374, 42375, 42376, 42377, 42378, 42379, 42380, 42381, 42382, 42383],
    activeRowStart := 34733, activeRowEnd := 34742 }
, { sourceRowStart := 3646, sourceRowEnd := 3650,
    sourceColumnStart := 3491, sourceColumnEnd := 3493,
    sourceInputColumns := [3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433], sourceAcceptColumn := 3491, sourceInverseColumn := 3492,
    encodedInputColumns := [657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672],
    encodedAcceptanceColumns := [27165, 42384, 42385, 42386, 42387, 42388, 42389, 42390, 42391, 42392, 42393, 42394, 42395, 42396, 42397],
    activeRowStart := 34742, activeRowEnd := 34751 }
, { sourceRowStart := 3672, sourceRowEnd := 3676,
    sourceColumnStart := 3514, sourceColumnEnd := 3516,
    sourceInputColumns := [3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449], sourceAcceptColumn := 3514, sourceInverseColumn := 3515,
    encodedInputColumns := [673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688],
    encodedAcceptanceColumns := [27179, 42398, 42399, 42400, 42401, 42402, 42403, 42404, 42405, 42406, 42407, 42408, 42409, 42410, 42411],
    activeRowStart := 34751, activeRowEnd := 34760 }
, { sourceRowStart := 3698, sourceRowEnd := 3702,
    sourceColumnStart := 3537, sourceColumnEnd := 3539,
    sourceInputColumns := [3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465], sourceAcceptColumn := 3537, sourceInverseColumn := 3538,
    encodedInputColumns := [689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704],
    encodedAcceptanceColumns := [27193, 42412, 42413, 42414, 42415, 42416, 42417, 42418, 42419, 42420, 42421, 42422, 42423, 42424, 42425],
    activeRowStart := 34760, activeRowEnd := 34769 }
, { sourceRowStart := 3793, sourceRowEnd := 3797,
    sourceColumnStart := 3626, sourceColumnEnd := 3628,
    sourceInputColumns := [3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575], sourceAcceptColumn := 3626, sourceInverseColumn := 3627,
    encodedInputColumns := [705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720],
    encodedAcceptanceColumns := [27303, 42426, 42427, 42428, 42429, 42430, 42431, 42432, 42433, 42434, 42435, 42436, 42437, 42438, 42439],
    activeRowStart := 34769, activeRowEnd := 34778 }
, { sourceRowStart := 3819, sourceRowEnd := 3823,
    sourceColumnStart := 3649, sourceColumnEnd := 3651,
    sourceInputColumns := [3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591], sourceAcceptColumn := 3649, sourceInverseColumn := 3650,
    encodedInputColumns := [721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736],
    encodedAcceptanceColumns := [27317, 42440, 42441, 42442, 42443, 42444, 42445, 42446, 42447, 42448, 42449, 42450, 42451, 42452, 42453],
    activeRowStart := 34778, activeRowEnd := 34787 }
, { sourceRowStart := 3845, sourceRowEnd := 3849,
    sourceColumnStart := 3672, sourceColumnEnd := 3674,
    sourceInputColumns := [3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607], sourceAcceptColumn := 3672, sourceInverseColumn := 3673,
    encodedInputColumns := [737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752],
    encodedAcceptanceColumns := [27331, 42454, 42455, 42456, 42457, 42458, 42459, 42460, 42461, 42462, 42463, 42464, 42465, 42466, 42467],
    activeRowStart := 34787, activeRowEnd := 34796 }
, { sourceRowStart := 3871, sourceRowEnd := 3875,
    sourceColumnStart := 3695, sourceColumnEnd := 3697,
    sourceInputColumns := [3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623], sourceAcceptColumn := 3695, sourceInverseColumn := 3696,
    encodedInputColumns := [753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768],
    encodedAcceptanceColumns := [27345, 42468, 42469, 42470, 42471, 42472, 42473, 42474, 42475, 42476, 42477, 42478, 42479, 42480, 42481],
    activeRowStart := 34796, activeRowEnd := 34805 }
, { sourceRowStart := 4570, sourceRowEnd := 4574,
    sourceColumnStart := 4388, sourceColumnEnd := 4390,
    sourceInputColumns := [4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337], sourceAcceptColumn := 4388, sourceInverseColumn := 4389,
    encodedInputColumns := [769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784],
    encodedAcceptanceColumns := [35625, 42482, 42483, 42484, 42485, 42486, 42487, 42488, 42489, 42490, 42491, 42492, 42493, 42494, 42495],
    activeRowStart := 34805, activeRowEnd := 34814 }
, { sourceRowStart := 4596, sourceRowEnd := 4600,
    sourceColumnStart := 4411, sourceColumnEnd := 4413,
    sourceInputColumns := [4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353], sourceAcceptColumn := 4411, sourceInverseColumn := 4412,
    encodedInputColumns := [785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800],
    encodedAcceptanceColumns := [35639, 42496, 42497, 42498, 42499, 42500, 42501, 42502, 42503, 42504, 42505, 42506, 42507, 42508, 42509],
    activeRowStart := 34814, activeRowEnd := 34823 }
, { sourceRowStart := 4622, sourceRowEnd := 4626,
    sourceColumnStart := 4434, sourceColumnEnd := 4436,
    sourceInputColumns := [4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369], sourceAcceptColumn := 4434, sourceInverseColumn := 4435,
    encodedInputColumns := [801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816],
    encodedAcceptanceColumns := [35653, 42510, 42511, 42512, 42513, 42514, 42515, 42516, 42517, 42518, 42519, 42520, 42521, 42522, 42523],
    activeRowStart := 34823, activeRowEnd := 34832 }
, { sourceRowStart := 4648, sourceRowEnd := 4652,
    sourceColumnStart := 4457, sourceColumnEnd := 4459,
    sourceInputColumns := [4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385], sourceAcceptColumn := 4457, sourceInverseColumn := 4458,
    encodedInputColumns := [817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832],
    encodedAcceptanceColumns := [35667, 42524, 42525, 42526, 42527, 42528, 42529, 42530, 42531, 42532, 42533, 42534, 42535, 42536, 42537],
    activeRowStart := 34832, activeRowEnd := 34841 }
, { sourceRowStart := 4743, sourceRowEnd := 4747,
    sourceColumnStart := 4546, sourceColumnEnd := 4548,
    sourceInputColumns := [4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495], sourceAcceptColumn := 4546, sourceInverseColumn := 4547,
    encodedInputColumns := [833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848],
    encodedAcceptanceColumns := [35777, 42538, 42539, 42540, 42541, 42542, 42543, 42544, 42545, 42546, 42547, 42548, 42549, 42550, 42551],
    activeRowStart := 34841, activeRowEnd := 34850 }
, { sourceRowStart := 4769, sourceRowEnd := 4773,
    sourceColumnStart := 4569, sourceColumnEnd := 4571,
    sourceInputColumns := [4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511], sourceAcceptColumn := 4569, sourceInverseColumn := 4570,
    encodedInputColumns := [849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864],
    encodedAcceptanceColumns := [35791, 42552, 42553, 42554, 42555, 42556, 42557, 42558, 42559, 42560, 42561, 42562, 42563, 42564, 42565],
    activeRowStart := 34850, activeRowEnd := 34859 }
, { sourceRowStart := 4795, sourceRowEnd := 4799,
    sourceColumnStart := 4592, sourceColumnEnd := 4594,
    sourceInputColumns := [4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527], sourceAcceptColumn := 4592, sourceInverseColumn := 4593,
    encodedInputColumns := [865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880],
    encodedAcceptanceColumns := [35805, 42566, 42567, 42568, 42569, 42570, 42571, 42572, 42573, 42574, 42575, 42576, 42577, 42578, 42579],
    activeRowStart := 34859, activeRowEnd := 34868 }
, { sourceRowStart := 4821, sourceRowEnd := 4825,
    sourceColumnStart := 4615, sourceColumnEnd := 4617,
    sourceInputColumns := [4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538, 4539, 4540, 4541, 4542, 4543], sourceAcceptColumn := 4615, sourceInverseColumn := 4616,
    encodedInputColumns := [881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896],
    encodedAcceptanceColumns := [35819, 42580, 42581, 42582, 42583, 42584, 42585, 42586, 42587, 42588, 42589, 42590, 42591, 42592, 42593],
    activeRowStart := 34868, activeRowEnd := 34877 }
, { sourceRowStart := 4916, sourceRowEnd := 4920,
    sourceColumnStart := 4704, sourceColumnEnd := 4706,
    sourceInputColumns := [4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653], sourceAcceptColumn := 4704, sourceInverseColumn := 4705,
    encodedInputColumns := [897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912],
    encodedAcceptanceColumns := [35929, 42594, 42595, 42596, 42597, 42598, 42599, 42600, 42601, 42602, 42603, 42604, 42605, 42606, 42607],
    activeRowStart := 34877, activeRowEnd := 34886 }
, { sourceRowStart := 4942, sourceRowEnd := 4946,
    sourceColumnStart := 4727, sourceColumnEnd := 4729,
    sourceInputColumns := [4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669], sourceAcceptColumn := 4727, sourceInverseColumn := 4728,
    encodedInputColumns := [913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928],
    encodedAcceptanceColumns := [35943, 42608, 42609, 42610, 42611, 42612, 42613, 42614, 42615, 42616, 42617, 42618, 42619, 42620, 42621],
    activeRowStart := 34886, activeRowEnd := 34895 }
, { sourceRowStart := 4968, sourceRowEnd := 4972,
    sourceColumnStart := 4750, sourceColumnEnd := 4752,
    sourceInputColumns := [4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685], sourceAcceptColumn := 4750, sourceInverseColumn := 4751,
    encodedInputColumns := [929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944],
    encodedAcceptanceColumns := [35957, 42622, 42623, 42624, 42625, 42626, 42627, 42628, 42629, 42630, 42631, 42632, 42633, 42634, 42635],
    activeRowStart := 34895, activeRowEnd := 34904 }
, { sourceRowStart := 4994, sourceRowEnd := 4998,
    sourceColumnStart := 4773, sourceColumnEnd := 4775,
    sourceInputColumns := [4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701], sourceAcceptColumn := 4773, sourceInverseColumn := 4774,
    encodedInputColumns := [945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960],
    encodedAcceptanceColumns := [35971, 42636, 42637, 42638, 42639, 42640, 42641, 42642, 42643, 42644, 42645, 42646, 42647, 42648, 42649],
    activeRowStart := 34904, activeRowEnd := 34913 }
, { sourceRowStart := 5089, sourceRowEnd := 5093,
    sourceColumnStart := 4862, sourceColumnEnd := 4864,
    sourceInputColumns := [4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810, 4811], sourceAcceptColumn := 4862, sourceInverseColumn := 4863,
    encodedInputColumns := [961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976],
    encodedAcceptanceColumns := [36081, 42650, 42651, 42652, 42653, 42654, 42655, 42656, 42657, 42658, 42659, 42660, 42661, 42662, 42663],
    activeRowStart := 34913, activeRowEnd := 34922 }
, { sourceRowStart := 5115, sourceRowEnd := 5119,
    sourceColumnStart := 4885, sourceColumnEnd := 4887,
    sourceInputColumns := [4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827], sourceAcceptColumn := 4885, sourceInverseColumn := 4886,
    encodedInputColumns := [977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992],
    encodedAcceptanceColumns := [36095, 42664, 42665, 42666, 42667, 42668, 42669, 42670, 42671, 42672, 42673, 42674, 42675, 42676, 42677],
    activeRowStart := 34922, activeRowEnd := 34931 }
, { sourceRowStart := 5141, sourceRowEnd := 5145,
    sourceColumnStart := 4908, sourceColumnEnd := 4910,
    sourceInputColumns := [4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843], sourceAcceptColumn := 4908, sourceInverseColumn := 4909,
    encodedInputColumns := [993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    encodedAcceptanceColumns := [36109, 42678, 42679, 42680, 42681, 42682, 42683, 42684, 42685, 42686, 42687, 42688, 42689, 42690, 42691],
    activeRowStart := 34931, activeRowEnd := 34940 }
, { sourceRowStart := 5167, sourceRowEnd := 5171,
    sourceColumnStart := 4931, sourceColumnEnd := 4933,
    sourceInputColumns := [4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859], sourceAcceptColumn := 4931, sourceInverseColumn := 4932,
    encodedInputColumns := [1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024],
    encodedAcceptanceColumns := [36123, 42692, 42693, 42694, 42695, 42696, 42697, 42698, 42699, 42700, 42701, 42702, 42703, 42704, 42705],
    activeRowStart := 34940, activeRowEnd := 34949 }
]
def gateArity : Nat := 48
def matrixBindings : List MatrixBinding :=
[
  { role := .selector, index := 0 }
, { role := (.productLeft 0), index := 3 }
, { role := (.productLeft 1), index := 4 }
, { role := (.productLeft 2), index := 5 }
, { role := (.productLeft 3), index := 6 }
, { role := (.productLeft 4), index := 7 }
, { role := (.productLeft 5), index := 8 }
, { role := (.productLeft 6), index := 9 }
, { role := (.productLeft 7), index := 10 }
, { role := (.productLeft 8), index := 11 }
, { role := (.productLeft 9), index := 12 }
, { role := (.productLeft 10), index := 13 }
, { role := (.productLeft 11), index := 14 }
, { role := (.productLeft 12), index := 15 }
, { role := (.productLeft 13), index := 16 }
, { role := (.productLeft 14), index := 17 }
, { role := (.productLeft 15), index := 18 }
, { role := (.productLeft 16), index := 19 }
, { role := (.productLeft 17), index := 20 }
, { role := (.productRight 0), index := 21 }
, { role := (.productRight 1), index := 22 }
, { role := (.productRight 2), index := 23 }
, { role := (.productRight 3), index := 24 }
, { role := (.productRight 4), index := 25 }
, { role := (.productRight 5), index := 26 }
, { role := (.productRight 6), index := 27 }
, { role := (.productRight 7), index := 28 }
, { role := (.productRight 8), index := 29 }
, { role := (.productRight 9), index := 30 }
, { role := (.productRight 10), index := 31 }
, { role := (.productRight 11), index := 32 }
, { role := (.productRight 12), index := 33 }
, { role := (.productRight 13), index := 34 }
, { role := (.productRight 14), index := 35 }
, { role := (.productRight 15), index := 36 }
, { role := (.productRight 16), index := 37 }
, { role := (.productRight 17), index := 38 }
, { role := .productOut, index := 39 }
, { role := .quadraticBitLeft, index := 44 }
, { role := .quadraticBitRight, index := 45 }
]
def activeRows : List ActiveRow :=
[
  [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 0), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 1), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 2), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 3), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 4), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 5), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 6), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 7), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 8), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 9), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 10), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 11), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 12), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 13), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨(.productLeft 0), [⟨(.chunkBit 0), 1⟩]⟩, ⟨(.productLeft 1), [⟨(.chunkBit 2), 3⟩]⟩, ⟨(.productLeft 2), [⟨(.chunkBit 4), 9⟩]⟩, ⟨(.productLeft 3), [⟨(.chunkBit 6), 27⟩]⟩, ⟨(.productLeft 4), [⟨(.treeOutput 0), 81⟩]⟩, ⟨(.productLeft 5), [⟨(.treeOutput 2), 243⟩]⟩, ⟨(.productLeft 6), [⟨(.treeOutput 4), 729⟩]⟩, ⟨(.productLeft 7), [⟨(.chunkBit 8), 2187⟩]⟩, ⟨(.productLeft 8), [⟨(.chunkBit 10), 6561⟩]⟩, ⟨(.productLeft 9), [⟨(.chunkBit 12), 19683⟩]⟩, ⟨(.productLeft 10), [⟨(.chunkBit 14), 59049⟩]⟩, ⟨(.productLeft 11), [⟨(.treeOutput 7), 177147⟩]⟩, ⟨(.productLeft 12), [⟨(.treeOutput 9), 531441⟩]⟩, ⟨(.productLeft 13), [⟨(.treeOutput 11), 1594323⟩]⟩, ⟨(.productRight 0), [⟨(.chunkBit 1), 1⟩]⟩, ⟨(.productRight 1), [⟨(.chunkBit 3), 1⟩]⟩, ⟨(.productRight 2), [⟨(.chunkBit 5), 1⟩]⟩, ⟨(.productRight 3), [⟨(.chunkBit 7), 1⟩]⟩, ⟨(.productRight 4), [⟨(.treeOutput 1), 1⟩]⟩, ⟨(.productRight 5), [⟨(.treeOutput 3), 1⟩]⟩, ⟨(.productRight 6), [⟨(.treeOutput 5), 1⟩]⟩, ⟨(.productRight 7), [⟨(.chunkBit 9), 1⟩]⟩, ⟨(.productRight 8), [⟨(.chunkBit 11), 1⟩]⟩, ⟨(.productRight 9), [⟨(.chunkBit 13), 1⟩]⟩, ⟨(.productRight 10), [⟨(.chunkBit 15), 1⟩]⟩, ⟨(.productRight 11), [⟨(.treeOutput 8), 1⟩]⟩, ⟨(.productRight 12), [⟨(.treeOutput 10), 1⟩]⟩, ⟨(.productRight 13), [⟨(.treeOutput 12), 1⟩]⟩, ⟨.productOut, [⟨(.treeOutput 0), 1⟩, ⟨(.treeOutput 1), 3⟩, ⟨(.treeOutput 2), 9⟩, ⟨(.treeOutput 3), 27⟩, ⟨(.treeOutput 4), 81⟩, ⟨(.treeOutput 5), 243⟩, ⟨(.treeOutput 6), 729⟩, ⟨(.treeOutput 7), 2187⟩, ⟨(.treeOutput 8), 6561⟩, ⟨(.treeOutput 9), 19683⟩, ⟨(.treeOutput 10), 59049⟩, ⟨(.treeOutput 11), 177147⟩, ⟨(.treeOutput 12), 531441⟩, ⟨(.treeOutput 13), 1594323⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨(.productLeft 0), [⟨(.treeOutput 6), 1⟩]⟩, ⟨(.productRight 0), [⟨(.treeOutput 13), 1⟩]⟩, ⟨.productOut, [⟨.one, 1⟩, ⟨.accept, -1⟩]⟩]
]
def polynomialTerms : List PolynomialTerm :=
[
  ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 0), 1⟩, ⟨(.productRight 0), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 1), 1⟩, ⟨(.productRight 1), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 2), 1⟩, ⟨(.productRight 2), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 3), 1⟩, ⟨(.productRight 3), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 4), 1⟩, ⟨(.productRight 4), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 5), 1⟩, ⟨(.productRight 5), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 6), 1⟩, ⟨(.productRight 6), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 7), 1⟩, ⟨(.productRight 7), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 8), 1⟩, ⟨(.productRight 8), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 9), 1⟩, ⟨(.productRight 9), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 10), 1⟩, ⟨(.productRight 10), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 11), 1⟩, ⟨(.productRight 11), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 12), 1⟩, ⟨(.productRight 12), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 13), 1⟩, ⟨(.productRight 13), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 14), 1⟩, ⟨(.productRight 14), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 15), 1⟩, ⟨(.productRight 15), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 16), 1⟩, ⟨(.productRight 16), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 17), 1⟩, ⟨(.productRight 17), 1⟩]⟩
, ⟨-1, [⟨.selector, 1⟩, ⟨.productOut, 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 4⟩]⟩
, ⟨-2, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 3⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 2⟩]⟩
, ⟨-7, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 4⟩]⟩
, ⟨14, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 3⟩]⟩
, ⟨-7, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 2⟩]⟩
]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifactData
