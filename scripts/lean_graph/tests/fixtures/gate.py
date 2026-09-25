"""Small process fixture for runner tests; this is not a protocol validator."""

import json
import sys

value = json.load(open(sys.argv[1]))
assert 0 <= value["opening"] < value["bound"]
assert value["commitment"] == value["key"] * value["opening"]
assert value["public"] == value["opening"]
assert value["evaluation"] == value["matrix"] * value["opening"]
print("fixture validity checked")
print("test required_case ... ok")
print("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
