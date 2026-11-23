(module
  (memory 1 1)

  (func (export "hash") (param $rounds i32) (result i32)
    (local $i i32) (local $hash i32)
    i32.const 0x811c9dc5
    local.set $hash
    block $done
      local.get $rounds
      i32.eqz
      br_if $done
      loop $next
        local.get $hash
        local.get $i
        i32.const 255
        i32.and
        i32.xor
        i32.const 0x01000193
        i32.mul
        local.set $hash
        local.get $i
        i32.const 1
        i32.add
        local.tee $i
        local.get $rounds
        i32.lt_u
        br_if $next
      end
    end
    local.get $hash)

  (func (export "memory") (param $rounds i32) (param $mask i32) (result i32)
    (local $seed i32) (local $address i32) (local $value i32) (local $sum i32)
    i32.const 0x12345678
    local.set $seed
    block $done
      local.get $rounds
      i32.eqz
      br_if $done
      loop $next
        local.get $seed
        i32.const 1664525
        i32.mul
        i32.const 1013904223
        i32.add
        local.tee $seed
        i32.const 28
        i32.shr_u
        local.get $mask
        i32.and
        i32.const 2
        i32.shl
        local.tee $address
        local.get $address
        i32.load
        local.get $seed
        i32.add
        local.tee $value
        i32.store
        local.get $sum
        local.get $value
        i32.add
        local.set $sum
        local.get $rounds
        i32.const 1
        i32.sub
        local.tee $rounds
        br_if $next
      end
    end
    local.get $sum))
