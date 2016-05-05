-- Multiplicative inverse on 16-bit numbers.  Returned as a 32-bit
-- number to print better (because we do not print unsigned types).
-- At one point the compiler missimplified the convergence loop.
--
-- ==
-- input { 2i16 } output { 32769i32 }
-- input { 33799i16 } output { 28110i32 }

fun u32 main(u16 a) =
  let b = 0x10001u32 in
  let u = 0i32 in
  let v = 1i32 in
  loop ((a,b,u,v)) = while a > 0u16 do
    let q = b / u32(a) in
    let r = b % u32(a) in

    let b = u32(a) in
    let a = u16(r) in

    let t = v in
    let v = u - i32(q) * v in
    let u = t in
    (a,b,u,v) in

  u32(if u < 0 then u + 0x10001 else u)
