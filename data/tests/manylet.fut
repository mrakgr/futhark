-- Parser test.  'in' is optional except at the end of a chain of
-- let/loop-bindings.

fun [int,n] main(*[int,n] a, int x) =
  let y = x + 2
  let z = y + 3 + x
  loop ((a,z)) = for i < n do
    let tmp = z * z
    let a[i] = tmp
    let b = a with [i] <- a[i]-1
    in (b, tmp+2)
  in a
