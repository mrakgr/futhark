-- If the reduction function accumulator type is unique, consume the
-- initial value, but only as much as is actually unique!
-- ==
-- input {
--   [0,1,2,3,4,5,6,7,8,9]
--   [9,8,7,6,5,4,3,2,1,0]
-- }
-- output {
--   20
-- }

fun int main(*[int] a,[int] b) =
  let c =
    scan(fn (*[int], [int]) ((*[int], [int]) acc, ([int], [int]) i) =>
             let (a2,b2) = acc in (a2,b2),
           (a,b), zip(replicate(10,iota(10)),
                      replicate(10,iota(10)))) in
  size(0,c) + size(0,b) -- Should be OK, because only a has been
                        -- consumed.
