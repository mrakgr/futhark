-- If the reduction function accumulator type is unique, consume the
-- initial value, but only as much as is actually unique!
-- ==
-- input {
--   [0,0,0,0,0,0,0,0,0,0]
--   [1,1,1,1,1,1,1,1,1,1]
-- }
-- output {
--   [1i32, 11i32, 21i32, 31i32, 41i32, 51i32, 61i32, 71i32, 81i32, 91i32]
-- }

fun []int main(*[]int a,[]int b) =
  let (x,y) =
    reduce(fn (*[]int, []int) ((*[]int, []int) acc, ([]int, []int) arr) =>
             let (a1,b1) = acc
             let (a2,b2) = arr
             in (zipWith(+, a1, a2),
                 zipWith(*, b1, b2)),
           (a,b),
           zip(copy(replicate(10,iota(10))),
               replicate(10,iota(10)))) in
  map(+, zip(b, x)) -- Should be OK, because only a has been consumed.
