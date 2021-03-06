-- Test a redomap with map-out where each element is also an array.
--
-- ==
-- input { 5 2 }
-- output { [[0i32, 1i32],
--           [2i32, 3i32],
--           [4i32, 5i32],
--           [6i32, 7i32],
--           [8i32, 9i32]]
--          False
-- }

fun ([][]int, bool) main(int n, int m) =
  let ass = map (fn [m]int (int l) =>
                   map(+l*m, iota(m)),
                 iota(n))
  let ps = zipWith(fn bool ([]int as, int i) =>
                     unsafe as[i] % 2 == 0,
                   ass, map(%m, iota(n)))
  in (ass, reduce(&&, True, ps))
