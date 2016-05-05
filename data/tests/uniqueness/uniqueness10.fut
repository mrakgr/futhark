-- Test that complex shadowing does not break alias analysis.
-- ==
-- input {
-- }
-- output {
--   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
-- }

fun *[int] main() =
  let n = 10 in
  let a = iota(n) in
  let c = let (a, b) = (iota(n), a) in let a[0] = 42 in a
  in a -- OK, because the outer a was never consumed.
