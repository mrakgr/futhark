-- Some simple currying of operators.
-- ==
-- input {
--   [-3,-2,-1,0,1,2,3]
-- }
-- output {
--   [-5, -4, -3, -2, -1, 0, 1]
--   [5, 4, 3, 2, 1, 0, -1]
-- }

fun ([int],[int]) main([int] a) =
  (map(- 2, a), map(2 -, a))
