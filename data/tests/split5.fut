-- Split with a multi-dimensional array.
-- ==
-- input {
--   2
--   [[4,3],[3,2],[2,1],[1,0]]
-- }
-- output {
--   [[4,3],[3,2]]
--   [[2,1],[1,0]]
-- }
fun ([[int]], [[int]]) main(int n, [[int]] a) =
  split( (n), a)
