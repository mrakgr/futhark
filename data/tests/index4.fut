-- Test indexing of a high-dimension array!
-- ==
-- input {
--   [[[1,2,3], [4,5,6], [7,8,9]], [[2,1,3], [4,6,5], [8,7,9]]]
--   1
--   1
-- }
-- output {
--   [4,6,5]
-- }
fun []int main([][][]int a, int i, int j) =
  a[i,j]
