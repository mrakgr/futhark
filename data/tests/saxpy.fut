-- Single-Precision A·X Plus Y
--
-- ==
-- input {
--   2.0f32
--   [1.0f32,2.0f32,3.0f32]
--   [4.0f32,5.0f32,6.0f32]
-- }
-- output {
--   [6.0f32, 9.0f32, 12.0f32]
-- }

fun [f32] main(f32 a, [f32] x, [f32] y) =
  zipWith(+, map(a*, x), y)
