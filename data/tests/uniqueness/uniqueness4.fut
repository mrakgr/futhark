-- This test inspired by code often created by
-- arrays-of-tuples-to-tuple-of-arrays transformation.
-- ==
-- input {
-- }
-- output {
--   [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
--   [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
-- }

fun ([f64], [f64]) main() =
  let n = 10 in
  loop (looparr = (copy(replicate(n,0.0)),
                   copy(replicate(n,0.0)))) = for i < n  do
    let (a, b) = looparr in
    let a[ i ] = 0.0 in
    let b[ i ] = 0.0 in
    (a, b)
  in looparr
