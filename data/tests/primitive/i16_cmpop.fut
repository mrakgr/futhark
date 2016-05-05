-- Test comparison of i16 values.
--
-- ==
-- input {  0i16  0i16 } output { False True True }
-- input {  1i16  2i16 } output { True False True }
-- input { -1i16  1i16 } output { True False True }
-- input {  1i16 -1i16 } output { False False False }
-- input { -2i16 -1i16 } output { True False True }

fun (bool, bool, bool) main(i16 x, i16 y) =
  (x < y, x == y, x <= y)
