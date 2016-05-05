-- Test of division-like operators for u16 values.
--
-- ==
-- input {  7u16  3u16 } output {  2u16  1u16  2u16  1u16 }
-- input {  32768u16  9u16 } output {  3640u16  8u16  3640u16  8u16 }

fun (u16,u16,u16,u16) main(u16 x, u16 y) =
  (x / y, x % y, x // y, x %% y)
