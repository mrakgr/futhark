-- Bitwise operations on i8 values.
--
-- ==
-- input { 0  0i8  0i8 } output {  0i8 }
-- input { 0  0i8  1i8 } output {  0i8 }
-- input { 0  0i8 -1i8 } output {  0i8 }
-- input { 0  1i8  0i8 } output {  0i8 }
-- input { 0  1i8  1i8 } output {  1i8 }
-- input { 0  1i8 -1i8 } output {  1i8 }
-- input { 0 -1i8  0i8 } output {  0i8 }
-- input { 0 -1i8  1i8 } output {  1i8 }
-- input { 0 -1i8 -1i8 } output { -1i8 }
--
-- input { 1  0i8  0i8 } output {  0i8 }
-- input { 1  0i8  1i8 } output {  1i8 }
-- input { 1  0i8 -1i8 } output { -1i8 }
-- input { 1  1i8  0i8 } output {  1i8 }
-- input { 1  1i8  1i8 } output {  1i8 }
-- input { 1  1i8 -1i8 } output { -1i8 }
-- input { 1 -1i8  0i8 } output { -1i8 }
-- input { 1 -1i8  1i8 } output { -1i8 }
-- input { 1 -1i8 -1i8 } output { -1i8 }
-- input { 1 64i8 32i8 } output { 96i8 }
--
-- input { 2  0i8  0i8 } output {  0i8 }
-- input { 2  0i8  1i8 } output {  1i8 }
-- input { 2  0i8 -1i8 } output { -1i8 }
-- input { 2  1i8  0i8 } output {  1i8 }
-- input { 2  1i8  1i8 } output {  0i8 }
-- input { 2  1i8 -1i8 } output { -2i8 }
-- input { 2 -1i8  0i8 } output { -1i8 }
-- input { 2 -1i8  1i8 } output { -2i8 }
-- input { 2 -1i8 -1i8 } output {  0i8 }
-- input { 2 64i8 32i8 } output { 96i8 }
--
-- input { 3  0i8  0i8 } output {  0i8 }
-- input { 3  0i8  1i8 } output {  0i8 }
-- input { 3  1i8  0i8 } output {  1i8 }
-- input { 3  1i8  1i8 } output {  2i8 }
-- input { 3 -1i8  0i8 } output { -1i8 }
-- input { 3 -1i8  1i8 } output { -2i8 }
--
-- input { 4  0i8  0i8 } output {  0i8 }
-- input { 4  0i8  1i8 } output {  0i8 }
-- input { 4  1i8  0i8 } output {  1i8 }
-- input { 4  1i8  1i8 } output {  0i8 }
-- input { 4  2i8  1i8 } output {  1i8 }
-- input { 4 -1i8  0i8 } output { -1i8 }
-- input { 4 -1i8  1i8 } output { -1i8 }
--
-- input { 5  0i8  0i8 } output {  0i8 }
-- input { 5  0i8  1i8 } output {  0i8 }
-- input { 5  1i8  0i8 } output {  1i8 }
-- input { 5  1i8  1i8 } output {  0i8 }
-- input { 5  2i8  1i8 } output {  1i8 }
-- input { 5 -1i8  0i8 } output { -1i8 }
-- input { 5 -1i8  1i8 } output { 127i8 }

fun i8 main(int f, i8 x, i8 y) =
  if      f == 0 then x & y
  else if f == 1 then x | y
  else if f == 2 then x ^ y
  else if f == 3 then x << y
  else if f == 4 then x >> y
  else                x >>> y
