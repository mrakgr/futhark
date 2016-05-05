-- ==
-- input {
--   (100, 1)
-- }
-- output {
--   1
-- }
-- structure { DoLoop 0 }

fun int main(int chunk, int m) =
  loop (m) = for j < chunk do
      let chunk_in = chunk+1 in
      -- setting chunk_in to chunk will enable a simplification, WHY not in this case also?
      loop (m) = for i < chunk_in do
                    let ip1   = i + 1       in
                    let diff0 = ip1 - chunk_in in
                    let cond  = 0 < diff0   in
                    let diff  = if   cond
                                then diff0
                                else 0
                    in  m + diff
      in m
  in m
