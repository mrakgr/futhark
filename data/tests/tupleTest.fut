-- Test various abuse of tuples - specifically, the flattening done by
-- internalisation.
-- ==
-- input {
-- }
-- output {
--   8
--   11
-- }

fun (int,int) f((int,int) x) = x

fun (int,int) main() =
    let x = 1 + 2        in
    let y = (x + 5, 4+7) in
    let (z, (t,q)) = (x, y) in
        f(y)
