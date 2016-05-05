-- This loop nest was derived from an LU factorisation program, and
-- exposed a bug in a simplification rule.  It does not compute
-- anything interesting.
--
-- Specifically, the bug was in the detection of loop-invariant
-- variables - an array might be considered loop-invariant, even
-- though some of its existential parameters (specifically shape
-- arguments) are not considered loop-invariant (due to missing copy
-- propagation).
-- ==

fun ([[f64]], [[f64]]) main(*[[f64]] a, *[[f64]] u) =
  let n = size(0, a) in
  loop ((a,u)) =
    for k < n do
      let u[k,k] = a[k,k] in
      loop (a) = for i < n-k do
        a
      in (a,u)
    in
  (a,u)
