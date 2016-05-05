--------------------------------------------
--subroutine tridag(a,b,c,d,nn)
--
--    dimension a(nn),b(nn),c(nn),d(nn)
--
--    if(nn .eq. 1) then
--        d(1)=d(1)/b(1)
--        return
--    end if
--
--    do 10 k = 2,nn
--        xm = a(k)/b(k-1)
--        b(k) = b(k) - xm*c(k-1)
--        d(k) = d(k) - xm*d(k-1)
--10  continue
--
--    d(nn) = d(nn)/b(nn)
--    k = nn
--    do 20 i = 2,nn
--        k = nn + 1 - i
--        d(k) = (d(k) - c(k)*d(k+1))/b(k)
--20  continue
--    return
--end
--------------------------------------------/
-- ==
-- input {
-- }
-- output {
--   [1.000000, 0.335000, -13.003881, 4.696531, 2.284400, -1.201104, 23.773315,
--    6.997077, 5.064199, 3.832115]
--   [-3.039058, 6.578116, -1.103211, -6.002021, 8.084014, -3.267883, -0.332640,
--    2.726331, -1.618253, 1.472878]
-- }


fun ([f64],[f64]) tridag(  int  nn,
                            *[f64] b, *[f64] d,
                            [f64] a, [f64] c ) =
    if (nn == 1)
    --then ( b, zipWith(fn f64 (f64 x, f64 y) => x / y, d, b) )
    then (b, [d[0]/b[0]])
    else
        loop((b, d)) = for i < (nn-1) do
            let xm     = a[i+1] / b[i]    in
            let b[i+1] = b[i+1] - xm*c[i] in
            let d[i+1] = d[i+1] - xm*d[i] in
            (b, d)
        in
        let d[nn-1] = d[nn-1] / b[nn-1]   in

        loop(d)    = for i < (nn-1) do
            let k = nn - 2 - i                       in
            let d[k] = ( d[k] - c[k]*d[k+1] ) / b[k] in
            d
        in
            (b, d)


fun ([f64],[f64]) main() =
    let nn = reduce(+, 0, [1,2,3,4]) in
    let a = replicate(nn, 3.33) in
    let b = map(fn f64 (int x) => f64(x) + 1.0, iota(nn)) in
    let c = map(fn f64 (int x) => 1.11*f64(x) + 0.5, iota(nn)) in
    let d = map(fn f64 (int x) => 1.01*f64(x) + 0.25, iota(nn)) in
        tridag(nn, b, d, a, c)
