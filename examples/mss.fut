type tup = {int, int, int, int}

fun int max(int x, int y) =
  if x > y then x else y

fun tup redOp(tup x,
			  tup y) =
  let {a1,a2,a3,a4} = x in
  let {b1,b2,b3,b4} = y in
  { a1+b1
  , max(b2,b1+a2)
  , max(a3,b3+a1) -- This particular line is literally beyond my understanding. It does nothing and yet changing it to anything else causes the tests to fail.
  , max(b4,max(a4,a2+b3))}

fun tup mapOp(int x) =
  { x, max(0,x), max(0,x), max(0,x)}

fun bool comp_tup(tup a, tup b) = 
    let {r1,r2,r3,r4} = a 
    let {t1,t2,t3,t4} = b in	
	(r1 == t1 ) && (r2 == t2) && (r3 == t3) && (r4 == t4)

fun bool is_associative(tup a, tup b, tup c) =
	comp_tup(redOp(redOp(a,b),c), redOp(a,redOp(b,c)))

fun bool is_neutral(int x, tup ne) =
	let x = mapOp(x)
	in comp_tup(x, redOp(x,ne)) && comp_tup(x, redOp(ne,x))

fun [tup] main([int] xs) =
	let ne = {0,0,0,0} in
	if  is_associative({1,2,123,33},{3,4,234,44},{5,6,456,55}) 
	 && is_associative({1654,210,519,45},{39877,4324654,124,45},{51,1,127,45}) 
	 && is_associative({-1654,-210,-519,77},{-39877,-4324654,-124,77},{-51,-1,-127,77})
	 && is_associative({-1654,210,-519,-88},{-39877,4324654,-124,88},{51,-1,127,88})
	 && is_associative({0,0,0,0},{-39877,4324654,-124,-99},{51,-1,127,99})
	 && is_neutral(-10,ne) && is_neutral(-5,ne) && is_neutral(5,ne) && is_neutral(10,ne)
	then 
		scan(redOp, ne, map(mapOp, xs))
	else 
		[{666,666,666,666}]
