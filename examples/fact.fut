fun int fact(int n) = reduce(*, 1, map(1+, iota(n)))

fun int main(int n) = fact(n)