fun int fib(int n) =
  loop ({x, y} = {1,1}) = for i < n do
                            {y, x+y}
  in x

fun int main(int n) = fib(n)
