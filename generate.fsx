open System

let rng = Random()

let file =
    IO.File.Open("10M_integers.dat",IO.FileMode.Create)
    |> IO.StreamWriter

for i=1 to 10000000 do
    rng.Next(1000)
    |> file.WriteLine

file.Close()
