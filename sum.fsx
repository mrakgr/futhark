open System

let stopwatch = System.Diagnostics.Stopwatch.StartNew()

let rng = Random()

let file =
    IO.File.Open("10M_integers.dat",IO.FileMode.Open)
    |> IO.StreamReader

let mutable ar = ResizeArray()
while (
        if file.EndOfStream = false then
            file.ReadLine()
            |> (ar.Add << int64)
            true
        else
            false
    ) do ()

printfn "Sum: %i" (Seq.sum ar)
printfn "Elapsed time: %A" stopwatch.Elapsed
