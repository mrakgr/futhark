# Futhark Parser Attempts

After giving up on doing this two times, going crazy more than that, spitting on the language and telling myself I would never use it again...seeing that a guy managing to get it to run on Reddit just by reading from standard IO, I decided to just try ByteString to test if Text was the reason why it is so inefficient and it turns out that it is. All this pain and suffering could have been avoided by ditching Text at the start. It should have occurred to me to do this as well given that I got a hint already when I found out it had O(n) indexing.

There is something in Haskell that brings me on tilt much more easily than in other languages. It seems I am going to have to be very careful of the data structures used in the underlying libraries if I ever use it again...which I won't.

I am tsundere for it at this point.

I've also tested switching the Attoparsec parser to ByteString.Char8, but that did not help it. Here laziness is in fact the culprit.

But for performance now that I know where the hole is, that leaves the door open to integrating it with some library that does operate using continuation passing.

I'll do this at some later date just to bring this thing to a closure.
