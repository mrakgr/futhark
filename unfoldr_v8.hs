{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Control.Arrow (first, second)
import Control.Monad.ST
import Data.Char
import qualified Data.Vector.Unboxed.Mutable as MVec
import qualified Data.Vector.Unboxed as Vec
-- import qualified Data.Vector.Mutable as MVec
-- import qualified Data.Vector as Vec
--import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as B
import System.TimeIt


unfoldrVec' :: Vec.Unbox a => (b -> Either b (a, b)) -> b -> (Vec.Vector a,b)
unfoldrVec' cond init_state = runST $ do
    empty <- MVec.new 1024
    (arr, state) <- runUnfoldr 0 empty init_state
    -- It might be better to use freeze here instead of unsafeFreeze as
    -- the unsafe version will lead to a memory leak.
    arr <- Vec.freeze arr
    return (arr, state)
    where
      growIfFilled i arr =
        if i >= capacity
        then MVec.grow arr capacity
        else return arr
        where capacity = MVec.length arr
      runUnfoldr i arr state =
        case cond state of
          Right (x, state) -> do
            arr <- growIfFilled i arr
            MVec.write arr i x
            runUnfoldr (i+1) arr state
          Left state ->
            return (MVec.slice 0 i arr, state)

toInt :: (B.ByteString, B.ByteString) -> Either B.ByteString (Int, B.ByteString)
toInt (int, text) =
  case B.readInt int of
    Just(v,_) -> Right(v,text)
    Nothing -> Left text

readInt :: B.ByteString -> Either B.ByteString (Int, B.ByteString)
readInt = toInt . B.span isDigit . B.dropWhile isSpace

manyInts :: B.ByteString -> (Vec.Vector Int, B.ByteString)
manyInts = unfoldrVec' readInt

main :: IO ()
main = timeIt $ do
  text <- B.readFile "10M_integers.dat"
  print $ Vec.sum $ fst $ manyInts text
