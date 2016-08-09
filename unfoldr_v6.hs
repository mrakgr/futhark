{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Control.Arrow (first, second)
import Data.Char
import Control.Monad.ST
import qualified Data.Vector.Unboxed.Mutable as UMVec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Mutable as MVec
import qualified Data.Vector as Vec
import qualified Data.Text as T
import qualified Data.Text.Read as TR
import qualified Data.Text.IO as TIO
import System.TimeIt

unfoldrVec' :: forall a b. (b -> Either b (a, b)) -> b -> (Vec.Vector a,b)
unfoldrVec' cond init_state = runST $ do
    empty <- MVec.new 1024
    (arr, state) <- runUnfoldr 0 empty init_state
    arr <- Vec.unsafeFreeze arr
    return (arr, state)
    where
      growIfFilled :: Int -> MVec.STVector s a -> ST s (MVec.STVector s a)
      growIfFilled i arr =
        if i >= capacity
        then MVec.grow arr capacity
        else return arr
        where capacity = MVec.length arr
      runUnfoldr :: Int -> MVec.STVector s a -> b -> ST s (MVec.STVector s a, b)
      runUnfoldr i arr state =
        case cond state of
          Right (x, state) -> do
            arr <- growIfFilled i arr
            MVec.write arr i x
            runUnfoldr (i+1) arr state
          Left state ->
            return (MVec.slice 0 i arr, state)

toInt :: (T.Text, T.Text) -> Either T.Text (Int, T.Text)
toInt (int, text) =
  case TR.decimal int of
    Right (int, rest) ->
      if T.null rest then
        Right (int, text)
      else
        Left text
    Left rest ->
      Left text

readInt :: T.Text -> Either T.Text (Int, T.Text)
readInt = toInt . T.span isDigit . T.dropWhile isSpace

manyInts :: T.Text -> (Vec.Vector Int, T.Text)
manyInts = unfoldrVec' readInt

main :: IO ()
main = timeIt $ do
  text <- TIO.readFile "10M_integers.dat"
  print $ Vec.sum $ fst $ manyInts text
