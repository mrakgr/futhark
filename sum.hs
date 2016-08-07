{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import Data.Attoparsec.Text
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import Control.Monad.ST
import Control.Applicative
import Data.Word
import System.TimeIt
import Debug.Trace

data DirtyVector a = DirtyVector
  { dirtyVec :: V.Vector a
  , dirtySize :: Int
  , dirtyCapacity :: Int
}

dirtyVectorAppend :: a -> DirtyVector a -> DirtyVector a
dirtyVectorAppend x vec =
  let sizeVec = traceShow ("capacity is " ++ show (dirtySize vec)) (dirtySize vec)
      capacityVec = dirtyCapacity vec
      (newVec,newSize,newCapacity) =
        traceShow
          "Main loop"
          (if sizeVec >= capacityVec then
            runST $ do
              vec <- V.unsafeThaw (dirtyVec vec)
              vec <- traceShow ("Growing to " ++ show (capacityVec*2+1)) (VM.grow vec (capacityVec*2+1))
              VM.write vec sizeVec x
              vec <- V.unsafeFreeze vec
              return (vec, capacityVec*2+1, sizeVec+1)
          else
            runST $ do
              vec <- V.unsafeThaw (dirtyVec vec)
              VM.write vec sizeVec x
              vec <- V.unsafeFreeze vec
              return (vec, capacityVec, sizeVec+1))
  in
      traceShow
        "WTF"
        DirtyVector newVec newSize newCapacity

deDirtyVector :: DirtyVector a -> V.Vector a
deDirtyVector v =
  let v' = dirtyVec v
  in
    V.generate (dirtySize v) (\i -> v' V.! i)

-- | A version of 'liftA2' that is strict in the result of its first
-- action.
liftA2' :: (Monad m) => (a -> b -> c) -> m a -> m b -> m c
liftA2' f a b =
  traceShow
    "Lifting"
    (
    do
      !x <- traceShow "1..." a
      y <- traceShow "2..." b
      return $ traceShow "3..." (f x y))
{-# INLINE liftA2' #-}

manyDirty :: Parser a -> Parser (V.Vector a)
manyDirty v = deDirtyVector <$> many_v
  where
    many_v =
      traceShow
        "Many"
        some_v <|> pure (DirtyVector V.empty 0 0)
    some_v =
      traceShow
        "Some"
        liftA2' dirtyVectorAppend v many_v

parseManyNumbers :: Parser (V.Vector Int)
parseManyNumbers =
    traceShow
      "Hello"
      manyDirty (decimal <* skipSpace)

main :: IO ()
main =
  timeIt $ do
    --text <- TIO.readFile "10M_integers.dat"
    print $ sum <$> parseOnly parseManyNumbers "1 2 3 4"
