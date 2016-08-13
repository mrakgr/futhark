-- Since I've all of a sudden figured out the continuation passing style
-- I'll do it from scratch and implement the parser and integrate it in the
-- Attoparsec library.

{-# LANGUAGE ScopedTypeVariables,BangPatterns #-}

import qualified Data.Attoparsec.Internal as I
import qualified Data.Attoparsec.Internal.Types as T
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Unboxed.Mutable as UMVec
import qualified Data.Vector.Generic.Mutable as GenericM
import qualified Data.Vector.Generic as Generic

import Control.Monad.ST
import Control.Monad.Primitive

type Parser = T.Parser

-- I got lost here somehow.

-- applies the unboxed parser repeatedly.
unfoldrUVec :: UMVec.Unbox a => Parser i a -> Parser i (UVec.Vector a)
unfoldrUVec parser = T.Parser $ \t pos more lose succ ->
    succ t pos more (result t pos more) where
        result t pos more = runST $ do
          empty <- UMVec.new 1024
          return $ loop 0 empty parser t pos more
        loop :: Int -> UMVec.STVector s a -> Parser i a -> t -> T.Pos -> T.More -> UVec.Vector a
        loop i arr parser = \t pos more ->
          T.runParser parser t pos more lose' succ' where
              succ' i arr t !pos more a = do
                (arr' :: UMVec.STVector s a) <- growIfFilled i arr
                UMVec.write arr' i a
                T.runParser parser t pos more (lose' arr) (succ' (i+1) arr)
                --loop (i+1) arr' parser t' pos' more' lose succ
              lose' arr t' _pos' more' _ctx _msg =
                UVec.freeze arr



-- helper
growIfFilled
      :: (GenericM.MVector v a, PrimMonad m) =>
         Int -> v (PrimState m) a -> m (v (PrimState m) a)
growIfFilled i arr = do
  let capacity = GenericM.length arr
  if i >= capacity
  then GenericM.grow arr capacity
  else return arr

main = print "Hello"
