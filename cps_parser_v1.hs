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

-- applies the unboxed parser repeatedly.
unfoldrUVec :: UMVec.Unbox a => Parser i a -> Parser i (UVec.Vector a)
unfoldrUVec parser = T.Parser $ \t pos more _lose succ ->
    let result = runST $ do
          empty <- UMVec.new 1024
          UVec.freeze empty
    in
        succ t pos more result
    -- return $ loop 0 empty parser
    --   where
    --     loop :: Int -> UMVec.STVector s a -> Parser i a -> Parser i (UVec.Vector a)
    --     loop i arr parser = T.Parser $ \t pos more _lose succ ->
    --       T.runParser parser t pos more lose' succ' where
    --           succ' t' !pos' more' a = do
    --             (arr' :: UMVec.STVector s a) <- growIfFilled i arr
    --             UMVec.write arr' i a
    --             loop (i+1) arr' parser t' pos' more' _lose succ
    --           lose' t' _pos' more' _ctx _msg = do
    --             arr <- UVec.freeze arr
    --             succ t' _pos' more' arr


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
