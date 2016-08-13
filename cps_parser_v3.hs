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
unfoldrUVec parser = T.Parser $ \t pos more win_fin lose_fin -> runST $ do
      empty <- UMVec.new 1024
      loop empty t pos more win_fin lose_fin where
          loop arr t pos more win lose =
              T.runParser t pos more (win arr) lose where
                  win arr t !pos more a = loop arr t pos more win lose
                  lose t pos more _ _ = succ t pos more arr

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
