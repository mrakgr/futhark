-- Since I've all of a sudden figured out the continuation passing style
-- I'll do it from scratch and implement the parser and integrate it in the
-- Attoparsec library.

{-# LANGUAGE ScopedTypeVariables,BangPatterns,OverloadedStrings #-}

import Data.Attoparsec.Text
import qualified Data.Text.IO as TIO
import qualified Data.Attoparsec.Internal as I
import qualified Data.Attoparsec.Internal.Types as T
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Unboxed.Mutable as UMVec
import qualified Data.Vector as Vec
import qualified Data.Vector.Mutable as MVec
import qualified Data.Vector.Generic.Mutable as GenericM
import qualified Data.Vector.Generic as Generic
import qualified Data.Text as Text
import qualified System.IO.Unsafe as Unsafe
import System.TimeIt
import Debug.Trace

import Control.Monad.ST
import Control.Monad.Primitive


manyCPSVec :: Parser a -> Parser (Vec.Vector a)
manyCPSVec parser = T.Parser $ \t pos more lose_fin win_fin ->
      let arr = Unsafe.unsafePerformIO (MVec.new 1024) in
      loop 0 arr t pos more lose_fin win_fin where
          loop i (arr :: MVec.MVector RealWorld a) t pos more lose_fin win_fin =
              T.runParser parser t pos more lose win where
                  win t !pos more a =
                    Unsafe.unsafePerformIO $ do
                        arr' <- growIfFilled i arr
                        MVec.write arr' i a
                        return $ loop (i+1) arr' t pos more lose_fin win_fin
                  {-# NOINLINE win #-}
                  lose t pos more _ _ =
                    Unsafe.unsafePerformIO $ do
                      arr' <- Vec.freeze (MVec.slice 0 i arr)
                      return $ win_fin t pos more arr'
                  {-# NOINLINE lose #-}
{-# NOINLINE manyCPSVec #-} -- Inlining does not seem to affect performance so I left this in.

manyCPSUVec :: UVec.Unbox a => Parser a -> Parser (UVec.Vector a)
manyCPSUVec parser = T.Parser $ \t pos more lose_fin win_fin ->
      let arr = Unsafe.unsafePerformIO (UMVec.new 1024) in
      loop 0 arr t pos more lose_fin win_fin where
          loop !i !arr !t !pos !more !lose_fin !win_fin =
              T.runParser parser t pos more lose win where
                  win !t !pos !more !a =
                    Unsafe.unsafePerformIO $ do
                        arr' <- growIfFilled i arr
                        UMVec.write arr' i a
                        return $! loop (i+1) arr' t pos more lose_fin win_fin
                  {-# NOINLINE win #-}
                  lose !t !pos !more !_ !_ =
                    Unsafe.unsafePerformIO $ do
                      arr' <- UVec.freeze (UMVec.slice 0 i arr)
                      return $! win_fin t pos more arr'
                  {-# NOINLINE lose #-}
{-# NOINLINE manyCPSUVec #-} -- Inlining does not seem to affect performance so I left this in.

-- helper
growIfFilled
      :: (GenericM.MVector v a, PrimMonad m) =>
         Int -> v (PrimState m) a -> m (v (PrimState m) a)
growIfFilled i arr = do
  let capacity = GenericM.length arr
  if i >= capacity
  then GenericM.grow arr capacity
  else return arr

parseManyNumbers :: Parser (UVec.Vector Int)
parseManyNumbers = manyCPSUVec (decimal <* skipSpace)

main :: IO ()
main =
  timeIt $ do
    text <- TIO.readFile "10M_integers.dat"
    print $ UVec.sum <$> parseOnly parseManyNumbers text
