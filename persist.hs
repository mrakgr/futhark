{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import Data.Attoparsec.Text
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Vector as V
import qualified Data.Vector.Persistent as VP
import Control.Monad.ST
import Control.Applicative
import Data.Word
import System.TimeIt
import Debug.Trace

-- | A version of 'liftA2' that is strict in the result of its first
-- action.
liftA2' :: (Monad m) => (a -> b -> c) -> m a -> m b -> m c
liftA2' f a b =
    do
      !x <- a
      y <- b
      return $ (f x y)
{-# INLINE liftA2' #-}

manyVP :: Parser a -> Parser (VP.Vector a)
manyVP v = many_v
  where
    many_v = some_v <|> pure VP.empty
    some_v = liftA2' (flip VP.snoc) v many_v

parseManyNumbers :: Parser (VP.Vector Int)
parseManyNumbers = manyVP (decimal <* skipSpace)

main :: IO ()
main =
  timeIt $ do
    --text <- TIO.readFile "10M_integers.dat"
    print $ sum <$> parseOnly parseManyNumbers "1 2 3 4"
