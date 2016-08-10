{-# LANGUAGE OverloadedStrings #-}

import Data.Attoparsec.Text
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Control.Monad.ST
import Control.Applicative
import Data.Word
import System.TimeIt
import Debug.Trace

parseManyNumbers :: Parser [Int]
parseManyNumbers = many (decimal <* skipSpace)

main :: IO ()
main =
  timeIt $ do
    --text <- TIO.readFile "10M_integers.dat"
    print $ sum <$> parseOnly parseManyNumbers "1 2 3 4"
