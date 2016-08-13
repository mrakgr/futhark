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
import qualified Data.Text as Text

import Control.Monad.ST
import Control.Monad.Primitive

type Parser = T.Parser

manyCPS :: Parser Text.Text Char -> Parser Text.Text [Char]
manyCPS parser = T.Parser $ \t pos more lose_fin win_fin ->
      loop [] t pos more lose_fin win_fin where
          loop (arr :: [Char]) t pos more lose_fin win_fin =
              T.runParser parser t pos more lose win where
                  win t !pos more a = loop (a:arr) t pos more lose_fin win_fin
                  lose t pos more _ _ = win_fin t pos more arr

main = print "Hello"
