module Main where
  import qualified Language.Futhark.Parser as P
  import qualified Data.Text as T
  import qualified Data.Text.IO as TIO

  main :: IO()
  main = do
    code <- TIO.getContents
    x <- P.parseFuthark "stdin" code
    print x
