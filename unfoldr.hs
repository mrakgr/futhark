{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import Data.Text(Text)
import Data.Char
import qualified Data.Text as T
import qualified Data.Text.Read as TR
import qualified Data.Text.IO as TIO
import Data.Vector(Vector)
import qualified Data.Vector as V
import System.TimeIt
--import Debug.Trace
import Data.IORef
import System.IO.Unsafe

type State = (Int, Text)

readChar :: State -> (Char, State)
readChar (pos, text) =
  if pos >= T.length text then
    error $ "Error at pos " ++ show pos
  else
    (text `T.index` pos,(pos+1,text))

takeWhile :: (Char -> Bool) -> State -> (Text,State)
takeWhile f state =
    let mut = unsafePerformIO (newIORef state)
        {-# NOINLINE mut #-}
        unfolder state =
          let (c,state') = readChar state in
            case f c of
              True ->
                unsafePerformIO $ do
                  writeIORef mut state'
                  return $ Just(c,state')
              False -> Nothing
    in
      (T.unfoldr unfolder state, unsafePerformIO (readIORef mut))

toInt :: (Text,State) -> (Int,State)
toInt (x,state@(pos,_)) =
  let
    d =
      case TR.decimal x of
        Right (v,_) -> v
        Left _ -> error $ "Wrong parse at " ++ show pos
  in (d, state)

strip :: (a, State) -> State
strip = snd

readInt :: State -> (Int, State)
readInt = toInt . Main.takeWhile isDigit . strip . Main.takeWhile isSpace

readNInts :: Int -> State -> (Vector Int, State)
readNInts n state =
    let mut = unsafePerformIO (newIORef state)
        {-# NOINLINE mut #-}
        unfolder state =
          let (int,state') = readInt state in
                unsafePerformIO $ do
                  writeIORef mut state'
                  return $ Just(int,state')
    in
      (V.unfoldrN n unfolder state, unsafePerformIO (readIORef mut))

main :: IO ()
main =
  timeIt $ do
    text <- TIO.readFile "10M_integers.dat"
    print $ V.sum $ fst $ readNInts 10 (0,text)
