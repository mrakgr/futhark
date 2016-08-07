{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import Data.Text(Text)
import Data.Char
import qualified Data.List as L
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
takeWhile f state@(pos,text) =
    let unfolder state =
          let (c,state'@(pos',_)) = readChar state
          in
            case f c of
              True -> Just((c,pos'),state')
              False -> Nothing
        result = L.unfoldr unfolder state
        last_list =
          case map snd result of
            [] -> pos
            x -> last x
    in
      (T.pack $ map fst result, (last_list, text))

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
readNInts n state@(pos,text) =
    let unfolder state =
          let (int,state'@(pos',_)) = readInt state
          in Just((int,pos'),state')
        result = V.unfoldrN n unfolder state
        last_vec =
          if V.length result > 0 then
            snd $ V.last result
          else
            pos
    in
      (V.map fst result, (last_vec, text))

main :: IO ()
main =
  timeIt $ do
    text <- TIO.readFile "10M_integers.dat"
    print $ V.sum $ fst $ readNInts 100 (0,text)
