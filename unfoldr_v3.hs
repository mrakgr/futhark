{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import Data.Text(Text)
import Data.Char
import Control.Monad.ST
import Data.STRef
import qualified Data.List as L
import qualified Data.Text as T
import qualified Data.Text.Read as TR
import qualified Data.Text.IO as TIO
import Data.Vector(Vector)
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as VM
import System.TimeIt
import Debug.Trace

import Control.Monad.Loops

type State = (Int, Text)

readChar :: State -> (Char, State)
readChar (pos, text) =
  if pos >= T.length text then
    error $ "Error at pos " ++ show pos
  else
    (text `T.index` pos,(pos+1,text))

takeWhile :: (Char -> Bool) -> State -> (Text,State)
takeWhile f state@(pos,text) = runST $ do
  vec <- VM.new 1
  vec_ref <- newSTRef vec
  (capacity_ref :: STRef s Int) <- newSTRef 1
  (size_ref :: STRef s Int) <- newSTRef 0
  pos_ref <- newSTRef pos
  char_state_ref <- newSTRef (readChar state)

  let cond = do
        char_state <- readSTRef char_state_ref
        return $ f (fst char_state)

  whileM_ cond $ do
    vec <- readSTRef vec_ref
    char_state <- readSTRef char_state_ref
    --traceShow (fst char_state) $
    modifySTRef' pos_ref (+1)
    size <- readSTRef size_ref
    capacity <- readSTRef capacity_ref
    if size < capacity then do
      VM.write vec size (fst char_state)
      modifySTRef' size_ref (+1)
    else do
      vec <- VM.grow vec capacity
      writeSTRef vec_ref vec
      modifySTRef' capacity_ref (*2)
      VM.write vec size (fst char_state)
      modifySTRef' size_ref (+1)
    writeSTRef char_state_ref (readChar $ snd char_state)
  size <- readSTRef size_ref
  pos <- readSTRef pos_ref
  vec <- readSTRef vec_ref
  vec <- V.unsafeFreeze vec
  let txt = T.pack $ V.toList $ V.generate size (\i -> vec V.! i)
  return (txt, (pos,text))


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
readNInts n state@(pos,text) = runST $ do
  vec <- VM.new 1
  vec_ref <- newSTRef vec
  (capacity_ref :: STRef s Int) <- newSTRef 1
  (size_ref :: STRef s Int) <- newSTRef 0
  pos_ref <- newSTRef pos
  int_state_ref <- newSTRef (readInt state)
  i <- newSTRef 0

  let cond = do
        i' <- readSTRef i
        return $ i' < n

  whileM_ cond $ do
    modifySTRef' i (+1)
    vec <- readSTRef vec_ref
    int_state <- readSTRef int_state_ref
    --traceShow int_state $
    modifySTRef' pos_ref (+1)
    size <- readSTRef size_ref
    capacity <- readSTRef capacity_ref
    if size < capacity then do
      VM.write vec size (fst int_state)
      modifySTRef' size_ref (+1)
    else do
      vec <- VM.grow vec capacity
      writeSTRef vec_ref vec
      modifySTRef' capacity_ref (*2)
      VM.write vec size (fst int_state)
      modifySTRef' size_ref (+1)
    writeSTRef int_state_ref (readInt $ snd int_state)
  size <- readSTRef size_ref
  pos <- readSTRef pos_ref
  vec <- readSTRef vec_ref
  vec <- V.unsafeFreeze vec
  return (V.generate size (\i -> vec V.! i), (pos,text))

main :: IO ()
main =
  timeIt $ do
    !text <- TIO.readFile "10M_integers.dat"
    print $ V.sum $ fst $ readNInts 30 (0,text)
