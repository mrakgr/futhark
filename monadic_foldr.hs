-- The additions were by michaelt_ from https://www.reddit.com/r/haskell/comments/4wrpvi/suggestion_extend_unfoldr_to_return_state/d6acwlp?st=irnmnaku&sh=7d746631
-- I am glad I asked

import qualified Data.Vector.Generic.Mutable as GenericM
import qualified Data.Vector.Generic as Generic
import qualified Data.Vector.Unboxed as Vec
import qualified Data.Vector.Unboxed.Mutable as MVec
import Control.Monad.ST
import Control.Monad.Primitive

-- for the unfoldM below using the fusion internals
import qualified Data.Vector.Fusion.Stream.Monadic as Stream
import qualified Data.Vector.Fusion.Bundle.Monadic as Bundle
import Data.Vector.Fusion.Bundle.Size
import Control.Monad.Trans.State.Strict


unfoldrVec :: Generic.Vector v a => (s -> Either r (a, s)) -> s -> (v a, r)
unfoldrVec step init_state = runST $ do
    empty <- GenericM.new 1024
    let loop i arr state =             -- nb. use of a tuple (i,arr) in a loop like this is a little suspicious
           case step state of          -- it might be prudent to add strictness to i in particular
             Right (x, state') -> do
               arr' <- growIfFilled i arr
               GenericM.write arr' i x
               loop (i+1) arr' state'
             Left r ->  return (GenericM.slice 0 i arr, r)
    (marr, r) <- loop 0 empty init_state
    arr <- Generic.freeze marr
    return (arr, r)

-- helper
growIfFilled
      :: (GenericM.MVector v a, PrimMonad m) =>
         Int -> v (PrimState m) a -> m (v (PrimState m) a)
growIfFilled i arr = do
  let capacity = GenericM.length arr
  if i >= capacity
  then GenericM.grow arr capacity
  else return arr


-- monadic versions fwiw
-- these parallel Pipes.unfoldr in using (s -> m (Either r (a,s)))
-- There, unfoldr next = id; here unfoldr next = return

unfoldrMutable
  :: (GenericM.MVector v a, PrimMonad m)
  => (s -> m (Either r (a, s)))
  -> s
  -> m (v (PrimState m) a, r)
unfoldrMutable step begin = do
  empty <- GenericM.new 1024
  let loop i arr state = do
         e <- step state
         case e of
           Right (x, state') -> do
              arr' <- growIfFilled i arr
              GenericM.write arr' i x
              loop (i+1) arr' state'
           Left r -> return (GenericM.slice 0 i arr, r)
  loop 0 empty begin

unfoldrM
  :: (PrimMonad m, Generic.Vector v a)
  => (s -> m (Either t (a, s)))
  -> s
  -> m (v a, t)
unfoldrM step begin = do
  (mvec, r) <- unfoldrMutable step begin
  vec <- Generic.freeze mvec
  return (vec,r)

-- test that unfoldrM next = return

next (vec,r) =
  case vec Generic.!? 0 of
    Nothing -> return (Left r)
    Just a -> return (Right (a, (Generic.unsafeTail vec, r)))

test vec ret = do
  (v,r) <- unfoldrM next (vec, ret)
  print $ vec == v
  print $ r == ret

t1 = test (Vec.fromList "California") 12

-- implementation using fusion internals. It won't itself fuse, but maybe `munstream` is faster than recursive `grow`
--  `StateT m ...` is a PrimMonad if m is, so this can be used to collect the final state as well
unfoldM :: (PrimMonad m, Generic.Vector v a)
        => (s -> m (Maybe (a, s)))
        -> s
        -> m (v a)
unfoldM step begin = help step begin Unknown >>= Generic.unsafeFreeze
   where
     help step begin size  =
      GenericM.munstream (Bundle.fromStream (Stream.unfoldrM step begin) size)

-- something like the desired function:
unfoldMs :: (PrimMonad m, Generic.Vector v a)
          => (s -> m (Maybe (a, s)))
          -> s
          -> m (v a, s)
unfoldMs step begin = runStateT (unfoldM (carry_state step) begin) begin
  where
  carry_state step s = StateT $ \_ -> do
    m <- step s
    return $! case m of
      Nothing      -> (Nothing, s)
      Just (a,s'') -> (Just (a,s''), s'')

main = t1
