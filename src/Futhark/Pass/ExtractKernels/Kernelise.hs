{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
-- | Sequentialise to kernel statements.
module Futhark.Pass.ExtractKernels.Kernelise
       ( transformBinding
       , transformBindings
       , transformBody

       , mapIsh
       )
       where

import Control.Applicative
import Control.Monad
import Data.Monoid

import Prelude

import qualified Futhark.Analysis.Alias as Alias
import qualified Futhark.Transform.FirstOrderTransform as FOT
import Futhark.Representation.SOACS
import qualified Futhark.Representation.Kernels as Out
import Futhark.MonadFreshNames
import Futhark.Tools

type Transformer m = (MonadFreshNames m,
                      LocalScope Out.Kernels m)

scopeForSOACs :: Scope Out.Kernels -> Scope SOACS
scopeForSOACs = castScope

transformBindings :: Transformer m => [Binding] -> m [Out.KernelStm Out.Kernels]
transformBindings [] = return []
transformBindings (bnd:bnds) = do
  bnd_stms <- transformBinding bnd
  inScopeOf bnd_stms $ (bnd_stms++) <$> transformBindings bnds

transformBinding :: Transformer m => Binding -> m [Out.KernelStm Out.Kernels]

transformBinding (Let pat _ (Op (Redomap cs w _ _ fold_lam nes arrs))) = do
  chunk_size <- newVName "chunk_size"
  chunk_offset <- newVName "chunk_offset"
  let arr_idents = drop (length nes) $ patternIdents pat
      (fold_acc_params, fold_elem_params) =
        splitAt (length nes) $ lambdaParams fold_lam
  arr_chunk_params <- mapM (mkArrChunkParam $ Var chunk_size) fold_elem_params

  map_arr_params <- forM arr_idents $ \arr ->
    newParam (baseString (identName arr) <> "_in") $
    setOuterSize (identType arr) (Var chunk_size)

  fold_acc_params' <- forM fold_acc_params $ \p ->
    newParam (baseString $ paramName p) $ paramType p

  let param_scope =
        scopeOfLParams $ fold_acc_params' ++ arr_chunk_params ++ map_arr_params

  redomap_pes <- forM (patternValueElements pat) $ \pe ->
    PatElem <$> newVName (baseString $ patElemName pe) <*>
    pure BindVar <*> pure (patElemType pe)

  redomap_kstms <- localScope param_scope $
    groupStreamMapAccumL redomap_pes cs (Var chunk_size) fold_lam
    (map (Var . paramName) fold_acc_params') (map paramName arr_chunk_params)

  let stream_kbody = Out.KernelBody redomap_kstms $
                     map (Var . patElemName) redomap_pes
      stream_lam = Out.GroupStreamLambda { Out.groupStreamChunkSize = chunk_size
                                         , Out.groupStreamChunkOffset = chunk_offset
                                         , Out.groupStreamAccParams = fold_acc_params'
                                         , Out.groupStreamArrParams = arr_chunk_params
                                         , Out.groupStreamLambdaBody = stream_kbody
                                         }
  return [Out.GroupStream (patternValueElements pat) w w stream_lam nes arrs]

  where mkArrChunkParam chunk_size arr_param =
          newParam (baseString (paramName arr_param) <> "_chunk") $
            arrayOfRow (paramType arr_param) chunk_size

transformBinding (Let pat _ (DoLoop [] val (ForLoop i bound) body)) = do
  dummy_chunk_size <- newVName "dummy_chunk_size"
  body' <- localScope (scopeOfFParams (map fst val)) $ transformBody body
  let lam = Out.GroupStreamLambda { Out.groupStreamChunkSize = dummy_chunk_size
                                  , Out.groupStreamChunkOffset = i
                                  , Out.groupStreamAccParams = map (fmap fromDecl . fst) val
                                  , Out.groupStreamArrParams = []
                                  , Out.groupStreamLambdaBody = body' }
  return [Out.GroupStream (patternValueElements pat)
          bound (constant (1::Int32)) lam (map snd val) []]

transformBinding (Let pat _ (If cond tb fb ts)) | all primType ts = do
  tb' <- transformBody tb
  fb' <- transformBody fb
  return [Out.GroupIf (patternValueElements pat) cond tb' fb']

transformBinding bnd =
  map (Out.Thread Out.ThreadsInSpace) <$> runBinder_ (FOT.transformBindingRecursively bnd)

transformBody :: Transformer m => Body -> m (Out.NestedKernelBody Out.Kernels)
transformBody (Body _ bnds res) = do
  stms <- transformBindings bnds
  return $ Out.KernelBody stms res

groupStreamMapAccumL :: Transformer m =>
                        [Out.PatElem]
                     -> Certificates
                     -> SubExp
                     -> Lambda
                     -> [SubExp]
                     -> [VName]
                     -> m [Out.KernelStm Out.Kernels]
groupStreamMapAccumL pes cs w fold_lam accexps arrexps = do
  let acc_num     = length accexps
      res_tps     = lambdaReturnType fold_lam
      map_arr_tps = drop acc_num res_tps

  soacs_scope <- asksScope scopeForSOACs

  let fold_lam' = fold_lam { lambdaParams = take acc_num $ lambdaParams fold_lam }
  ((merge, i, redomap_loop), bnds) <- flip runBinderT soacs_scope $ do
    maparrs <- resultArray [ arrayOf t (Shape [w]) NoUniqueness
                           | t <- map_arr_tps ]
    FOT.doLoopMapAccumL' cs w (Alias.analyseLambda fold_lam') accexps [] maparrs

  -- HACK: we manually inject the indexing here.
  dummy_chunk_size <- newVName "dummy_chunk_size"
  let arr_params = drop acc_num $ lambdaParams fold_lam
  arr_params_chunked <- forM arr_params $ \arr_param ->
    newParam (baseString (paramName arr_param) <> "_chunked") $
    paramType arr_param `arrayOfRow` Var dummy_chunk_size
  let index_bnds = do (p, arr) <- zip arr_params $ map paramName arr_params_chunked
                      return $ mkLet' [] [paramIdent p] $
                        PrimOp $ Index cs arr [constant (0::Int32)]

  redomap_kbody <- transformBody $ index_bnds `insertBindings` redomap_loop

  let acc_params = map (fmap fromDecl . fst) merge
      stream_lam = Out.GroupStreamLambda { Out.groupStreamChunkSize = dummy_chunk_size
                                         , Out.groupStreamChunkOffset = i
                                         , Out.groupStreamAccParams = acc_params
                                         , Out.groupStreamArrParams = arr_params_chunked
                                         , Out.groupStreamLambdaBody = redomap_kbody
                                         }
      stream_kstm = Out.GroupStream pes w (constant (1::Int32)) stream_lam accexps arrexps

  bnds_kstms <- transformBindings bnds

  return $ bnds_kstms ++ [stream_kstm]

resultArray :: MonadBinder m => [Type] -> m [VName]
resultArray = mapM oneArray
  where oneArray t = letExp "result" $ PrimOp $ Scratch (elemType t) (arrayDims t)

-- | Turn what is morally a map with a lambda containing kernel
-- statements into a sequential kernel statement loop.
mapIsh :: Transformer m =>
          Pattern
       -> Certificates
       -> SubExp
       -> [LParam]
       -> Out.NestedKernelBody Out.Kernels
       -> [VName]
       -> m [Out.KernelStm Out.Kernels]
mapIsh pat cs w params (Out.KernelBody kstms kres) arrs = do
  soacs_scope <- asksScope scopeForSOACs
  i <- newVName "i"

  (outarrs, bnds) <-
    flip runBinderT soacs_scope $ resultArray $ patternTypes pat

  outarr_params <- forM (patternElements pat) $ \pe ->
    newParam (baseString (patElemName pe) <> "_out") $
    patElemType pe

  dummy_chunk_size <- newVName "dummy_chunk_size"
  params_chunked <- forM params $ \param ->
    newParam (baseString (paramName param) <> "_chunked") $
    paramType param `arrayOfRow` Var dummy_chunk_size

  (outarr_params_new, write_elems) <-
    fmap unzip $ forM (zip outarr_params kres) $ \(outarr_param, se) -> do
      outarr_param_new <- newParam' (<>"_new") outarr_param
      return (outarr_param_new,
               Out.Thread Out.ThreadsInSpace $
               mkLet [] [(paramIdent outarr_param_new,
                          BindInPlace [] (paramName outarr_param) [Var i])] $
               PrimOp $ SubExp se)

  let index_stms = do (p, arr) <- zip params $ map paramName params_chunked
                      return $ Out.Thread Out.ThreadsInSpace $
                        mkLet' [] [paramIdent p] $
                        PrimOp $ Index cs arr [constant (0::Int32)]
      kbody' = Out.KernelBody (index_stms++kstms++write_elems) $
               map (Var . paramName) outarr_params_new

  let stream_lam = Out.GroupStreamLambda { Out.groupStreamChunkSize = dummy_chunk_size
                                         , Out.groupStreamChunkOffset = i
                                         , Out.groupStreamAccParams = outarr_params
                                         , Out.groupStreamArrParams = params_chunked
                                         , Out.groupStreamLambdaBody = kbody'
                                         }
      stream_kstm = Out.GroupStream (patternElements pat) w (constant (1::Int32))
                    stream_lam (map Var outarrs) arrs

  bnds_kstms <- transformBindings bnds
  return $ bnds_kstms ++ [stream_kstm]
