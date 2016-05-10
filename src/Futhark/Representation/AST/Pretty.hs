{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
-- | Futhark prettyprinter.  This module defines 'Pretty' instances
-- for the AST defined in "Futhark.Representation.AST.Syntax",
-- but also a number of convenience functions if you don't want to use
-- the interface from 'Pretty'.
module Futhark.Representation.AST.Pretty
  ( prettyTuple
  , pretty
  , PrettyLore (..)
  , ppCertificates
  , ppCertificates'
  , ppTuple'
  )
  where

import Data.Array (elems, listArray)
import Data.Monoid

import Futhark.Util.Pretty

import Futhark.Representation.AST.Syntax
import Futhark.Util

-- | The class of lores whose annotations can be prettyprinted.
class (Annotations lore,
       Pretty (RetType lore),
       Pretty (ParamT (FParamAttr lore)),
       Pretty (ParamT (LParamAttr lore)),
       Pretty (PatElemT (LetAttr lore)),
       Pretty (Op lore)) => PrettyLore lore where
  ppBindingLore :: Binding lore -> Maybe Doc
  ppBindingLore = const Nothing
  ppFunDefLore :: FunDef lore -> Maybe Doc
  ppFunDefLore = const Nothing
  ppLambdaLore :: Lambda lore -> Maybe Doc
  ppLambdaLore = const Nothing
  ppExpLore :: Exp lore -> Maybe Doc
  ppExpLore = const Nothing

commastack :: [Doc] -> Doc
commastack = align . stack . punctuate comma

instance Pretty NoUniqueness where
  ppr _ = mempty

instance Pretty Commutativity where
  ppr Commutative = text "commutative"
  ppr Noncommutative = text "noncommutative"

instance Pretty Value where
  ppr (PrimVal bv) = ppr bv
  ppr (ArrayVal a t _)
    | null $ elems a = text "empty" <> parens (ppr t)
  ppr (ArrayVal a t (_:rowshape@(_:_))) =
    brackets $ commastack
    [ ppr $ ArrayVal (listArray (0, rowsize-1) a') t rowshape
      | a' <- chunk rowsize $ elems a ]
    where rowsize = product rowshape
  ppr (ArrayVal a _ _) =
    brackets $ commasep $ map ppr $ elems a

instance Pretty Shape where
  ppr = brackets . commasep . map ppr . shapeDims

instance Pretty ExtDimSize where
  ppr (Free e) = ppr e
  ppr (Ext x)  = text "?" <> text (show x)

instance Pretty ExtShape where
  ppr = brackets . commasep . map ppr . extShapeDims

instance Pretty Space where
  ppr DefaultSpace = mempty
  ppr (Space s)    = text "@" <> text s

instance Pretty u => Pretty (TypeBase Shape u) where
  ppr (Prim et) = ppr et
  ppr (Array et (Shape ds) u) = ppr u <> foldr f (ppr et) ds
    where f e s = brackets $ s <> comma <> ppr e
  ppr (Mem s DefaultSpace) = text "mem" <> parens (ppr s)
  ppr (Mem s (Space sp)) = text "mem" <> parens (ppr s) <> text "@" <> text sp

instance Pretty u => Pretty (TypeBase ExtShape u) where
  ppr (Prim et) = ppr et
  ppr (Array et (ExtShape ds) u) = ppr u <> foldr f (ppr et) ds
    where f dim s = brackets $ s <> comma <> ppr dim
  ppr (Mem s DefaultSpace) = text "mem" <> parens (ppr s)
  ppr (Mem s (Space sp)) = text "mem" <> parens (ppr s) <> text "@" <> text sp

instance Pretty u => Pretty (TypeBase Rank u) where
  ppr (Prim et) = ppr et
  ppr (Array et (Rank n) u) = ppr u <> foldl f (ppr et) [1..n]
    where f s _ = brackets s
  ppr (Mem s DefaultSpace) = text "mem" <> parens (ppr s)
  ppr (Mem s (Space sp)) = text "mem" <> parens (ppr s) <> text "@" <> text sp

instance Pretty Ident where
  ppr ident = ppr (identType ident) <+> ppr (identName ident)

instance Pretty SubExp where
  ppr (Var v)      = ppr v
  ppr (Constant v) = ppr v

instance PrettyLore lore => Pretty (Body lore) where
  ppr (Body _ (bnd:bnds) res) =
    stack (map ppr (bnd:bnds)) </>
    text "in" <+> braces (commasep $ map ppr res)
  ppr (Body _ [] res) =
    braces (commasep $ map ppr res)

bindingAnnotation :: PrettyLore lore => Binding lore -> Doc -> Doc
bindingAnnotation bnd doc =
  case ppBindingLore bnd of
    Nothing    -> doc
    Just annot -> annot </> doc

instance Pretty (PatElemT attr) => Pretty (PatternT attr) where
  ppr pat = ppPattern (patternContextElements pat) (patternValueElements pat)

instance Pretty (PatElemT b) => Pretty (PatElemT (a,b)) where
  ppr = ppr . fmap snd

instance Pretty (PatElemT Type) where
  ppr (PatElem name BindVar t) =
    ppr t <+>
    ppr name

  ppr (PatElem name (BindInPlace cs src is) t) =
    ppCertificates cs <>
    parens (ppr t <+>
            ppr name <+>
            text "<-" <+>
            ppr src) <>
    brackets (commasep $ map ppr is)

instance Pretty (ParamT b) => Pretty (ParamT (a,b)) where
  ppr = ppr . fmap snd

instance Pretty (ParamT DeclType) where
  ppr (Param name t) =
    ppr t <+>
    ppr name

instance Pretty (ParamT Type) where
  ppr (Param name t) =
    ppr t <+>
    ppr name

instance PrettyLore lore => Pretty (Binding lore) where
  ppr bnd@(Let pat _ e) =
    bindingAnnotation bnd $ align $
    text "let" <+> align (ppr pat) <+>
    case (linebreak, ppExpLore e) of
      (True, Nothing) -> equals </>
                         indent 2 e'
      (_, Just annot) -> equals </>
                         indent 2 (annot </>
                                   e')
      (False, Nothing) -> equals <+> align e'
    where e' = ppr e
          linebreak = case e of
                        DoLoop{} -> True
                        Op{} -> True
                        If{} -> True
                        PrimOp ArrayLit{} -> False
                        _ -> False

instance PrettyLore lore => Pretty (PrimOp lore) where
  ppr (SubExp se) = ppr se
  ppr (ArrayLit [] rt) =
    text "empty" <> parens (ppr rt)
  ppr (ArrayLit es rt) =
    case rt of
      Array {} -> brackets $ commastack $ map ppr es
      _        -> brackets $ commasep   $ map ppr es
  ppr (BinOp bop x y) = ppr bop <> parens (ppr x <> comma <+> ppr y)
  ppr (CmpOp op x y) = ppr op <> parens (ppr x <> comma <+> ppr y)
  ppr (ConvOp conv x) =
    text (convOpFun conv) <+> ppr fromtype <+> ppr x <+> text "to" <+> ppr totype
    where (fromtype, totype) = convTypes conv
  ppr (UnOp Not e) = text "!" <+> pprPrec 9 e
  ppr (UnOp (Abs t) e) = taggedI "abs" t <+> pprPrec 9 e
  ppr (UnOp (FAbs t) e) = taggedF "fabs" t <+> pprPrec 9 e
  ppr (UnOp (SSignum t) e) = taggedI "ssignum" t <+> pprPrec 9 e
  ppr (UnOp (USignum t) e) = taggedI "usignum" t <+> pprPrec 9 e
  ppr (UnOp (Complement t) e) = taggedI "~" t <> pprPrec 9 e
  ppr (Index cs v idxs) =
    ppCertificates cs <> ppr v <>
    brackets (commasep (map ppr idxs))
  ppr (Iota e x s) = text "iota" <> apply [ppr e, ppr x, ppr s]
  ppr (Replicate ne ve) =
    text "replicate" <> apply [ppr ne, align (ppr ve)]
  ppr (Scratch t shape) =
    text "scratch" <> apply (ppr t : map ppr shape)
  ppr (Reshape cs shape e) =
    ppCertificates cs <> text "reshape" <> apply [apply (map ppr shape), ppr e]
  ppr (Rearrange cs perm e) =
    ppCertificates cs <> text "rearrange" <> apply [apply (map ppr perm), ppr e]
  ppr (Split cs sizeexps a) =
    ppCertificates cs <> text "split" <> apply [apply (map ppr sizeexps), ppr a]
  ppr (Concat cs x ys _) =
    ppCertificates cs <> text "concat" <> apply (ppr x : map ppr ys)
  ppr (Copy e) = text "copy" <> parens (ppr e)
  ppr (Assert e _) = text "assert" <> parens (ppr e)
  ppr (Partition cs n flags arrs) =
    ppCertificates' cs <>
    text "partition" <>
    parens (commasep $ [ ppr n, ppr flags ] ++ map ppr arrs)

instance PrettyLore lore => Pretty (Exp lore) where
  ppr (If c t f _) = text "if" <+> ppr c </>
                     text "then" <+> align (ppr t) </>
                     text "else" <+> align (ppr f)
  ppr (PrimOp op) = ppr op
  ppr (Apply fname args _) = text (nameToString fname) <>
                             apply (map (align . ppr . fst) args)
  ppr (Op op) = ppr op
  ppr (DoLoop ctx val form loopbody) =
    text "loop" <+> ppPattern ctxparams valparams <+>
    equals <+> ppTuple' (ctxinit++valinit) </>
    (case form of
      ForLoop i bound ->
        text "for" <+> ppr i <+> text "<" <+> align (ppr bound)
      WhileLoop cond ->
        text "while" <+> ppr cond
    ) <+> text "do" </>
    indent 2 (ppr loopbody)
    where (ctxparams, ctxinit) = unzip ctx
          (valparams, valinit) = unzip val

instance PrettyLore lore => Pretty (Lambda lore) where
  ppr lambda@(Lambda params body rettype) =
    maybe id (</>) (ppLambdaLore lambda) $
    text "fn" <+> ppTuple' rettype <+>
    parens (commasep (map ppr params)) <+>
    text "=>" </> indent 2 (ppr body)

instance PrettyLore lore => Pretty (ExtLambda lore) where
  ppr (ExtLambda params body rettype) =
    text "fn" <+> ppTuple' rettype <+>
    parens (commasep (map ppr params)) <+>
    text "=>" </> indent 2 (ppr body)

instance Pretty ExtRetType where
  ppr = ppTuple' . retTypeValues

instance PrettyLore lore => Pretty (FunDef lore) where
  ppr fundec@(FunDef entry name rettype args body) =
    maybe id (</>) (ppFunDefLore fundec) $
    text fun <+> ppr rettype <+>
    text (nameToString name) <//>
    apply (map ppr args) <+>
    equals </> indent 2 (ppr body)
    where fun | entry = "entry"
              | otherwise = "fun"

instance PrettyLore lore => Pretty (Prog lore) where
  ppr = stack . punctuate line . map ppr . progFunctions

instance Pretty BinOp where
  ppr (Add t) = taggedI "add" t
  ppr (FAdd t) = taggedF "fadd" t
  ppr (Sub t) = taggedI "sub" t
  ppr (FSub t) = taggedF "fsub" t
  ppr (Mul t) = taggedI "mul" t
  ppr (FMul t) = taggedF "fmul" t
  ppr (UDiv t) = taggedI "udiv" t
  ppr (UMod t) = taggedI "umod" t
  ppr (SDiv t) = taggedI "sdiv" t
  ppr (SMod t) = taggedI "smod" t
  ppr (SQuot t) = taggedI "squot" t
  ppr (SRem t) = taggedI "srem" t
  ppr (FDiv t) = taggedF "fdiv" t
  ppr (Shl t) = taggedI "shl" t
  ppr (LShr t) = taggedI "lshr" t
  ppr (AShr t) = taggedI "ashr" t
  ppr (And t) = taggedI "and" t
  ppr (Or t) = taggedI "or" t
  ppr (Xor t) = taggedI "xor" t
  ppr (Pow t) = taggedI "pow" t
  ppr (FPow t) = taggedF "fpow" t
  ppr LogAnd = text "logand"
  ppr LogOr = text "logor"

instance Pretty CmpOp where
  ppr (CmpEq t) = text "eq_" <> ppr t
  ppr (CmpUlt t) = taggedI "ult" t
  ppr (CmpUle t) = taggedI "ule" t
  ppr (CmpSlt t) = taggedI "slt" t
  ppr (CmpSle t) = taggedI "sle" t
  ppr (FCmpLt t) = taggedF "lt" t
  ppr (FCmpLe t) = taggedF "le" t

instance Pretty ConvOp where
  ppr op = convOp (convOpFun op) from to
    where (from, to) = convTypes op

convOpFun :: ConvOp -> String
convOpFun ZExt{} = "zext"
convOpFun SExt{} = "sext"
convOpFun FPConv{} = "fpconv"
convOpFun FPToUI{} = "fptoui"
convOpFun FPToSI{} = "fptosi"
convOpFun UIToFP{} = "uitofp"
convOpFun SIToFP{} = "sitofp"

taggedI :: String -> IntType -> Doc
taggedI s Int8 = text $ s ++ "8"
taggedI s Int16 = text $ s ++ "16"
taggedI s Int32 = text $ s ++ "32"
taggedI s Int64 = text $ s ++ "64"

taggedF :: String -> FloatType -> Doc
taggedF s Float32 = text $ s ++ "32"
taggedF s Float64 = text $ s ++ "64"

convOp :: (Pretty from, Pretty to) => String -> from -> to -> Doc
convOp s from to = text s <> text "_" <> ppr from <> text "_" <> ppr to

instance Pretty d => Pretty (DimChange d) where
  ppr (DimCoercion se) = text "~" <> ppr se
  ppr (DimNew      se) = ppr se

ppPattern :: (Pretty a, Pretty b) => [a] -> [b] -> Doc
ppPattern [] bs = braces $ commasep $ map ppr bs
ppPattern as bs = braces $ commasep (map ppr as) <> semi <+> commasep (map ppr bs)

ppTuple' :: Pretty a => [a] -> Doc
ppTuple' ets = braces $ commasep $ map ppr ets

ppCertificates :: Certificates -> Doc
ppCertificates [] = empty
ppCertificates cs = text "<" <> commasep (map ppr cs) <> text ">"

ppCertificates' :: Certificates -> Doc
ppCertificates' [] = empty
ppCertificates' cs = ppCertificates cs <> line
