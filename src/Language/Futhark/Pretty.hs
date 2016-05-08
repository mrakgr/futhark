{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
-- | Futhark prettyprinter.  This module defines 'Pretty' instances
-- for the AST defined in "Language.Futhark.Syntax" and re-exports
-- "Futhark.Util.Pretty" for convenience.
module Language.Futhark.Pretty
  ( module Futhark.Util.Pretty
  )
  where

import Data.Array
import Data.Monoid
import Data.Hashable
import Data.Word
import qualified Data.HashSet as HS

import Prelude

import Futhark.Util.Pretty

import Language.Futhark.Syntax
import Language.Futhark.Attributes

commastack :: [Doc] -> Doc
commastack = align . stack . punctuate comma

class AliasAnnotation f where
  aliasComment :: (Eq vn, Pretty vn, Hashable vn) => PatternBase f vn -> Doc -> Doc

instance AliasAnnotation NoInfo where
  aliasComment _ = id

instance AliasAnnotation Info where
  aliasComment pat d = case aliasComment' pat of
    []   -> d
    l:ls -> foldl (</>) l ls </> d
    where aliasComment' Wildcard{} = []
          aliasComment' (TuplePattern pats _) = concatMap aliasComment' pats
          aliasComment' (Id ident) =
            case clean . HS.toList . aliases $ unInfo $ identType ident of
              [] -> []
              als -> [oneline $
                      text "// " <> ppr ident <> text " aliases " <>
                      commasep (map ppr als)]
            where clean = filter (/= identName ident)
                  oneline s = text $ displayS (renderCompact s) ""

instance Pretty Value where
  ppr (PrimValue bv) = ppr bv
  ppr (TupValue vs)
    | any (not . primType . valueType) vs =
      parens $ commastack $ map ppr vs
    | otherwise =
      parens $ commasep $ map ppr vs
  ppr (ArrayValue a t)
    | [] <- elems a = text "empty" <> parens (ppr t)
    | Array{} <- t = brackets $ commastack $ map ppr $ elems a
    | otherwise     = brackets $ commasep $ map ppr $ elems a

instance Pretty PrimType where
  ppr (Unsigned Int8) = text "u8"
  ppr (Unsigned Int16) = text "u16"
  ppr (Unsigned Int32) = text "u32"
  ppr (Unsigned Int64) = text "u64"
  ppr (Signed t) = ppr t
  ppr (FloatType t) = ppr t
  ppr Bool = text "bool"

instance Pretty PrimValue where
  ppr (UnsignedValue (Int8Value v)) =
    text (show (fromIntegral v::Word8)) <> text "u8"
  ppr (UnsignedValue (Int16Value v)) =
    text (show (fromIntegral v::Word16)) <> text "u16"
  ppr (UnsignedValue (Int32Value v)) =
    text (show (fromIntegral v::Word32)) <> text "u32"
  ppr (UnsignedValue (Int64Value v)) =
    text (show (fromIntegral v::Word64)) <> text "u64"
  ppr (SignedValue v) = ppr v
  ppr (BoolValue b) = text $ show b
  ppr (FloatValue v) = ppr v

instance Pretty Uniqueness where
  ppr Unique    = star
  ppr Nonunique = empty

instance (Eq vn, Hashable vn, Pretty vn) =>
         Pretty (TupleArrayElemTypeBase ShapeDecl as vn) where
  ppr (PrimArrayElem bt _ u) = ppr u <> ppr bt
  ppr (ArrayArrayElem at)    = ppr at
  ppr (TupleArrayElem ts)    = parens $ commasep $ map ppr ts

instance (Eq vn, Hashable vn, Pretty vn) =>
         Pretty (TupleArrayElemTypeBase Rank as vn) where
  ppr (PrimArrayElem bt _ u) = ppr u <> ppr bt
  ppr (ArrayArrayElem at)    = ppr at
  ppr (TupleArrayElem ts)    = parens $ commasep $ map ppr ts

instance (Eq vn, Hashable vn, Pretty vn) =>
         Pretty (ArrayTypeBase ShapeDecl as vn) where
  ppr (PrimArray et (ShapeDecl ds) u _) =
    ppr u <> foldl f (ppr et) ds
    where f s AnyDim       = brackets s
          f s (NamedDim v) = brackets $ s <> comma <> ppr v
          f s (ConstDim n) = brackets $ s <> comma <> ppr n

  ppr (TupleArray et (ShapeDecl ds) u) =
    ppr u <> foldl f (parens $ commasep $ map ppr et) ds
    where f s AnyDim       = brackets s
          f s (NamedDim v) = brackets $ s <> comma <> ppr v
          f s (ConstDim n) = brackets $ s <> comma <> ppr n

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (ArrayTypeBase Rank as vn) where
  ppr (PrimArray et (Rank n) u _) =
    ppr u <> foldl (.) id (replicate n brackets) (ppr et)
  ppr (TupleArray ts (Rank n) u) =
    ppr u <> foldl (.) id (replicate n brackets)
    (parens $ commasep $ map ppr ts)

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (TypeBase ShapeDecl as vn) where
  ppr (Prim et) = ppr et
  ppr (Array at) = ppr at
  ppr (Tuple ts) = parens $ commasep $ map ppr ts

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (UserType vn) where
  ppr (UserPrim et _) = ppr et
  ppr (UserUnique t _) = text "*" <> ppr t
  ppr (UserArray at d _) = brackets (ppr at <> f d)
    where f AnyDim = mempty
          f (NamedDim v) = comma <+> ppr v
          f (ConstDim n) = comma <+> ppr n
  ppr (UserTuple ts _) = parens $ commasep $ map ppr ts
  ppr (UserTypeAlias name _) = ppr name

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (TypeBase Rank as vn) where
  ppr (Prim et) = ppr et
  ppr (Array at) = ppr at
  ppr (Tuple ts) = parens $ commasep $ map ppr ts

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (TypeDeclBase f vn) where
  ppr = ppr . declaredType

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (ParamBase f vn) where
  ppr = ppr . paramName

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (IdentBase f vn) where
  ppr = ppr . identName

instance Pretty UnOp where
  ppr Not = text "!"
  ppr Negate = text "-"
  ppr Complement = text "~"
  ppr Abs = text "abs "
  ppr Signum = text "signum "
  ppr (ToFloat t) = ppr t
  ppr (ToSigned t) = ppr (Signed t)
  ppr (ToUnsigned t) = ppr (Unsigned t)

instance Pretty BinOp where
  ppr Plus = text "+"
  ppr Minus = text "-"
  ppr Pow = text "**"
  ppr Times = text "*"
  ppr Divide = text "/"
  ppr Mod = text "%"
  ppr Quot = text "//"
  ppr Rem = text "%%"
  ppr ShiftR = text ">>"
  ppr ZShiftR = text ">>>"
  ppr ShiftL = text "<<"
  ppr Band = text "&"
  ppr Xor = text "^"
  ppr Bor = text "|"
  ppr LogAnd = text "&&"
  ppr LogOr = text "||"
  ppr Equal = text "=="
  ppr NotEqual = text "!="
  ppr Less = text "<"
  ppr Leq = text "<="
  ppr Greater = text ">="
  ppr Geq = text ">="

hasArrayLit :: ExpBase ty vn -> Bool
hasArrayLit ArrayLit{} = True
hasArrayLit (TupLit es2 _) = any hasArrayLit es2
hasArrayLit (Literal val _) = hasArrayVal val
hasArrayLit _ = False

hasArrayVal :: Value -> Bool
hasArrayVal ArrayValue{} = True
hasArrayVal (TupValue vs) = any hasArrayVal vs
hasArrayVal _ = False

instance (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty) => Pretty (ExpBase ty vn) where
  ppr = pprPrec (-1)
  pprPrec _ (Var v) = ppr v
  pprPrec _ (Literal v _) = ppr v
  pprPrec _ (TupLit es _)
    | any hasArrayLit es = parens $ commastack $ map ppr es
    | otherwise          = parens $ commasep $ map ppr es
  pprPrec _ (Empty (TypeDecl t _) _) =
    text "empty" <> parens (ppr t)
  pprPrec _ (ArrayLit es _ _) =
    brackets $ commasep $ map ppr es
  pprPrec p (BinOp bop x y _ _) = prettyBinOp p bop x y
  pprPrec _ (UnOp op e _) = ppr op <+> pprPrec 9 e
  pprPrec _ (If c t f _ _) = text "if" <+> ppr c </>
                             text "then" <+> align (ppr t) </>
                             text "else" <+> align (ppr f)
  pprPrec _ (Apply fname args _ _) = text (nameToString fname) <>
                                     apply (map (align . ppr . fst) args)
  pprPrec p (LetPat pat e body _) =
    aliasComment pat $ mparens $ align $
    text "let" <+> align (ppr pat) <+>
    (if linebreak
     then equals </> indent 2 (ppr e)
     else equals <+> align (ppr e)) <+> text "in" </>
    ppr body
    where mparens = if p == -1 then id else parens
          linebreak = case e of
                        Map{} -> True
                        Reduce{} -> True
                        Filter{} -> True
                        Scan{} -> True
                        DoLoop{} -> True
                        LetPat{} -> True
                        LetWith{} -> True
                        Literal ArrayValue{} _ -> False
                        If{} -> True
                        ArrayLit{} -> False
                        _ -> hasArrayLit e
  pprPrec _ (LetWith dest src idxs ve body _)
    | dest == src =
      text "let" <+> ppr dest <+> list (map ppr idxs) <+>
      equals <+> align (ppr ve) <+>
      text "in" </> ppr body
    | otherwise =
      text "let" <+> ppr dest <+> equals <+> ppr src <+>
      text "with" <+> brackets (commasep (map ppr idxs)) <+>
      text "<-" <+> align (ppr ve) <+>
      text "in" </> ppr body
  pprPrec _ (Index e idxs _) =
    pprPrec 9 e <> brackets (commasep (map ppr idxs))
  pprPrec _ (Iota e _) = text "iota" <> parens (ppr e)
  pprPrec _ (Size i e _) =
    text "size" <> apply [text $ show i, ppr e]
  pprPrec _ (Replicate ne ve _) =
    text "replicate" <> apply [ppr ne, align (ppr ve)]
  pprPrec _ (Reshape shape e _) =
    text "reshape" <> apply [apply (map ppr shape), ppr e]
  pprPrec _ (Rearrange perm e _) =
    text "rearrange" <> apply [apply (map ppr perm), ppr e]
  pprPrec _ (Transpose e _) =
    text "transpose" <> apply [ppr e]
  pprPrec _ (Map lam a _) = ppSOAC "map" [lam] [a]
  pprPrec _ (Reduce Commutative lam e a _) = ppSOAC "reduceComm" [lam] [e, a]
  pprPrec _ (Reduce Noncommutative lam e a _) = ppSOAC "reduce" [lam] [e, a]
  pprPrec _ (Stream form lam arr _) =
    case form of
      MapLike o ->
        let ord_str = if o == Disorder then "Per" else ""
        in  text ("streamMap"++ord_str) <>
            parens ( ppList [lam] </> commasep [ppr arr] )
      RedLike o comm lam0 acc ->
        let ord_str = if o == Disorder then "Per" else ""
            comm_str = case comm of Commutative -> "Comm"
                                    Noncommutative -> ""
        in  text ("streamRed"++ord_str++comm_str) <>
            parens ( ppList [lam0, lam] </> commasep [ppr acc, ppr arr] )
      Sequential acc ->
            text "streamSeq" <>
            parens ( ppList [lam] </> commasep [ppr acc, ppr arr] )
  pprPrec _ (Scan lam e a _) = ppSOAC "scan" [lam] [e, a]
  pprPrec _ (Filter lam a _) = ppSOAC "filter" [lam] [a]
  pprPrec _ (Partition lams a _) = ppSOAC "partition" lams [a]
  pprPrec _ (Zip es _) = text "zip" <> apply (map (ppr . fst) es)
  pprPrec _ (Unzip e _ _) = text "unzip" <> parens (ppr e)
  pprPrec _ (Unsafe e _) = text "unsafe" <+> pprPrec 9 e
  pprPrec _ (Split e a _) =
    text "split" <> apply [ppr e, ppr a]
  pprPrec _ (Concat x y _) =
    text "concat" <> apply [ppr x, ppr y]
  pprPrec _ (Copy e _) = text "copy" <> parens (ppr e)
  pprPrec _ (DoLoop pat initexp form loopbody letbody _) =
    aliasComment pat $
    text "loop" <+> parens (ppr pat <+> equals <+> ppr initexp) <+> equals <+>
    ppr form <+>
    text "do" </>
    indent 2 (ppr loopbody) <+> text "in" </>
    ppr letbody
  pprPrec _ (Write i v a _) = text "write" <> parens (commasep [ppr i, ppr v, ppr a])

instance (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty) => Pretty (LoopFormBase ty vn) where
  ppr (For FromUpTo lbound i ubound) =
    text "for" <+> align (ppr lbound) <+> ppr i <+> text "<" <+> align (ppr ubound)
  ppr (For FromDownTo lbound i ubound) =
    text "for" <+> align (ppr ubound) <+> ppr i <+> text ">" <+> align (ppr lbound)
  ppr (While cond) =
    text "while" <+> ppr cond

instance (Eq vn, Hashable vn, Pretty vn) => Pretty (PatternBase ty vn) where
  ppr (Id ident)     = ppr ident
  ppr (TuplePattern pats _) = parens $ commasep $ map ppr pats
  ppr (Wildcard _ _) = text "_"

instance (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty) => Pretty (LambdaBase ty vn) where
  ppr (CurryFun fname [] _ _) = text $ nameToString fname
  ppr (CurryFun fname curryargs _ _) =
    text (nameToString fname) <+> apply (map ppr curryargs)
  ppr (AnonymFun params body rettype _) =
    text "fn" <+> ppr rettype <+>
    apply (map ppParam params) <+>
    text "=>" </> indent 2 (ppr body)
  ppr (UnOpFun unop _ _ _) =
    ppr unop
  ppr (BinOpFun binop _ _ _ _) =
    ppr binop
  ppr (CurryBinOpLeft binop x _ _ _) =
    ppr x <+> ppr binop
  ppr (CurryBinOpRight binop x _ _ _) =
    ppr binop <+> ppr x

instance (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty) => Pretty (ProgBase ty vn) where
  ppr = stack . punctuate line . map ppDec . progFunctions
    where ppDec (FunDef entry name rettype args body _) =
            text fun <+> ppr rettype <+>
            text (nameToString name) <//>
            apply (map ppParam args) <+>
            equals </> indent 2 (ppr body)
            where fun | entry     = "entry"
                      | otherwise = "fun"

ppParam :: (Eq vn, Hashable vn, Pretty vn) => ParamBase t vn -> Doc
ppParam param = ppr (paramDeclaredType param) <+> ppr param

prettyBinOp :: (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty) =>
               Int -> BinOp -> ExpBase ty vn -> ExpBase ty vn -> Doc
prettyBinOp p bop x y = parensIf (p > precedence bop) $
                        pprPrec (precedence bop) x <+/>
                        ppr bop <+>
                        pprPrec (rprecedence bop) y
  where precedence LogAnd = 0
        precedence LogOr = 0
        precedence Band = 1
        precedence Bor = 1
        precedence Xor = 1
        precedence Equal = 2
        precedence NotEqual = 2
        precedence Less = 2
        precedence Leq = 2
        precedence Greater = 2
        precedence Geq = 2
        precedence ShiftL = 3
        precedence ShiftR = 3
        precedence ZShiftR = 3
        precedence Plus = 4
        precedence Minus = 4
        precedence Times = 5
        precedence Divide = 5
        precedence Mod = 5
        precedence Quot = 5
        precedence Rem = 5
        precedence Pow = 6
        rprecedence Minus = 10
        rprecedence Divide = 10
        rprecedence op = precedence op

ppSOAC :: (Eq vn, Hashable vn, Pretty vn, AliasAnnotation ty, Pretty fn) =>
          String -> [fn] -> [ExpBase ty vn] -> Doc
ppSOAC name funs es =
  text name <> parens (ppList funs </>
                       commasep (map ppr es))

ppList :: (Pretty a) => [a] -> Doc
ppList as = case map ppr as of
              []     -> empty
              a':as' -> foldl (</>) (a' <> comma) $ map (<> comma) as'
