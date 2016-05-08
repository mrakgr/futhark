{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances, FlexibleContexts, StandaloneDeriving #-}
-- | This Is an ever-changing abstract syntax for Futhark.  Some types,
-- such as @Exp@, are parametrised by type and name representation.
-- See the @docs/@ subdirectory in the Futhark repository for a language
-- reference, or this module may be a little hard to understand.
module Language.Futhark.Syntax
  (
   module Language.Futhark.Core

  -- * Types
  , Uniqueness(..)
  , IntType(..)
  , FloatType(..)
  , PrimType(..)
  , ArrayShape (..)
  , DimDecl (..)
  , ShapeDecl (..)
  , Rank (..)
  , TypeBase(..)
  , UserType(..)
  , TupleArrayElemTypeBase(..)
  , ArrayTypeBase(..)
  , CompTypeBase
  , StructTypeBase
  , DeclArrayTypeBase
  , DeclTupleArrayElemTypeBase
  , Diet(..)
  , TypeDeclBase (..)

    -- * Values
  , IntValue(..)
  , FloatValue(..)
  , PrimValue(..)
  , Value(..)

  -- * Abstract syntax tree
  , UnOp (..)
  , BinOp (..)
  , IdentBase (..)
  , ParamBase (..)
  , ExpBase(..)
  , LoopFormBase (..)
  , ForLoopDirection (..)
  , LambdaBase(..)
  , PatternBase(..)
  , StreamForm(..)

  -- * Definitions
  , FunDefBase(..)
  , TypeDefBase(..)
  , ProgBase(..)
  , ProgBaseWithHeaders(..)
  , ProgHeader(..)
  , DecBase(..)

  -- * Miscellaneous
  , NoInfo(..)
  , Info(..)
  , Names
  )
  where

import Control.Applicative
import Data.Array
import Data.Hashable
import Data.Loc
import Data.Functor
import Data.Monoid
import Data.Foldable
import Data.Traversable
import qualified Data.HashSet as HS

import Prelude

import Futhark.Representation.Primitive
  (IntType(..), FloatType(..), IntValue(..), FloatValue(..))
import Language.Futhark.Core

-- | Convenience class for deriving Show instances for the AST.
class (Show vn,
       Show (f vn),
       Show (f (CompTypeBase vn)),
       Show (f (StructTypeBase vn))) => Showable f vn where

-- | No information.  Usually used for placeholder type- or aliasing
-- information.
data NoInfo a = NoInfo
              deriving (Eq, Ord, Show)

instance Show vn => Showable NoInfo vn where
instance Functor NoInfo where
  fmap _ NoInfo = NoInfo
instance Foldable NoInfo where
  foldr _ b NoInfo = b
instance Traversable NoInfo where
  traverse _ NoInfo = pure NoInfo
instance Monoid (NoInfo a) where
  mempty = NoInfo
  _ `mappend` _ = NoInfo

-- | Some information.  The dual to 'NoInfo'
newtype Info a = Info { unInfo :: a }
            deriving (Eq, Ord, Show)

instance Show vn => Showable Info vn where
instance Functor Info where
  fmap f (Info x) = Info $ f x
instance Foldable Info where
  foldr f b (Info x) = f x b
instance Traversable Info where
  traverse f (Info x) = Info <$> f x

-- | Low-level primitive types.
data PrimType = Signed IntType
              | Unsigned IntType
              | FloatType FloatType
              | Bool
              deriving (Eq, Ord, Show)

-- | Non-array values.
data PrimValue = SignedValue !IntValue
               | UnsignedValue !IntValue
               | FloatValue !FloatValue
               | BoolValue !Bool
               deriving (Eq, Ord, Show)

-- | The class of types that can represent an array size.  The
-- 'Monoid' instance must define 'mappend' such that @dims1 `mappend`
-- dims2@ adds @dims1@ as the outer dimensions of @dims2@.
class (Eq shape, Ord shape, Monoid shape) => ArrayShape shape where
  -- | Number of dimensions.
  shapeRank :: shape -> Int
  -- | @stripDims n shape@ strips the outer @n@ dimensions from
  -- @shape@, returning 'Nothing' if this would result in zero or
  -- fewer dimensions.
  stripDims :: Int -> shape -> Maybe shape

-- | Declaration of a dimension size.
data DimDecl vn = NamedDim vn
                  -- ^ The size of the dimension is this name.  In a
                  -- function parameter, this is in a binding
                  -- position.  In a return type, this will give rise
                  -- to an assertion.
                | ConstDim Int
                  -- ^ The size is a constant.
                | AnyDim
                  -- ^ No dimension declaration.
                deriving (Eq, Ord, Show)

-- | The size of an array type is a list of its dimension sizes.  If
-- 'Nothing', that dimension is of a (statically) unknown size.
newtype ShapeDecl vn = ShapeDecl { shapeDims :: [DimDecl vn] }
                     deriving (Eq, Ord, Show)

-- | The rank of an array as a positive natural number.
newtype Rank vn = Rank Int
                deriving (Eq, Ord, Show)

instance Monoid (Rank vn) where
  mempty = Rank 0
  Rank n `mappend` Rank m = Rank $ n + m

instance ArrayShape (Rank vn) where
  shapeRank (Rank n) = n
  stripDims i (Rank n) | i < n     = Just $ Rank $ n - i
                       | otherwise = Nothing

instance Monoid (ShapeDecl vn) where
  mempty = ShapeDecl []
  ShapeDecl l1 `mappend` ShapeDecl l2 = ShapeDecl $ l1 ++ l2

instance (Eq vn, Ord vn) => ArrayShape (ShapeDecl vn) where
  shapeRank (ShapeDecl l) = length l
  stripDims i (ShapeDecl l)
    | i < length l = Just $ ShapeDecl $ drop i l
    | otherwise    = Nothing

-- | Types that can be elements of tuple-arrays.
data TupleArrayElemTypeBase shape as vn =
    PrimArrayElem PrimType (as vn) Uniqueness
  | ArrayArrayElem (ArrayTypeBase shape as vn)
  | TupleArrayElem [TupleArrayElemTypeBase shape as vn]
  deriving (Show)

instance Eq (shape vn) =>
         Eq (TupleArrayElemTypeBase shape as vn) where
  PrimArrayElem bt1 _ u1 == PrimArrayElem bt2 _ u2 = bt1 == bt2 && u1 == u2
  ArrayArrayElem at1     == ArrayArrayElem at2     = at1 == at2
  TupleArrayElem ts1     == TupleArrayElem ts2     = ts1 == ts2
  _                      == _                      = False

-- | An array type.
data ArrayTypeBase shape as vn =
    PrimArray PrimType (shape vn) Uniqueness (as vn)
    -- ^ An array whose elements are primitive types.
  | TupleArray [TupleArrayElemTypeBase shape as vn] (shape vn) Uniqueness
    -- ^ An array whose elements are tuples.
    deriving (Show)

instance Eq (shape vn) =>
         Eq (ArrayTypeBase shape as vn) where
  PrimArray et1 dims1 u1 _ == PrimArray et2 dims2 u2 _ =
    et1 == et2 && dims1 == dims2 && u1 == u2
  TupleArray ts1 dims1 u1 == TupleArray ts2 dims2 u2 =
    ts1 == ts2 && dims1 == dims2 && u1 == u2
  _ == _ =
    False

-- | An Futhark type is either an array, a prim type, or a tuple.
-- When comparing types for equality with '==', aliases are ignored,
-- but dimensions much match.
data TypeBase shape as vn = Prim PrimType
                          | Array (ArrayTypeBase shape as vn)
                          | Tuple [TypeBase shape as vn]
                          deriving (Eq, Show)


-- | A type with aliasing information and no shape annotations, used
-- for describing the type of a computation.
type CompTypeBase = TypeBase Rank Names

-- | An unstructured type with type variables and possibly shape
-- declarations - this is what the user types in the source program.
data UserType vn = UserPrim PrimType SrcLoc
                 | UserArray (UserType vn) (DimDecl vn) SrcLoc
                 | UserTuple [UserType vn] SrcLoc
                 | UserTypeAlias Name SrcLoc
                 | UserUnique (UserType vn) SrcLoc
    deriving (Show)

instance Located (UserType vn) where
  locOf (UserPrim _ loc) = locOf loc
  locOf (UserArray _ _ loc) = locOf loc
  locOf (UserTuple _ loc) = locOf loc
  locOf (UserTypeAlias _ loc) = locOf loc
  locOf (UserUnique _ loc) = locOf loc

--
-- | A "structural" type with shape annotations and no aliasing
-- information, used for declarations.

type StructTypeBase = TypeBase ShapeDecl NoInfo


-- | An array type with shape annotations and no aliasing information,
-- used for declarations.
type DeclArrayTypeBase = ArrayTypeBase ShapeDecl NoInfo

-- | A tuple array element type with shape annotations and no aliasing
-- information, used for declarations.
type DeclTupleArrayElemTypeBase = TupleArrayElemTypeBase ShapeDecl NoInfo

-- | A declaration of the type of something.
data TypeDeclBase f vn =
  TypeDecl { declaredType :: UserType vn
                             -- ^ The type declared by the user.
           , expandedType :: f (StructTypeBase vn)
                             -- ^ The type deduced by the type checker.
           }
deriving instance Showable f vn => Show (TypeDeclBase f vn)

-- | Information about which parts of a value/type are consumed.  For
-- example, we might say that a function taking an argument of type
-- @([int], *[int], [int])@ has diet @ConsumeTuple [Observe, Consume,
-- Observe]@.
data Diet = TupleDiet [Diet] -- ^ Consumes these parts of the tuple.
          | Consume -- ^ Consumes this value.
          | Observe -- ^ Only observes value in this position, does
                    -- not consume.
            deriving (Eq, Ord, Show)

-- | Every possible value in Futhark.  Values are fully evaluated and their
-- type is always unambiguous.
data Value = PrimValue !PrimValue
           | TupValue ![Value]
           | ArrayValue !(Array Int Value) (TypeBase Rank NoInfo ())
             -- ^ It is assumed that the array is 0-indexed.  The type
             -- is the row type.
             deriving (Eq, Show)

-- | An identifier consists of its name and the type of the value
-- bound to the identifier.
data IdentBase f vn = Ident { identName :: vn
                            , identType :: f (CompTypeBase vn)
                            , identSrcLoc :: SrcLoc
                            }
deriving instance Showable f vn => Show (IdentBase f vn)

instance Eq vn => Eq (IdentBase ty vn) where
  x == y = identName x == identName y

instance Located (IdentBase ty vn) where
  locOf = locOf . identSrcLoc

instance Hashable vn => Hashable (IdentBase ty vn) where
  hashWithSalt salt = hashWithSalt salt . identName


-- | A name with no aliasing information, but known type.  These are
-- used for function parameters.
data ParamBase f vn = Param { paramName :: vn
                            , paramTypeDecl :: TypeDeclBase f vn
                            , paramSrcLoc :: SrcLoc
                          }
deriving instance Showable f vn => Show (ParamBase f vn)

instance Eq vn => Eq (ParamBase f vn) where
  x == y = paramName x == paramName y

instance Located (ParamBase f vn) where
  locOf = locOf . paramSrcLoc

instance Hashable vn => Hashable (ParamBase f vn) where
  hashWithSalt salt = hashWithSalt salt . paramName

-- | Unary operators.
data UnOp = Not
          | Negate
          | Complement
          | Abs
          | Signum
          | ToFloat FloatType
          | ToSigned IntType
          | ToUnsigned IntType
          deriving (Eq, Ord, Show)

-- | Binary operators.
data BinOp = Plus -- Binary Ops for Numbers
           | Minus
           | Pow
           | Times
           | Divide
           | Mod
           | Quot
           | Rem
           | ShiftR
           | ZShiftR -- ^ Zero-extend right shift.
           | ShiftL
           | Band
           | Xor
           | Bor
           | LogAnd
           | LogOr
           -- Relational Ops for all primitive types at least
           | Equal
           | NotEqual
           | Less
           | Leq
           | Greater
           | Geq
             deriving (Eq, Ord, Show, Enum, Bounded)

-- | The Futhark expression language.
--
-- In a value of type @Exp tt vn@, all 'Type' values are kept as @tt@
-- values, and all (variable) names are of type @vn@.
--
-- This allows us to encode whether or not the expression has been
-- type-checked in the Haskell type of the expression.  Specifically,
-- the parser will produce expressions of type @Exp 'NoInfo'@, and the
-- type checker will convert these to @Exp 'Type'@, in which type
-- information is always present.
data ExpBase f vn =
              Literal Value SrcLoc

            | TupLit    [ExpBase f vn] SrcLoc
            -- ^ Tuple literals, e.g., @{1+3, {x, y+z}}@.

            | ArrayLit  [ExpBase f vn] (f (CompTypeBase vn)) SrcLoc

            | Empty (TypeDeclBase f vn) SrcLoc

            | Var    (IdentBase f vn)
            -- ^ Array literals, e.g., @[ [1+x, 3], [2, 1+4] ]@.
            -- Second arg is the type of of the rows of the array (not
            -- the element type).
            | LetPat (PatternBase f vn) (ExpBase f vn) (ExpBase f vn) SrcLoc

            | If     (ExpBase f vn) (ExpBase f vn) (ExpBase f vn) (f (CompTypeBase vn)) SrcLoc

            | Apply  Name [(ExpBase f vn, Diet)] (f (CompTypeBase vn)) SrcLoc

            | DoLoop
              (PatternBase f vn) -- Merge variable pattern
              (ExpBase f vn) -- Initial values of merge variables.
              (LoopFormBase f vn) -- Do or while loop.
              (ExpBase f vn) -- Loop body.
              (ExpBase f vn) -- Let-body.
              SrcLoc

            | BinOp BinOp (ExpBase f vn) (ExpBase f vn) (f (CompTypeBase vn)) SrcLoc
            | UnOp UnOp (ExpBase f vn) SrcLoc

            -- Primitive array operations
            | LetWith (IdentBase f vn) (IdentBase f vn)
                      [ExpBase f vn] (ExpBase f vn)
                      (ExpBase f vn) SrcLoc

            | Index (ExpBase f vn)
                    [ExpBase f vn]
                    SrcLoc

            | Size Int (ExpBase f vn) SrcLoc
            -- ^ The size of the specified array dimension.

            | Split [ExpBase f vn] (ExpBase f vn) SrcLoc
            -- ^ @split( (1,1,3), [ 1, 2, 3, 4 ]) = {[1], [], [2, 3], [4]}@.
            -- Note that this is different from the internal representation

            | Concat (ExpBase f vn) [ExpBase f vn] SrcLoc
            -- ^ @concat([1],[2, 3, 4]) = [1, 2, 3, 4]@.

            | Copy (ExpBase f vn) SrcLoc
            -- ^ Copy the value return by the expression.  This only
            -- makes a difference in do-loops with merge variables.

            -- Array construction.
            | Iota (ExpBase f vn) SrcLoc
            -- ^ @iota(n) = [0,1,..,n-1]@
            | Replicate (ExpBase f vn) (ExpBase f vn) SrcLoc
            -- ^ @replicate(3,1) = [1, 1, 1]@

            -- Array index space transformation.
            | Reshape [ExpBase f vn] (ExpBase f vn) SrcLoc
             -- ^ 1st arg is the new shape, 2nd arg is the input array.

            | Transpose (ExpBase f vn) SrcLoc
            -- ^ Transpose two-dimensional array.  @transpose(a) =
            -- rearrange((1,0), a)@.

            | Rearrange [Int] (ExpBase f vn) SrcLoc
            -- ^ Permute the dimensions of the input array.  The list
            -- of integers is a list of dimensions (0-indexed), which
            -- must be a permutation of @[0,n-1]@, where @n@ is the
            -- number of dimensions in the input array.

            -- Second-Order Array Combinators accept curried and
            -- anonymous functions as first params.
            | Map (LambdaBase f vn) (ExpBase f vn) SrcLoc
             -- ^ @map(op +(1), [1,2,..,n]) = [2,3,..,n+1]@.

            | Reduce Commutativity (LambdaBase f vn) (ExpBase f vn) (ExpBase f vn) SrcLoc
             -- ^ @reduce(op +, 0, [1,2,...,n]) = (0+1+2+...+n)@.

            | Scan (LambdaBase f vn) (ExpBase f vn) (ExpBase f vn) SrcLoc
             -- ^ @scan(plus, 0, [ 1, 2, 3 ]) = [ 1, 3, 6 ]@.

            | Filter (LambdaBase f vn) (ExpBase f vn) SrcLoc
            -- ^ Return those elements of the array that satisfy the
            -- predicate.

            | Partition [LambdaBase f vn] (ExpBase f vn) SrcLoc
            -- ^ @partition(f_1, ..., f_n, a)@ returns @n+1@ arrays, with
            -- the @i@th array consisting of those elements for which
            -- function @f_1@ returns 'True', and no previous function
            -- has returned 'True'.  The @n+1@th array contains those
            -- elements for which no function returns 'True'.

            | Stream (StreamForm f vn) (LambdaBase f vn) (ExpBase f vn) SrcLoc
            -- ^ Streaming: intuitively, this gives a size-parameterized
            -- composition for SOACs that cannot be fused, e.g., due to scan.
            -- For example, assuming @A : [int], f : int->int, g : real->real@,
            -- the code: @let x = map(f,A) in let y = scan(op+,0,x) in map(g,y)@
            -- can be re-written (streamed) in the source-Futhark language as:
            -- @let {acc, z} =
            -- @stream( 0, A,@
            -- @      , fn {int,[real]} (real chunk, real acc, [int] a) =>@
            -- @            let x = map (f,         A ) in@
            -- @            let y0= scan(op +, 0,   x ) in@
            -- @            let y = map (op +(acc), y0) in@
            -- @            { acc+y0[chunk-1], map(g, y) }@
            -- @      )@
            -- where (i)  @chunk@ is a symbolic int denoting the chunk
            -- size, (ii) @0@ is the initial value of the accumulator,
            -- which allows the streaming of @scan@.
            -- Finally, the unnamed function (@fn...@) implements the a fold that:
            -- computes the accumulator of @scan@, as defined inside its body, AND
            -- implicitly concatenates each of the result arrays across
            -- the iteration space.
            -- In essence, sequential codegen can choose chunk = 1 and thus
            -- eliminate the SOACs on the outermost level, while parallel codegen
            -- may choose the maximal chunk size that still satisfies the memory
            -- requirements of the device.

            | Write (ExpBase f vn) (ExpBase f vn) (ExpBase f vn) SrcLoc
            -- ^ @write([0, 2, -1], [9, 7, 0], [3, 4, 5]) = [9, 4, 7]@.

            | Zip [(ExpBase f vn, f (CompTypeBase vn))] SrcLoc
            -- ^ Normal zip supporting variable number of arguments.
            -- The type paired to each expression is the full type of
            -- the array returned by that expression.

            | Unzip (ExpBase f vn) [f (CompTypeBase vn)] SrcLoc
            -- ^ Unzip that can unzip to tuples of arbitrary size.
            -- The types are the elements of the tuple.

            | Unsafe (ExpBase f vn) SrcLoc
            -- ^ Explore the Danger Zone and elide safety checks on
            -- array operations that are (lexically) within this
            -- expression.  Make really sure the code is correct.
deriving instance Showable f vn => Show (ExpBase f vn)

data StreamForm f vn = MapLike    StreamOrd
                     | RedLike    StreamOrd Commutativity (LambdaBase f vn) (ExpBase f vn)
                     | Sequential (ExpBase f vn)
deriving instance Showable f vn => Show (StreamForm f vn)

instance Located (ExpBase f vn) where
  locOf (Literal _ loc) = locOf loc
  locOf (TupLit _ pos) = locOf pos
  locOf (ArrayLit _ _ pos) = locOf pos
  locOf (Empty _ pos) = locOf pos
  locOf (BinOp _ _ _ _ pos) = locOf pos
  locOf (UnOp _ _ pos) = locOf pos
  locOf (If _ _ _ _ pos) = locOf pos
  locOf (Var ident) = locOf ident
  locOf (Apply _ _ _ pos) = locOf pos
  locOf (LetPat _ _ _ pos) = locOf pos
  locOf (LetWith _ _ _ _ _ pos) = locOf pos
  locOf (Index _ _ pos) = locOf pos
  locOf (Iota _ pos) = locOf pos
  locOf (Size _ _ pos) = locOf pos
  locOf (Replicate _ _ pos) = locOf pos
  locOf (Reshape _ _ pos) = locOf pos
  locOf (Transpose _ pos) = locOf pos
  locOf (Rearrange _ _ pos) = locOf pos
  locOf (Map _ _ pos) = locOf pos
  locOf (Reduce _ _ _ _ pos) = locOf pos
  locOf (Zip _ pos) = locOf pos
  locOf (Unzip _ _ pos) = locOf pos
  locOf (Scan _ _ _ pos) = locOf pos
  locOf (Filter _ _ pos) = locOf pos
  locOf (Partition _ _ pos) = locOf pos
  locOf (Split _ _ pos) = locOf pos
  locOf (Concat _ _ pos) = locOf pos
  locOf (Copy _ pos) = locOf pos
  locOf (DoLoop _ _ _ _ _ pos) = locOf pos
  locOf (Stream _ _ _  pos) = locOf pos
  locOf (Unsafe _ loc) = locOf loc
  locOf (Write _ _ _ loc) = locOf loc

-- | Whether the loop is a @for@-loop or a @while@-loop.
data LoopFormBase f vn = For ForLoopDirection (ExpBase f vn) (IdentBase f vn) (ExpBase f vn)
                       | While (ExpBase f vn)
deriving instance Showable f vn => Show (LoopFormBase f vn)

-- | The iteration order of a @for@-loop.
data ForLoopDirection = FromUpTo -- ^ Iterates from the lower bound to
                                 -- just below the upper bound.
                      | FromDownTo -- ^ Iterates from just below the
                                   -- upper bound to the lower bound.
                        deriving (Eq, Ord, Show)

-- | Anonymous Function
data LambdaBase f vn = AnonymFun [ParamBase f vn] (ExpBase f vn) (TypeDeclBase f vn) SrcLoc
                      -- ^ @fn int (bool x, char z) => if(x) then ord(z) else ord(z)+1 *)@
                      | CurryFun Name [ExpBase f vn] (f (CompTypeBase vn)) SrcLoc
                        -- ^ @f(4)@
                      | UnOpFun UnOp (f (CompTypeBase vn)) (f (CompTypeBase vn)) SrcLoc
                        -- ^ @-@; first type is operand, second is result.
                      | BinOpFun BinOp (f (CompTypeBase vn)) (f (CompTypeBase vn)) (f (CompTypeBase vn)) SrcLoc
                        -- ^ @+@; first two types are operands, third is result.
                      | CurryBinOpLeft BinOp (ExpBase f vn) (f (CompTypeBase vn)) (f (CompTypeBase vn)) SrcLoc
                        -- ^ @2+@; first type is operand, second is result.
                      | CurryBinOpRight BinOp (ExpBase f vn) (f (CompTypeBase vn)) (f (CompTypeBase vn)) SrcLoc
                        -- ^ @+2@; first type is operand, second is result.
deriving instance Showable f vn => Show (LambdaBase f vn)

instance Located (LambdaBase f vn) where
  locOf (AnonymFun _ _ _ loc)         = locOf loc
  locOf (CurryFun  _ _ _ loc)         = locOf loc
  locOf (UnOpFun _ _ _ loc)           = locOf loc
  locOf (BinOpFun _ _ _ _ loc)        = locOf loc
  locOf (CurryBinOpLeft _ _ _ _ loc)  = locOf loc
  locOf (CurryBinOpRight _ _ _ _ loc) = locOf loc

-- | Tuple IdentBaseifier, i.e., pattern matching
data PatternBase f vn = TuplePattern [PatternBase f vn] SrcLoc
                       | Id (IdentBase f vn)
                       | Wildcard (f (CompTypeBase vn)) SrcLoc -- Nothing, i.e. underscore.
deriving instance Showable f vn => Show (PatternBase f vn)

instance Located (PatternBase f vn) where
  locOf (TuplePattern _ loc) = locOf loc
  locOf (Id ident) = locOf ident
  locOf (Wildcard _ loc) = locOf loc

-- | Function Declarations
data FunDefBase f vn = FunDef { funDefEntryPoint :: Bool
                                -- ^ True if this function is an entry point.
                              , funDefName :: Name
                              , funDefRetType :: TypeDeclBase f vn
                              , funDefParams :: [ParamBase f vn]
                              , funDefBody :: ExpBase f vn
                              , funDefLocation :: SrcLoc
                              }
deriving instance Showable f vn => Show (FunDefBase f vn)

-- | Type Declarations
data TypeDefBase f vn = TypeDef { typeAlias :: Name -- Den selverklærede types navn
                                , userType :: TypeDeclBase f vn -- type-definitionen
                                , typeDefLocation :: SrcLoc
                                }
deriving instance Showable f vn => Show (TypeDefBase f vn)


data DecBase f vn = FunDec (FunDefBase f vn)
                  | TypeDec (TypeDefBase f vn)
-- | Coming soon  | SigDec ..
--                | ModDec ..
deriving instance Showable f vn => Show (DecBase f vn)

data ProgBase f vn =
         Prog  { progTypes :: [TypeDefBase f vn]
               , progFunctions :: [FunDefBase f vn]
               }
deriving instance Showable f vn => Show (ProgBase f vn)

data ProgBaseWithHeaders f vn =
  ProgWithHeaders { progWHHeaders :: [ProgHeader]
                  , progWHDecs :: [DecBase f vn]
                  }
deriving instance Showable f vn => Show (ProgBaseWithHeaders f vn)

data ProgHeader = Include [String]
                deriving (Show)

-- | A set of names.
type Names = HS.HashSet
