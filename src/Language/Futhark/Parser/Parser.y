{
-- | Futhark parser written with Happy.
module Language.Futhark.Parser.Parser
  ( prog
  , expression
  , lambda
  , futharkType
  , anyValue
  , anyValues
  , ParserEnv (..)
  , ParserMonad
  , ReadLineMonad(..)
  , getLinesFromIO
  , getLinesFromTexts
  , getNoLines
  , newParserEnv
  )
  where

import Control.Applicative
import Control.Monad
import Control.Monad.Trans
import Control.Monad.Except
import Control.Monad.Reader
import Control.Monad.Trans.State
import Control.Applicative ((<$>), (<*>))
import Control.Arrow
import Data.Array
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Char (ord)
import Data.Maybe (fromMaybe)
import Data.Loc hiding (L) -- Lexer has replacements.
import qualified Data.HashMap.Lazy as HM
import Data.Monoid

import Language.Futhark.Syntax hiding (ID)
import Language.Futhark.Attributes
import Language.Futhark.Parser.Lexer

}

%name prog Prog
%name futharkType UserType
%name expression Exp
%name lambda FunAbstr
%name anyValue Value
%name anyValues CatValues

%tokentype { L Token }
%error { parseError }
%monad { ParserMonad }
%lexer { lexer } { L _ EOF }

%token
      if              { L $$ IF }
      then            { L $$ THEN }
      else            { L $$ ELSE }
      let             { L $$ LET }
      loop            { L $$ LOOP }
      in              { L $$ IN }
      default         { L $$ DEFAULT }
      int             { L $$ INT }
      float           { L $$ FLOAT }
      i8              { L $$ I8 }
      i16             { L $$ I16 }
      i32             { L $$ I32 }
      i64             { L $$ I64 }
      u8              { L $$ U8 }
      u16             { L $$ U16 }
      u32             { L $$ U32 }
      u64             { L $$ U64 }
      bool            { L $$ BOOL }
      f32             { L $$ F32 }
      f64             { L $$ F64 }

      id              { L _ (ID _) }

      intlit          { L _ (INTLIT _) }
      i8lit           { L _ (I8LIT _) }
      i16lit          { L _ (I16LIT _) }
      i32lit          { L _ (I32LIT _) }
      i64lit          { L _ (I64LIT _) }
      u8lit           { L _ (U8LIT _) }
      u16lit          { L _ (U16LIT _) }
      u32lit          { L _ (U32LIT _) }
      u64lit          { L _ (U64LIT _) }
      reallit         { L _ (REALLIT _) }
      f32lit          { L _ (F32LIT _) }
      f64lit          { L _ (F64LIT _) }
      stringlit       { L _ (STRINGLIT _) }
      charlit         { L _ (CHARLIT _) }

      '+'             { L $$ PLUS }
      '-'             { L $$ MINUS }
      '*'             { L $$ TIMES }
      '/'             { L $$ DIVIDE }
      '%'             { L $$ MOD }
      '//'            { L $$ QUOT }
      '%%'            { L $$ REM }
      '='             { L $$ EQU }
      '=='            { L $$ EQU2 }
      '!='            { L $$ NEQU }
      '<'             { L $$ LTH }
      '>'             { L $$ GTH }
      '<='            { L $$ LEQ }
      '>='            { L $$ GEQ }
      pow             { L $$ POW }
      '<<'            { L $$ SHIFTL }
      '>>'            { L $$ SHIFTR }
      '>>>'           { L $$ ZSHIFTR }
      '|'             { L $$ BOR }
      '&'             { L $$ BAND }
      '^'             { L $$ XOR }
      '('             { L $$ LPAR }
      ')'             { L $$ RPAR }
      '['             { L $$ LBRACKET }
      ']'             { L $$ RBRACKET }
      ','             { L $$ COMMA }
      '_'             { L $$ UNDERSCORE }
      '!'             { L $$ BANG }
      '.'             { L $$ DOT }
      fun             { L $$ FUN }
      entry           { L $$ ENTRY }
      fn              { L $$ FN }
      '=>'            { L $$ ARROW }
      '<-'            { L $$ SETTO }
      for             { L $$ FOR }
      do              { L $$ DO }
      with            { L $$ WITH }
      iota            { L $$ IOTA }
      size            { L $$ SIZE }
      replicate       { L $$ REPLICATE }
      map             { L $$ MAP }
      reduce          { L $$ REDUCE }
      reduceComm      { L $$ REDUCECOMM }
      reshape         { L $$ RESHAPE }
      rearrange       { L $$ REARRANGE }
      transpose       { L $$ TRANSPOSE }
      zipWith         { L $$ ZIPWITH }
      zip             { L $$ ZIP }
      unzip           { L $$ UNZIP }
      unsafe          { L $$ UNSAFE }
      scan            { L $$ SCAN }
      split           { L $$ SPLIT }
      concat          { L $$ CONCAT }
      filter          { L $$ FILTER }
      partition       { L $$ PARTITION }
      true            { L $$ TRUE }
      false           { L $$ FALSE }
      '~'             { L $$ TILDE }
      abs             { L $$ ABS }
      signum          { L $$ SIGNUM }
      '&&'            { L $$ AND }
      '||'            { L $$ OR }
      empty           { L $$ EMPTY }
      copy            { L $$ COPY }
      while           { L $$ WHILE }
      streamMap       { L $$ STREAM_MAP }
      streamMapPer    { L $$ STREAM_MAPPER }
      streamRed       { L $$ STREAM_RED }
      streamRedPer    { L $$ STREAM_REDPER }
      streamSeq       { L $$ STREAM_SEQ }
      include         { L $$ INCLUDE }
      write           { L $$ WRITE }
      type            { L $$ TYPE}

%nonassoc ifprec letprec
%left '||'
%left '&&'
%left '&' '^' '|'
%left '<=' '>=' '>' '<' '==' '!='

%left '<<' '>>' '>>>'
%left '+' '-'

%left '*' '/' '%' '//' '%%'
%left pow
%nonassoc '~' '!' signum abs float f32 f64 int i8 i16 i32 i64 unsafe default
%nonassoc '['
%%

Prog :: { UncheckedProgWithHeaders }
     :   Headers DecStart { ProgWithHeaders $1 $2 }
     |   DecStart { ProgWithHeaders [] $1 }
;


DecStart :: { [DecBase f vn] }
         :  DefaultDec Decs { $2 }
         |  Decs { $1 }
;

Decs :: { [DecBase f vn] }
     : Dec Decs { $1 : $2 }
     | Dec { [$1] }
;

Dec :: { DecBase f vn }
       : Fun { FunDec $1 }
       | UserTypeAlias { TypeDec $1 }
;


DefaultDec :: { () }
           :  default '(' SignedType ')' {% defaultIntType (fst $3)  }
           |  default '(' FloatType ')' {% defaultRealType (fst $3) }
           |  default '(' SignedType ',' FloatType ')'
                {% defaultIntType (fst $3) >> defaultRealType (fst $5) }
;

-- Note that this production does not include Minus.
BinOp :: { (BinOp, SrcLoc) }
      : '+'     { (Plus, $1) }
      | '*'     { (Times, $1) }
      | '/'     { (Divide, $1) }
      | '%'     { (Mod, $1) }
      | '//'    { (Quot, $1) }
      | '%%'    { (Rem, $1) }
      | '=='    { (Equal, $1) }
      | '!='    { (NotEqual, $1) }
      | '<'     { (Less, $1) }
      | '<='    { (Leq, $1) }
      | '>'     { (Greater, $1) }
      | '>='    { (Geq, $1) }
      | '&&'    { (LogAnd, $1) }
      | '||'    { (LogOr, $1) }
      | pow     { (Pow, $1) }
      | '^'     { (Xor, $1) }
      | '&'     { (Band, $1) }
      | '|'     { (Bor, $1) }
      | '>>'    { (ShiftR, $1) }
      | '>>>'   { (ZShiftR, $1) }
      | '<<'    { (ShiftL, $1) }

UnOp :: { (UnOp, SrcLoc) }
     : '~' { (Complement, $1) }
     | '!' { (Not, $1) }
     | abs { (Abs, $1) }
     | signum { (Signum, $1) }
     | SignedType { (ToSigned (fst $1), snd $1) }
     | UnsignedType { (ToUnsigned (fst $1), snd $1) }
     | FloatType { (ToFloat (fst $1), snd $1) }

Headers :: { [ProgHeader] }
        : Header Headers { $1 : $2 }
        | Header { [$1] }
;

Header :: { ProgHeader }
Header : include IncludeParts { Include $2 }
;

IncludeParts :: { [String] }
IncludeParts : id '.' IncludeParts { let L pos (ID name) = $1 in nameToString name : $3 }
IncludeParts : id { let L pos (ID name) = $1 in [nameToString name] }

Fun     : fun UserTypeDecl id '(' Params ')' '=' Exp
                        { let L pos (ID name) = $3
                          in FunDef (name==defaultEntryPoint) name $2 $5 $8 pos }
        | fun UserTypeDecl id '(' ')' '=' Exp
                        { let L pos (ID name) = $3
                          in FunDef (name==defaultEntryPoint) name $2 [] $7 pos }
        | entry UserTypeDecl id '(' Params ')' '=' Exp
                        { let L pos (ID name) = $3
                          in FunDef True name $2 $5 $8 pos }
        | entry UserTypeDecl id '(' ')' '=' Exp
                        { let L pos (ID name) = $3
                          in FunDef True name $2 [] $7 pos }
;
Uniqueness :: { Uniqueness }
Uniqueness : '*' { Unique }
           |     { Nonunique }


UserTypeDecl :: { TypeDeclBase NoInfo Name }
              : UserType { TypeDecl $1 NoInfo }

UserTypeAlias :: { TypeDefBase NoInfo Name }
UserTypeAlias  : type id '=' UserType { let L loc (ID name) = $2
                                        in TypeDef name (TypeDecl $4 NoInfo) loc
                                      }

UserType :: { UncheckedUserType }
         : PrimType      { let (t,loc) = $1 in UserPrim t loc }
         | '*' UserType  { UserUnique $2 $1 }
         | '[' UserType DimDecl ']' { UserArray $2 $3 $1 }
         | '(' UserType ',' UserTypes ')' { UserTuple ($2:$4) $1 }
         | id            { let L loc (ID name) = $1 in UserTypeAlias name loc }
;

UserTypes :: { [UncheckedUserType] }
UserTypes : UserType ',' UserTypes { $1 : $3 }
          | UserType               { [$1] }
DimDecl :: { DimDecl Name }
        : ',' id
          { let L _ (ID name) = $2
            in NamedDim name }
        | ',' intlit
          { let L _ (INTLIT n) = $2
            in ConstDim (fromIntegral n) }
        | { AnyDim }

PrimType :: { (PrimType, SrcLoc) }
         : UnsignedType { first Unsigned $1 }
         | SignedType   { first Signed $1 }
         | FloatType    { first FloatType $1 }
         | bool         { (Bool, $1) }

SignedType :: { (IntType, SrcLoc) }
           : int { (Int32, $1) }
           | i8  { (Int8, $1) }
           | i16 { (Int16, $1) }
           | i32 { (Int32, $1) }
           | i64 { (Int64, $1) }

UnsignedType :: { (IntType, SrcLoc) }
             : u8  { (Int8, $1) }
             | u16 { (Int16, $1) }
             | u32 { (Int32, $1) }
             | u64 { (Int64, $1) }

FloatType :: { (FloatType, SrcLoc) }
          : float { (Float64, $1) }
          | f32  { (Float32, $1) }
          | f64  { (Float64, $1) }

Params :: { [ParamBase NoInfo Name] }
Params : UserTypeDecl id ',' Params { let L pos (ID name) = $2 in (Param name $1 pos) : $4 }
       | UserTypeDecl id            { let L pos (ID name) = $2 in [Param name $1 pos] }

Exp  :: { UncheckedExp }
     : PrimLit        { Literal (PrimValue (fst $1)) (snd $1) }
     | stringlit      {% let L pos (STRINGLIT s) = $1 in do
                             s' <- mapM (getIntValue . fromIntegral . ord) s
                             t <- lift $ gets parserIntType
                             return $ Literal (ArrayValue (arrayFromList $ map (PrimValue . SignedValue) s') $ Prim $ Signed t) pos }
     | Id %prec letprec { Var $1 }
     | empty '(' UserType ')' { Empty (TypeDecl $3 NoInfo) $1 }
     | '[' Exps ']'   { ArrayLit $2 NoInfo $1 }
     | '(' Exp ',' Exps ')'   { TupLit ($2:$4) $1 }
     | '('      ')'   { TupLit [] $1 }
     | Exp '+' Exp    { BinOp Plus $1 $3 NoInfo $2 }
     | Exp '-' Exp    { BinOp Minus $1 $3 NoInfo $2 }
     | Exp '*' Exp    { BinOp Times $1 $3 NoInfo $2 }
     | Exp '/' Exp    { BinOp Divide $1 $3 NoInfo $2 }
     | Exp '%' Exp    { BinOp Mod $1 $3 NoInfo $2 }
     | Exp '//' Exp   { BinOp Quot $1 $3 NoInfo $2 }
     | Exp '%%' Exp   { BinOp Rem $1 $3 NoInfo $2 }
     | '-' Exp %prec '~' { UnOp Negate $2 $1 }
     | '!' Exp        { UnOp Not $2 $1 }
     | '~' Exp        { UnOp Complement $2 $1 }
     | abs Exp        { UnOp Abs $2 $1 }
     | signum Exp     { UnOp Signum $2 $1 }
     | SignedType '(' Exp ')' { UnOp (ToSigned (fst $1)) $3 (snd $1) }
     | UnsignedType '(' Exp ')' { UnOp (ToUnsigned (fst $1)) $3 (snd $1) }
     | FloatType '(' Exp ')' { UnOp (ToFloat (fst $1)) $3 (snd $1) }
     | Exp pow Exp    { BinOp Pow $1 $3 NoInfo $2 }
     | Exp '>>' Exp   { BinOp ShiftR $1 $3 NoInfo $2 }
     | Exp '>>>' Exp  { BinOp ZShiftR $1 $3 NoInfo $2 }
     | Exp '<<' Exp   { BinOp ShiftL $1 $3 NoInfo $2 }
     | Exp '&&' Exp   { BinOp LogAnd $1 $3 NoInfo $2 }
     | Exp '||' Exp   { BinOp LogOr $1 $3 NoInfo $2 }
     | Exp '&' Exp    { BinOp Band $1 $3 NoInfo $2 }
     | Exp '|' Exp    { BinOp Bor $1 $3 NoInfo $2 }
     | Exp '^' Exp    { BinOp Xor $1 $3 NoInfo $2 }

     | Exp '==' Exp   { BinOp Equal $1 $3 NoInfo $2 }
     | Exp '!=' Exp   { BinOp NotEqual $1 $3 NoInfo $2 }
     | Exp '<' Exp    { BinOp Less $1 $3 NoInfo $2 }
     | Exp '<=' Exp   { BinOp Leq  $1 $3 NoInfo $2 }
     | Exp '>' Exp    { BinOp Greater $1 $3 NoInfo $2 }
     | Exp '>=' Exp   { BinOp Geq  $1 $3 NoInfo $2 }

     | if Exp then Exp else Exp %prec ifprec
                      { If $2 $4 $6 NoInfo $1 }

     | id '(' Exps ')'
                      { let L pos (ID name) = $1
                        in Apply name [ (arg, Observe) | arg <- $3 ] NoInfo pos
                      }
     | id '(' ')'     { let L pos (ID name) = $1
                        in Apply name [] NoInfo pos }

     | iota '(' Exp ')' { Iota $3 $1 }

     | size '(' NaturalInt ',' Exp ')'
                      { Size $3 $5 $1 }

     | replicate '(' Exp ',' Exp ')' { Replicate $3 $5 $1 }

     | reshape '(' '(' Exps ')' ',' Exp ')'
                      { Reshape $4 $7 $1 }

     | rearrange '(' '(' NaturalInts ')' ',' Exp ')'
                      { Rearrange $4 $7 $1 }

     | transpose '(' Exp ')' { Transpose $3 $1 }

     | split '(' '(' Exps ')' ',' Exp ')'
                      { Split $4 $7 $1 }

     | concat '(' Exp ',' Exps ')'
                      { Concat $3 $5 $1 }

     | reduce '(' FunAbstr ',' Exp ',' Exp ')'
                      { Reduce (commutativity $3) $3 $5 $7 $1 }

     | reduceComm '(' FunAbstr ',' Exp ',' Exp ')'
                      { Reduce Commutative $3 $5 $7 $1 }


     | map '(' FunAbstr ',' Exp ')'
                      { Map $3 $5 $1 }

     | scan '(' FunAbstr ',' Exp ',' Exp ')'
                      { Scan $3 $5 $7 $1 }

     | zip '(' Exps ')'
                      { Zip (map (\x -> (x, NoInfo)) $3) $1 }

     | unzip '(' Exp ')'
                      { Unzip $3 [] $1 }

     | unsafe Exp     { Unsafe $2 $1 }

     | filter '(' FunAbstr ',' Exp ')'
                      { Filter $3 $5 $1 }

     | partition '(' FunAbstrsThenExp ')'
                      { Partition (fst $3) (snd $3) $1 }

     | zipWith '(' FunAbstr ',' Exps ')'
                      { Map $3 (Zip (map (\x -> (x, NoInfo)) $5) $1) $1 }

     | copy '(' Exp ')' { Copy $3 $1 }

     | '(' Exp ')' { $2 }

     | LetExp         { $1 }

     | Exp Index
                      { Index $1 $2 (srclocOf $1) }

     | streamMap       '(' FunAbstr ',' Exp ')'
                         { Stream (MapLike InOrder)  $3 $5 $1 }
     | streamMapPer    '(' FunAbstr ',' Exp ')'
                         { Stream (MapLike Disorder) $3 $5 $1 }
     | streamRed       '(' FunAbstr ',' FunAbstr ',' Exp ',' Exp ')'
                         { Stream (RedLike InOrder (commutativity $3) $3 $7) $5 $9 $1 }
     | streamRedPer    '(' FunAbstr ',' FunAbstr ',' Exp ',' Exp ')'
                         { Stream (RedLike Disorder Commutative $3 $7) $5 $9 $1 }
     | streamSeq       '(' FunAbstr ',' Exp ',' Exp ')'
                         { Stream (Sequential $5) $3 $7 $1 }
     | write           '(' Exp ',' Exp ',' Exp ')'
                         { Write $3 $5 $7 $1 }

LetExp :: { UncheckedExp }
     : let Id '=' Exp LetBody
                      { LetPat (Id $2) $4 $5 $1 }

     | let '_' '=' Exp LetBody
                      { LetPat (Wildcard NoInfo $2) $4 $5 $1 }

     | let '(' Patterns ')' '=' Exp LetBody
                      { LetPat (TuplePattern $3 $1) $6 $7 $1 }

     | let Id '=' Id with Index '<-' Exp LetBody
                      { LetWith $2 $4 $6 $8 $9 $1 }

     | let Id Index '=' Exp LetBody
                      { LetWith $2 $2 $3 $5 $6 $1 }

     | let Id '[' ']' '=' Exp LetBody
                      { LetWith $2 $2 [] $6 $7 $1 }

     | loop '(' Pattern ')' '=' LoopForm do Exp LetBody
                      {% liftM (\t -> DoLoop $3 t $6 $8 $9 $1)
                               (patternExp $3) }
     | loop '(' Pattern '=' Exp ')' '=' LoopForm do Exp LetBody
                  { DoLoop $3 $5 $8 $10 $11 $1 }

LetBody :: { UncheckedExp }
    : in Exp %prec letprec { $2 }
    | LetExp { $1 }

LoopForm : for Id '<' Exp
           { For FromUpTo (zeroExpression (srclocOf $1)) $2 $4 }
         | for Exp '<=' Id '<' Exp
           { For FromUpTo $2 $4 $6 }
         | for Exp '>' Id '>=' Exp
           { For FromDownTo $6 $4 $2 }
         | for Exp '>' Id
           { For FromDownTo (zeroExpression (srclocOf $1)) $4 $2 }
         | while Exp      { While $2 }

Index : '[' Exps ']'                  { $2 }

Exps : Exp ',' Exps { $1 : $3 }
     | Exp          { [$1] }

Id : id { let L loc (ID name) = $1 in Ident name NoInfo loc }

Patterns : Pattern ',' Patterns  { $1 : $3 }
         | Pattern               { [$1] }
         |                       { [] }
;

Pattern : id { let L pos (ID name) = $1 in Id $ Ident name NoInfo pos }
      | '_' { Wildcard NoInfo $1 }
      | '(' Patterns ')' { TuplePattern $2 $1 }

FunAbstr :: { UncheckedLambda }
         : fn UserTypeDecl '(' Params ')' '=>' Exp
           { AnonymFun $4 $7 $2 $1 }
         | id '(' Exps ')'
           { let L pos (ID name) = $1
             in CurryFun name $3 NoInfo pos }
         | id '(' ')'
           { let L pos (ID name) = $1
             in CurryFun name [] NoInfo pos }
         | id
           { let L pos (ID name) = $1
             in CurryFun name [] NoInfo pos }
           -- Minus is handed explicitly here because I could figure
           -- out how to resolve the ambiguity with negation.
         | '-' Exp
           { CurryBinOpRight Minus $2 NoInfo NoInfo $1 }
         | '-'
           { BinOpFun Minus NoInfo NoInfo NoInfo $1 }
         | Exp '-'
           { CurryBinOpLeft Minus $1 NoInfo NoInfo (srclocOf $1) }
         | BinOp Exp
           { CurryBinOpRight (fst $1) $2 NoInfo NoInfo (snd $1) }
         | Exp BinOp
           { CurryBinOpLeft (fst $2) $1 NoInfo NoInfo (snd $2) }
         | BinOp
           { BinOpFun (fst $1) NoInfo NoInfo NoInfo (snd $1) }
         | UnOp
           { UnOpFun (fst $1) NoInfo NoInfo (snd $1) }

FunAbstrsThenExp : FunAbstr ',' Exp              { ([$1], $3) }
                 | FunAbstr ',' FunAbstrsThenExp { ($1 : fst $3, snd $3) }

Value : IntValue { $1 }
      | FloatValue { $1 }
      | StringValue { $1 }
      | BoolValue { $1 }
      | ArrayValue { $1 }
      | TupleValue { $1 }

CatValues : Value CatValues { $1 : $2 }
          |                 { [] }

NaturalInt :: { Int }
           :  intlit { let L _ (INTLIT num) = $1 in fromIntegral num  }

NaturalInts :: { [Int] }
           : intlit                 { let L _ (INTLIT num) = $1 in [fromIntegral num] }
           | intlit ',' NaturalInts { let L _ (INTLIT num) = $1 in fromIntegral num : $3  }

IntValue :: { Value }
         : SignedLit { PrimValue (SignedValue (fst $1)) }
         | '-' SignedLit { PrimValue (SignedValue (intNegate (fst $2))) }
         | UnsignedLit { PrimValue (UnsignedValue (fst $1)) }

FloatValue :: { Value }
         : FloatLit { PrimValue (FloatValue (fst $1)) }
         | '-' FloatLit { PrimValue (FloatValue (floatNegate (fst $2))) }

StringValue : stringlit  {% let L pos (STRINGLIT s) = $1 in do
                             s' <- mapM (getIntValue . fromIntegral . ord) s
                             t <- lift $ gets parserIntType
                             return $ ArrayValue (arrayFromList $ map (PrimValue . SignedValue) s') $ Prim $ Signed t }
BoolValue : true           { PrimValue $ BoolValue True }
          | false          { PrimValue $ BoolValue False }

SignedLit :: { (IntValue, SrcLoc) }
          : i8lit   { let L pos (I8LIT num)  = $1 in (Int8Value num, pos) }
          | i16lit  { let L pos (I16LIT num) = $1 in (Int16Value num, pos) }
          | i32lit  { let L pos (I32LIT num) = $1 in (Int32Value num, pos) }
          | i64lit  { let L pos (I64LIT num) = $1 in (Int64Value num, pos) }
          | intlit  {% let L pos (INTLIT num) = $1 in do num' <- getIntValue num; return (num', pos) }
          | charlit {% let L pos (CHARLIT char) = $1 in do
                       num <- getIntValue $ fromIntegral $ ord char
                       return (num, pos) }

UnsignedLit :: { (IntValue, SrcLoc) }
            : u8lit  { let L pos (U8LIT num)  = $1 in (Int8Value num, pos) }
            | u16lit { let L pos (U16LIT num) = $1 in (Int16Value num, pos) }
            | u32lit { let L pos (U32LIT num) = $1 in (Int32Value num, pos) }
            | u64lit { let L pos (U64LIT num) = $1 in (Int64Value num, pos) }

FloatLit :: { (FloatValue, SrcLoc) }
         : f32lit { let L pos (F32LIT num) = $1 in (Float32Value num, pos) }
         | f64lit { let L pos (F64LIT num) = $1 in (Float64Value num, pos) }
         | reallit {% let L pos (REALLIT num) = $1 in do num' <- getRealValue num; return (num', pos) }

PrimLit :: { (PrimValue, SrcLoc) }
        : SignedLit { let (x,loc) = $1 in (SignedValue x, loc) }
        | UnsignedLit { let (x,loc) = $1 in (UnsignedValue x, loc) }
        | FloatLit { let (x,loc) = $1 in (FloatValue x, loc) }

        | true   { (BoolValue True, $1) }
        | false  { (BoolValue False, $1) }

ArrayValue :  '[' Value ']'
             {% return $ ArrayValue (arrayFromList [$2]) $ removeNames $ toStruct $ valueType $2
             }
           |  '[' Value ',' Values ']'
             {% case combArrayTypes (valueType $2) $ map valueType $4 of
                  Nothing -> throwError "Invalid array value"
                  Just ts -> return $ ArrayValue (arrayFromList $ $2:$4) $ removeNames ts
             }
           | empty '(' PrimType ')'
             { ArrayValue (listArray (0,-1) []) (Prim (fst $3)) }
TupleValue : '(' Values ')' { TupValue $2 }

Values : Value ',' Values { $1 : $3 }
       | Value            { [$1] }
       |                  { [] }

{

data ParserEnv = ParserEnv {
                 parserFile :: FilePath
               , parserIntType :: IntType
               , parserRealType :: FloatType
               , parserRealFun :: Double -> FloatValue
               }

newParserEnv :: FilePath -> IntType -> FloatType -> ParserEnv
newParserEnv path intType realType =
  let s = ParserEnv path intType realType Float64Value
  in modParserEnv s realType

modParserEnv :: ParserEnv -> FloatType -> ParserEnv
modParserEnv s realType =
  case realType of
    Float32 -> s {
        parserRealType = Float32,
        parserRealFun = float32RealFun
      }
    Float64 -> s {
        parserRealType = Float64,
        parserRealFun = float64RealFun
      }
  where

    float32RealFun x =
      let (m,n) = decodeFloat x
      in Float32Value $ encodeFloat m n
    float64RealFun = Float64Value

type ParserMonad a =
  ExceptT String (
    StateT ParserEnv (
       StateT [L Token] ReadLineMonad)) a

data ReadLineMonad a = Value a
                     | GetLine (T.Text -> ReadLineMonad a)

readLineFromMonad :: ReadLineMonad T.Text
readLineFromMonad = GetLine Value

instance Monad ReadLineMonad where
  return = Value
  Value x >>= f = f x
  GetLine g >>= f = GetLine $ \s -> g s >>= f

instance Functor ReadLineMonad where
  f `fmap` m = do x <- m
                  return $ f x

instance Applicative ReadLineMonad where
  (<*>) = ap

getLinesFromIO :: ReadLineMonad a -> IO a
getLinesFromIO (Value x) = return x
getLinesFromIO (GetLine f) = do
  s <- T.getLine
  getLinesFromIO $ f s

getLinesFromTexts :: [T.Text] -> ReadLineMonad a -> Either String a
getLinesFromTexts _ (Value x) = Right x
getLinesFromTexts (x : xs) (GetLine f) = getLinesFromTexts xs $ f x
getLinesFromTexts [] (GetLine _) = Left "Ran out of input"

getNoLines :: ReadLineMonad a -> Either String a
getNoLines (Value x) = Right x
getNoLines (GetLine _) = Left "Unexpected end of input"

combArrayTypes :: TypeBase Rank NoInfo Name
               -> [TypeBase Rank NoInfo Name]
               -> Maybe (TypeBase Rank NoInfo Name)
combArrayTypes t ts = foldM comb t ts
  where comb x y
          | x == y    = Just x
          | otherwise = Nothing

arrayFromList :: [a] -> Array Int a
arrayFromList l = listArray (0, length l-1) l

patternExp :: UncheckedPattern -> ParserMonad UncheckedExp
patternExp (Id ident) = return $ Var ident
patternExp (TuplePattern pats loc) = TupLit <$> (mapM patternExp pats) <*> return loc
patternExp (Wildcard _ loc) = throwError $ "Cannot have wildcard at " ++ locStr loc

zeroExpression :: SrcLoc -> UncheckedExp
zeroExpression = Literal $ PrimValue $ SignedValue $ Int32Value 0

commutativity :: LambdaBase ty vn -> Commutativity
commutativity (BinOpFun binop _ _ _ _)
  | commutative binop = Commutative
commutativity _ = Noncommutative

eof :: L Token
eof = L (SrcLoc $ Loc (Pos "" 0 0 0) (Pos "" 0 0 0)) EOF

getTokens :: ParserMonad [L Token]
getTokens = lift $ lift get

putTokens :: [L Token] -> ParserMonad ()
putTokens ts = lift $ lift $ put ts

defaultIntType :: IntType -> ParserMonad ()
defaultIntType intType = do
  s <- lift $ get
  lift $ put $ s { parserIntType = intType }

defaultRealType :: FloatType -> ParserMonad ()
defaultRealType realType = do
  s <- lift $ get
  lift $ put $ modParserEnv s realType

getFilename :: ParserMonad FilePath
getFilename = lift $ gets parserFile

getIntValue :: Int64 -> ParserMonad IntValue
getIntValue x = do
  t <- lift $ gets parserIntType
  return $ (getIntFun t) (toInteger x)

getIntFun :: IntType -> (Integer -> IntValue)
getIntFun Int8  = Int8Value . fromInteger
getIntFun Int16 = Int16Value . fromInteger
getIntFun Int32 = Int32Value . fromInteger
getIntFun Int64 = Int64Value . fromInteger

getRealValue :: Double -> ParserMonad FloatValue
getRealValue x = do f <- lift $ gets parserRealFun
                    return $ f x

intNegate :: IntValue -> IntValue
intNegate (Int8Value v) = Int8Value (-v)
intNegate (Int16Value v) = Int16Value (-v)
intNegate (Int32Value v) = Int32Value (-v)
intNegate (Int64Value v) = Int64Value (-v)

floatNegate :: FloatValue -> FloatValue
floatNegate (Float32Value v) = Float32Value (-v)
floatNegate (Float64Value v) = Float64Value (-v)

readLine :: ParserMonad T.Text
readLine = lift $ lift $ lift readLineFromMonad

lexer :: (L Token -> ParserMonad a) -> ParserMonad a
lexer cont = do
  ts <- getTokens
  case ts of
    [] -> do
      ended <- lift $ runExceptT $ cont eof
      case ended of
        Right x -> return x
        Left _ -> do
          ts' <- scanTokens <$> getFilename <*> readLine
          ts'' <- case ts' of Right x -> return x
                              Left e  -> throwError e
          case ts'' of
            [] -> cont eof
            xs -> do
              putTokens xs
              lexer cont
    (x : xs) -> do
      putTokens xs
      cont x

parseError :: L Token -> ParserMonad a
parseError (L _ EOF) = throwError "Parse error: End of file"
parseError tok       = throwError $ "Parse error at " ++ locStr (srclocOf tok)
}
