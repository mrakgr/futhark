module Main (main) where

import Control.Category ((>>>))
import Control.Monad.IO.Class
import Data.Maybe
import System.FilePath
import System.Process
import System.Exit
import System.Console.GetOpt
import qualified System.Info

import Futhark.Pipeline
import Futhark.Passes
import Futhark.Compiler
import Futhark.Representation.SOACS (SOACS)
import Futhark.Representation.ExplicitMemory (ExplicitMemory)
import Futhark.Pass.ExplicitAllocations
import qualified Futhark.CodeGen.Backends.COpenCL as COpenCL
import Futhark.Optimise.InPlaceLowering
import Futhark.Optimise.CSE
import Futhark.Pass.Simplify
import Futhark.Pass.ExtractKernels
import Futhark.Pass.KernelBabysitting
import Futhark.Pass.ExpandAllocations
import Futhark.Pass.CoalesceMemoryAccesses
import Futhark.Util.Options
import Futhark.Optimise.DoubleBuffer

main :: IO ()
main = mainWithOptions newCompilerConfig commandLineOptions inspectNonOptions
  where inspectNonOptions [file] config = Just $ compile config file
        inspectNonOptions _      _      = Nothing

compile :: CompilerConfig -> FilePath -> IO ()
compile config filepath =
  runCompilerOnProgram (futharkConfig config)
  compilerPipeline (openclCodeAction filepath config) filepath

openclCodeAction :: FilePath -> CompilerConfig -> Action ExplicitMemory
openclCodeAction filepath config =
  Action { actionName = "Compile OpenCL"
         , actionDescription = "Generate OpenCL/C code from optimised Futhark program."
         , actionProcedure = procedure
         }
  where procedure prog = do
          cprog <- either compileFail return =<< COpenCL.compileProg prog
          let binpath = outputFilePath filepath config
              cpath = binpath `replaceExtension` "c"
          liftIO $ writeFile cpath cprog
          let args
                | System.Info.os == "darwin" =
                    [cpath, "-o", binpath, "-lm", "-O3", "-std=c99", "-framework", "OpenCL"]
                | System.Info.os == "mingw32" =
                    [cpath, "-o", binpath, "-lm", "-O3", "-std=c99", "-lOpenCL64"]
                | otherwise =
                    [cpath, "-o", binpath, "-lm", "-O3", "-std=c99", "-lOpenCL"]
          (gccCode, _, gccerr) <-
            liftIO $ readProcessWithExitCode "gcc" args ""
          case gccCode of
            ExitFailure code -> compileFail $ "gcc failed with code " ++ show code ++ ":\n" ++ gccerr
            ExitSuccess      -> return ()

type CompilerOption = OptDescr (Either (IO ()) (CompilerConfig -> CompilerConfig))

commandLineOptions :: [CompilerOption]
commandLineOptions =
  [ Option "o" []
    (ReqArg (\filename -> Right $ \config -> config { compilerOutput = Just filename })
     "FILE")
    "Name of the compiled binary."
  , Option "V" ["verbose"]
    (OptArg (\file -> Right $ \config -> config { compilerVerbose = Just file }) "FILE")
    "Print verbose output on standard error; wrong program to FILE."
  ]

data CompilerConfig =
  CompilerConfig { compilerOutput :: Maybe FilePath
                 , compilerVerbose :: Maybe (Maybe FilePath)
                 }

newCompilerConfig :: CompilerConfig
newCompilerConfig = CompilerConfig { compilerOutput = Nothing
                                   , compilerVerbose = Nothing
                                   }

outputFilePath :: FilePath -> CompilerConfig -> FilePath
outputFilePath srcfile =
  fromMaybe (srcfile `replaceExtension` "") . compilerOutput

futharkConfig :: CompilerConfig -> FutharkConfig
futharkConfig config =
  newFutharkConfig { futharkVerbose = compilerVerbose config }

compilerPipeline :: Pipeline SOACS ExplicitMemory
compilerPipeline =
  standardPipeline >>>
  onePass extractKernels >>>
  passes [ simplifyKernels
         , babysitKernels
         , simplifyKernels
         , performCSE True
         , simplifyKernels
         , inPlaceLowering
         ] >>>
  onePass explicitAllocations >>>
  passes [ simplifyExplicitMemory
         , performCSE False
         , simplifyExplicitMemory
         , doubleBuffer
         , simplifyExplicitMemory
         , expandAllocations
         , coalesceMemoryAccesses
         , simplifyExplicitMemory
         ]
