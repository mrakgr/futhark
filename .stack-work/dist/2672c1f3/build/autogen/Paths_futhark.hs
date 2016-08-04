module Paths_futhark (
    version,
    getBinDir, getLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
catchIO = Exception.catch

version :: Version
version = Version [0,1] []
bindir, libdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "C:\\futhark_parser_only\\.stack-work\\install\\a604a5e3\\bin"
libdir     = "C:\\futhark_parser_only\\.stack-work\\install\\a604a5e3\\lib\\x86_64-windows-ghc-7.10.3\\futhark-0.1-GJqcXjzuDAgKwUW9oTf4td"
datadir    = "C:\\futhark_parser_only\\.stack-work\\install\\a604a5e3\\share\\x86_64-windows-ghc-7.10.3\\futhark-0.1"
libexecdir = "C:\\futhark_parser_only\\.stack-work\\install\\a604a5e3\\libexec"
sysconfdir = "C:\\futhark_parser_only\\.stack-work\\install\\a604a5e3\\etc"

getBinDir, getLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "futhark_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "futhark_libdir") (\_ -> return libdir)
getDataDir = catchIO (getEnv "futhark_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "futhark_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "futhark_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "\\" ++ name)
