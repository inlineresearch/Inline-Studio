@echo off
rem ===========================================================================================
rem Launch Inline Studio (Inline Core + the web UI) on one port, on Windows.
rem
rem This is the Windows twin of webui.sh - keep the two in sync. Friendly flags map onto the
rem engine's INLINE_* environment knobs, so you do not have to remember them. Double-click this
rem file, or run it from PowerShell/cmd:
rem
rem   .\webui.bat                             loopback, port 8848 (UI + API)
rem   .\webui.bat --listen --port 9000        bind all interfaces on 9000
rem   .\webui.bat --install --extra runtime   set up .venv with the model runtime, then exit
rem
rem Run  .\webui.bat --help  for every flag.
rem ===========================================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8848"
set "EXTRAS=server"
set "RUN_INSTALL=0"
set "DEV_MODE=0"
set "FORCE_REBUILD=0"
set "SMART_MEMORY=0"

:parse
if "%~1"=="" goto after_parse
if /i "%~1"=="--listen"       ( set "HOST=0.0.0.0" & shift & goto parse )
if /i "%~1"=="--host"         ( set "HOST=%~2" & shift & shift & goto parse )
if /i "%~1"=="--port"         ( set "PORT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--multi-gpu"    goto multigpu
if /i "%~1"=="--parallel"     goto multigpu
if /i "%~1"=="--lowvram"      ( set "INLINE_PROFILE=lowvram" & shift & goto parse )
if /i "%~1"=="--smart-memory" goto smartmem
if /i "%~1"=="--cpu"          ( set "INLINE_PROFILE=cpu" & shift & goto parse )
if /i "%~1"=="--profile"      ( set "INLINE_PROFILE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--vram-budget"  ( set "INLINE_VRAM_BUDGET_GB=%~2" & shift & shift & goto parse )
if /i "%~1"=="--models-dir"   ( set "INLINE_MODELS_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--data-dir"     ( set "INLINE_DATA_DIR=%~2" & shift & shift & goto parse )
if /i "%~1"=="--install"      ( set "RUN_INSTALL=1" & shift & goto parse )
if /i "%~1"=="--extra"        ( set "EXTRAS=!EXTRAS!,%~2" & shift & shift & goto parse )
if /i "%~1"=="--dev"          ( set "DEV_MODE=1" & shift & goto parse )
if /i "%~1"=="--rebuild"      ( set "FORCE_REBUILD=1" & shift & goto parse )
if /i "%~1"=="-h"             ( call :usage & exit /b 0 )
if /i "%~1"=="--help"         ( call :usage & exit /b 0 )
echo unknown option: %~1
call :usage
goto fail

:multigpu
rem Multi-GPU split is auto-detected with 2+ GPUs; an optional SPEC (e.g. pipefusion=2) overrides it.
set "SPEC=%~2"
if "!SPEC!"=="" (
  echo Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override.
  shift & goto parse
)
if "!SPEC:~0,1!"=="-" (
  echo Multi-GPU split is auto-detected with 2+ GPUs; pass e.g. pipefusion=2 to override.
  shift & goto parse
)
set "INLINE_PARALLEL=!SPEC!"
shift & shift & goto parse

:smartmem
rem Spread a too-big model across VRAM + RAM + CPU; force lowvram so the offload + int8 path engages.
set "SMART_MEMORY=1"
set "INLINE_SMART_MEMORY=1"
if not defined INLINE_PROFILE set "INLINE_PROFILE=lowvram"
shift & goto parse

:after_parse

set "INLINE_HOST=%HOST%"
set "INLINE_PORT=%PORT%"

rem Expandable CUDA segments cut low-VRAM fragmentation OOMs (harmless on CPU). Respect a user value.
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

if "%RUN_INSTALL%"=="1" goto do_install
goto pick_python

rem --- Install: create .venv (via uv) and install, then exit -------------------------------------
rem Flat label-based control, not nested (...) blocks: multi-line parens inside parens break .bat.
:do_install
where uv >nul 2>nul || ( echo uv not found: https://docs.astral.sh/uv/ & goto fail )
uv venv || goto fail
rem PyPI's default torch is CPU-only on Windows, so installing blind generates on the CPU ~100x
rem slower with no error. When an NVIDIA GPU is present, resolve torch from the CUDA index.
set "TORCH_ARGS="
where nvidia-smi >nul 2>nul || goto install_cpu
nvidia-smi -L >nul 2>nul || goto install_cpu
set "TORCH_ARGS=--extra-index-url https://download.pytorch.org/whl/cu124"
echo NVIDIA GPU detected - installing the CUDA build of PyTorch.
goto install_pkgs
:install_cpu
echo No NVIDIA GPU detected - installing the default (CPU) build of PyTorch.
:install_pkgs
uv pip install !TORCH_ARGS! -e ".[!EXTRAS!]" || goto fail
uv pip install inline-studio-frontend >nul 2>nul && echo Installed the prebuilt web UI (inline-studio-frontend). || echo Note: inline-studio-frontend not installed; the UI will build from source or run API-only.
echo Installed extras: !EXTRAS!. Start with: .\webui.bat
exit /b 0

rem --- Pick the Python interpreter (and matching pip), in priority order -------------------------
:pick_python
if defined VIRTUAL_ENV ( set "PY=python" & set "PIP=python -m pip install" & goto have_python )
if exist ".venv\Scripts\python.exe" ( set "PY=.venv\Scripts\python.exe" & set "PIP=.venv\Scripts\python.exe -m pip install" & goto have_python )
where uv >nul 2>nul && ( set "PY=uv run python" & set "PIP=uv pip install" & goto have_python )
echo No .venv found and uv is not installed. Run  .\webui.bat --install  first.
goto fail

:have_python
if "%DEV_MODE%"=="1" ( call :run_dev & exit /b !ERRORLEVEL! )
if "%FORCE_REBUILD%"=="1" call :rebuild_frontend
if "%SMART_MEMORY%"=="1" call :ensure_smart_memory_deps

call :ensure_frontend

%PY% -m inline_core.server
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" ( echo. & echo Inline Studio exited with code %RC%. & pause )
exit /b %RC%

rem ================================ subroutines =================================================

:frontend_available
rem Succeeds (errorlevel 0) when a web UI is resolvable; sets INLINE_FRONTEND_ROOT for a local build.
if defined INLINE_FRONTEND_ROOT if exist "%INLINE_FRONTEND_ROOT%\index.html" exit /b 0
%PY% -c "import os,sys; import inline_studio_frontend as f; sys.exit(0 if os.path.isfile(os.path.join(os.path.dirname(f.__file__),'static','index.html')) else 1)" >nul 2>nul && exit /b 0
if exist "..\dist-web\index.html" (
  for %%I in ("..\dist-web") do set "INLINE_FRONTEND_ROOT=%%~fI"
  exit /b 0
)
exit /b 1

:ensure_frontend
call :frontend_available && exit /b 0
echo No web UI found - installing the prebuilt package (inline-studio-frontend)...
%PIP% inline-studio-frontend >nul 2>nul
call :frontend_available && exit /b 0
if exist "..\package.json" (
  where npm >nul 2>nul && (
    echo Building the web UI from source (npm)...
    pushd ..
    call npm ci && call npm run build:spa
    popd
    call :frontend_available && exit /b 0
  )
)
echo WARNING: no web UI available - serving API only. Install Node to build it, or run
echo          "%PIP% inline-studio-frontend" once it's published.
exit /b 0

:rebuild_frontend
if not exist "..\package.json" ( echo --rebuild needs the repo checkout and npm; skipping the rebuild. & exit /b 0 )
where npm >nul 2>nul || ( echo --rebuild needs npm; skipping the rebuild. & exit /b 0 )
echo Rebuilding the web UI from source (npm run build:spa)...
pushd ..
if not exist node_modules call npm ci
call npm run build:spa
popd
for %%I in ("..\dist-web") do set "INLINE_FRONTEND_ROOT=%%~fI"
exit /b 0

:ensure_smart_memory_deps
%PY% -c "import torchao" >nul 2>nul && exit /b 0
echo Smart memory: installing torchao (int8 quantization)...
%PIP% torchao >nul 2>nul ^
  && echo Installed torchao. ^
  || echo WARNING: could not install torchao; smart memory will run without int8 quant.
exit /b 0

:run_dev
if not exist "..\package.json" ( echo --dev needs the repo checkout and Node/npm. & goto fail )
where npm >nul 2>nul || ( echo --dev needs Node/npm ^(https://nodejs.org/^). & goto fail )
pushd ..
if not exist node_modules call npm ci
popd
echo Starting Inline Core (API) on %HOST%:%PORT% in a new window...
start "Inline Core (API)" %PY% -m inline_core.server
echo Starting the Vite dev server (HMR). Open http://localhost:5173  (NOT :%PORT%)
pushd ..
set "INLINE_CORE_URL=http://127.0.0.1:%PORT%"
call npm run dev:web
popd
exit /b 0

:usage
echo Usage: .\webui.bat [options]
echo.
echo Networking
echo   --listen               bind all interfaces (0.0.0.0), so other machines can reach it
echo   --host ADDR            bind a specific address (default 127.0.0.1)
echo   --port N               listen on port N (default 8848)
echo.
echo Multi-GPU (split one image's denoise across GPUs)
echo   --multi-gpu [SPEC]     enable the split; auto-detected with 2+ GPUs. Optional SPEC, e.g.
echo                          pipefusion=2 or pipefusion=2,ulysses=2
echo   --parallel SPEC        alias for --multi-gpu SPEC
echo.
echo Device / memory
echo   --lowvram              tight-VRAM profile (slicing + tiling + int8, weights resident)
echo   --smart-memory         spread a too-big model across VRAM + RAM + CPU (offload + int8)
echo   --cpu                  force the CPU profile
echo   --profile NAME         set the profile explicitly (gpu-max ^| lowvram ^| cpu)
echo   --vram-budget GB       treat the GPU as having GB of usable VRAM
echo.
echo Paths
echo   --models-dir PATH      where weights are scanned from (default .\models)
echo   --data-dir PATH        where runs and takes are written (default .\.inline)
echo.
echo Setup
echo   --install              create .venv (via uv) and install, then exit
echo   --extra NAME           add an install extra (repeatable): runtime, parallel, server, training
echo   -h, --help             show this help
echo.
echo Development
echo   --dev                  live-reload dev loop (Core in a new window + Vite HMR on :5173)
echo   --rebuild              force a fresh SPA build and serve it on the one port
exit /b 0

:fail
echo.
pause
exit /b 1
