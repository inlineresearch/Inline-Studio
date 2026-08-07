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

rem Absolute, so uv is pinned to this venv no matter what is activated in the shell.
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

set "HOST=127.0.0.1"
set "PORT=8848"
set "EXTRAS=server"
set "RUN_INSTALL=0"
set "DEV_MODE=0"
set "FORCE_REBUILD=0"
set "SMART_MEMORY=0"
set "USE_ACTIVE_ENV=0"
set "RECREATE=0"
rem Empty means "decide from the GPU's compute capability at install time" (see :install_torch_args).
set "TORCH_CHOICE=%INLINE_TORCH_INDEX%"

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
if /i "%~1"=="--torch-index"  ( set "TORCH_CHOICE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--recreate"     ( set "RECREATE=1" & shift & goto parse )
if /i "%~1"=="--use-active-env" ( set "USE_ACTIVE_ENV=1" & shift & goto parse )
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

set "ACTIVE_ENV="
if defined CONDA_PREFIX set "ACTIVE_ENV=%CONDA_PREFIX%"
if defined VIRTUAL_ENV set "ACTIVE_ENV=%VIRTUAL_ENV%"
call :foreign_env

if "%RUN_INSTALL%"=="1" goto do_install
goto pick_python

rem --- Install: create .venv (via uv) and install, then exit -------------------------------------
rem Flat label-based control, not nested (...) blocks: multi-line parens inside parens break .bat.
:do_install
where uv >nul 2>nul || ( echo uv not found: https://docs.astral.sh/uv/ & goto fail )
call :normalize_extras || goto fail
rem Every uv call below is pinned with --python. uv's pip interface otherwise targets an active
rem VIRTUAL_ENV/CONDA_PREFIX ahead of the local .venv, which is how an unrelated venv activated in the
rem user's shell silently absorbed the install and had its packages replaced.
set "TARGET_PY=%VENV_PY%"
if "%USE_ACTIVE_ENV%"=="1" goto install_active_env
if defined FOREIGN_ENV echo NOTE: the environment active in this shell will NOT be modified:
if defined FOREIGN_ENV echo         %FOREIGN_ENV%
if defined FOREIGN_ENV echo       Installing into %VENV_DIR% instead ^(--use-active-env overrides this^).
if "%RECREATE%"=="1" goto install_recreate
rem uv venv refuses an existing environment and --clear would wipe manual installs, so a repeat
rem --install (say, to add an extra) has to reuse what is already there.
if exist "%VENV_PY%" ( echo Reusing the existing environment at %VENV_DIR% ^(--recreate rebuilds it from scratch^). & goto install_torch_args )
uv venv "%VENV_DIR%" || goto fail
goto install_torch_args

:install_recreate
echo Recreating %VENV_DIR% - anything installed into it by hand is lost.
uv venv --clear "%VENV_DIR%" || goto fail
goto install_torch_args

:install_active_env
if not defined ACTIVE_ENV ( echo --use-active-env needs an activated environment. & goto fail )
set "TARGET_PY=%ACTIVE_ENV%\Scripts\python.exe"
echo Installing into the active environment: %ACTIVE_ENV%

:install_torch_args
rem PyPI's default torch is CPU-only on Windows, so installing blind generates on the CPU ~100x
rem slower with no error. Which CUDA index is right depends on the card: Blackwell (sm_120) has no
rem wheels before cu128, while cu126 is the last index still built for Maxwell..Volta (sm_50..sm_70).
set "TORCH_ARGS="
if defined TORCH_CHOICE goto install_torch_index
where nvidia-smi >nul 2>nul || goto install_cpu
nvidia-smi -L >nul 2>nul || goto install_cpu
rem compute_cap reads "12.0" on an RTX 50-series card. A driver too old to know the query leaves
rem CAP_MAJOR unset and lands on cu126, which covers the widest range of what people actually own.
set "CAP_MAJOR="
for /f "usebackq tokens=1 delims=." %%c in (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul`) do call :cap_max %%c
set "TORCH_CHOICE=cu126"
if defined CAP_MAJOR if !CAP_MAJOR! GEQ 10 set "TORCH_CHOICE=cu130"

:install_torch_index
if /i "!TORCH_CHOICE!"=="cpu" goto install_cpu_forced
set "TORCH_URL=https://download.pytorch.org/whl/!TORCH_CHOICE!"
if /i "!TORCH_CHOICE:~0,4!"=="http" set "TORCH_URL=!TORCH_CHOICE!"
rem unsafe-best-match: torchao is on the CUDA index too, older there than our torchao>=0.14 pin on
rem some indexes; without this uv's first-index rule stops at that older copy instead of finding a
rem new enough one on PyPI. It also makes the +cuXXX local version outrank PyPI's plain one, which
rem is what pulls the CUDA build in rather than the CPU-only wheel PyPI serves on Windows.
rem no-sources-package: the pyproject pin names one fixed index, and the card decides here.
set "TORCH_ARGS=--extra-index-url !TORCH_URL! --index-strategy unsafe-best-match --no-sources-package torch"
echo NVIDIA GPU detected - installing the CUDA build of PyTorch (!TORCH_CHOICE!).
goto install_pkgs

:install_cpu_forced
echo Installing the default (CPU) build of PyTorch (--torch-index cpu).
goto install_pkgs

:install_cpu
set "TORCH_CHOICE=cpu"
echo No NVIDIA GPU detected - installing the default (CPU) build of PyTorch.

:install_pkgs
uv pip install --python "!TARGET_PY!" !TORCH_ARGS! -e ".[!EXTRAS!]" || goto fail
uv pip install --python "!TARGET_PY!" inline-studio-frontend >nul 2>nul && echo Installed the prebuilt web UI (inline-studio-frontend). || echo Note: inline-studio-frontend not installed; the UI will build from source or run API-only.
rem A CPU-only wheel on a GPU box is silent at runtime and ~100x slower, so say it here rather than
rem let it through: it can still happen if PyPI ever outranks the CUDA index on version.
if /i "!TORCH_CHOICE!"=="cpu" goto install_done
"!TARGET_PY!" -c "import importlib, importlib.util, sys; spec = importlib.util.find_spec('torch'); sys.exit(0 if spec is None or importlib.import_module('torch').version.cuda else 1)" 2>nul && goto install_done
echo WARNING: the torch that got installed is a CPU-ONLY build. Generation would run on the
echo          CPU, roughly 100x slower. Re-run with an explicit index, e.g.
echo          .\webui.bat --install --torch-index cu126

:install_done
echo Installed extras: !EXTRAS!. Start with: .\webui.bat
exit /b 0

rem Keeps the highest compute-capability major seen. Anything not a plain number is ignored: an
rem older driver answers an unknown query with an error string instead of failing outright.
:cap_max
echo(%~1| findstr /r /c:"^[0-9][0-9]*$" >nul || exit /b 0
if not defined CAP_MAJOR set "CAP_MAJOR=%~1"
if %~1 GTR %CAP_MAJOR% set "CAP_MAJOR=%~1"
exit /b 0

rem --- Pick the Python interpreter (and matching pip), in priority order -------------------------
rem Our own .venv outranks an env that merely happens to be activated, so a foreign venv can never
rem absorb the on-demand installs in :ensure_frontend / :ensure_smart_memory_deps.
:pick_python
if "%USE_ACTIVE_ENV%"=="1" goto pick_active_env
if exist "%VENV_PY%" goto pick_venv
if defined ACTIVE_ENV if exist "%ACTIVE_ENV%\Scripts\python.exe" ( set "PY=%ACTIVE_ENV%\Scripts\python.exe" & goto have_pip )
python -c "import inline_core" >nul 2>nul && ( set "PY=python" & goto have_pip )
echo No .venv found. Run  .\webui.bat --install  first.
goto fail

:pick_venv
set "PY=%VENV_PY%"
if defined FOREIGN_ENV echo NOTE: running from %VENV_DIR%, not the environment active in this shell ^(%FOREIGN_ENV%^).
goto have_pip

:pick_active_env
if not defined ACTIVE_ENV ( echo --use-active-env needs an activated environment. & goto fail )
set "PY=%ACTIVE_ENV%\Scripts\python.exe"

:have_pip
rem uv venvs are not seeded with pip, so 'python -m pip' can never work in one - go through uv instead.
set PIP="%PY%" -m pip install
where uv >nul 2>nul && set PIP=uv pip install --python "%PY%"
goto have_python

:have_python
if "%DEV_MODE%"=="1" ( call :run_dev & exit /b !ERRORLEVEL! )
if "%FORCE_REBUILD%"=="1" call :rebuild_frontend
if "%SMART_MEMORY%"=="1" call :ensure_smart_memory_deps

call :ensure_frontend

"%PY%" -m inline_core.server
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" ( echo. & echo Inline Studio exited with code %RC%. & pause )
exit /b %RC%

rem ================================ subroutines =================================================

:foreign_env
rem Sets FOREIGN_ENV when the environment active in this shell is NOT ours. Everything keys off this:
rem an env that merely happens to be activated must never absorb an install or a launch-time dep.
set "FOREIGN_ENV="
if not defined ACTIVE_ENV exit /b 0
for %%I in ("%ACTIVE_ENV%") do set "ACTIVE_FULL=%%~fI"
if /i "%ACTIVE_FULL%"=="%VENV_DIR%" exit /b 0
set "FOREIGN_ENV=%ACTIVE_FULL%"
exit /b 0

:normalize_extras
rem Validate and dedupe the extras, so a typo fails here with the valid list instead of reaching uv as
rem a raw resolver error. The names must match [project.optional-dependencies] in pyproject.toml.
set "EXTRAS_OUT="
for %%E in ("%EXTRAS:,=" "%") do call :add_extra %%E || exit /b 1
set "EXTRAS=!EXTRAS_OUT!"
exit /b 0

:add_extra
echo runtime server parallel training dev all | findstr /r /c:"\<%~1\>" >nul || goto bad_extra
if "!EXTRAS_OUT!"=="" ( set "EXTRAS_OUT=%~1" & exit /b 0 )
echo !EXTRAS_OUT! | findstr /r /c:"\<%~1\>" >nul && exit /b 0
set "EXTRAS_OUT=!EXTRAS_OUT!,%~1"
exit /b 0

:bad_extra
echo unknown extra: %~1 ^(valid: runtime, server, parallel, training, dev, all^)
exit /b 1

:frontend_available
rem Succeeds (errorlevel 0) when a web UI is resolvable; sets INLINE_FRONTEND_ROOT for a local build.
if defined INLINE_FRONTEND_ROOT if exist "%INLINE_FRONTEND_ROOT%\index.html" exit /b 0
"%PY%" -c "import os,sys; import inline_studio_frontend as f; sys.exit(0 if os.path.isfile(os.path.join(os.path.dirname(f.__file__),'static','index.html')) else 1)" >nul 2>nul && exit /b 0
if exist "..\dist-web\index.html" (
  for %%I in ("..\dist-web") do set "INLINE_FRONTEND_ROOT=%%~fI"
  exit /b 0
)
exit /b 1

:ensure_frontend
call :frontend_available && exit /b 0
echo No web UI found - installing the prebuilt package (inline-studio-frontend)...
%PIP% inline-studio-frontend >nul
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
echo          %PIP% inline-studio-frontend   once it's published.
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
"%PY%" -c "import torchao" >nul 2>nul && exit /b 0
echo Smart memory: installing torchao (int8 quantization)...
%PIP% torchao >nul ^
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
start "Inline Core (API)" "%PY%" -m inline_core.server
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
echo   --install              create .venv (via uv) and install, then exit. An existing .venv is
echo                          reused, and an unrelated environment activated in your shell is never
echo                          touched.
echo   --extra NAME           add an install extra (repeatable): runtime, parallel, server, training
echo   --torch-index WHICH    with --install, override the PyTorch wheel index picked from your GPU's
echo                          compute capability. A short name (cu130, cu128, cu126), a full index
echo                          URL, or "cpu" to force the CPU-only build. Also settable as
echo                          INLINE_TORCH_INDEX. Use cu128 on a Blackwell card whose driver
echo                          predates CUDA 13.
echo   --recreate             with --install, rebuild .venv from scratch (discards anything installed
echo                          into it by hand)
echo   --use-active-env       install into / run from the environment activated in this shell instead
echo                          of .venv
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
