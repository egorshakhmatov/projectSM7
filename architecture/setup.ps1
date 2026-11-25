param([switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtimeDir = Join-Path $projectRoot '.runtime'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$logPath = Join-Path $runtimeDir 'setup.log'
$lockStream = $null

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    }
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE). See $logPath" }
}

function Test-Python {
    param([string]$Executable)
    $ErrorActionPreference = 'Continue'
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) { return $false }
    & $Executable -c "import sys, tkinter, venv; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

try {
    try {
        $lockStream = [System.IO.File]::Open((Join-Path $runtimeDir 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { throw 'SM7 is already running or preparing its environment. Use the existing window.' }
    Add-Content -LiteralPath $logPath -Value "`n$(Get-Date -Format s) Starting SM7" -Encoding UTF8
    Set-Location -LiteralPath $projectRoot
    $env:PYTHONUTF8 = '1'
    $env:PIP_CACHE_DIR = Join-Path $runtimeDir 'pip-cache'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PIP_REQUIRE_VIRTUALENV = 'true'
    $env:MPLCONFIGDIR = Join-Path $runtimeDir 'matplotlib'

    if (-not (Test-Python $venvPython)) {
        Write-Host 'Preparing SM7. First launch may need an internet connection.'
        $basePython = $null
        $localPython = Join-Path $runtimeDir 'Python312\python.exe'
        $candidates = @($localPython)
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            $detected = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += $detected }
        }
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) { $candidates += $pythonCommand.Source }
        foreach ($candidate in $candidates) {
            if (Test-Python $candidate) { $basePython = $candidate; break }
        }
        if (-not $basePython) {
            Write-Host 'Downloading Python from python.org...'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $installer = Join-Path $runtimeDir 'python-3.12.10-amd64.exe'
            Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
            $signature = Get-AuthenticodeSignature -LiteralPath $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                throw 'Python installer signature could not be verified. Installation stopped.'
            }
            $targetDir = Join-Path $runtimeDir 'Python312'
            $installArgs = @('/quiet', 'InstallAllUsers=0', "TargetDir=`"$targetDir`"", 'PrependPath=0', 'Include_launcher=0', 'Include_test=0', 'Include_doc=0', 'Include_pip=1', 'Include_tcltk=1', 'AssociateFiles=0', 'Shortcuts=0')
            $installProcess = Start-Process -FilePath $installer -ArgumentList $installArgs -WindowStyle Hidden -Wait -PassThru
            if ($installProcess.ExitCode -notin @(0,3010) -or -not (Test-Python $localPython)) {
                throw 'Python installation did not complete. Check internet access and setup.log.'
            }
            $basePython = $localPython
        }
        Write-Host 'Creating a private environment for SM7...'
        Invoke-CheckedPython -Executable $basePython -Arguments @('-m','venv',(Join-Path $projectRoot '.venv'))
    }
    & $venvPython (Join-Path $projectRoot 'environment.py') --check *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing SM7 components. Needed only on first launch or after an update.'
        $constraints = Join-Path $projectRoot 'requirements-windows.lock.txt'
        Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','install','--only-binary=:all:','--index-url','https://download.pytorch.org/whl/cpu','-c',$constraints,'-r',(Join-Path $projectRoot 'requirements-torch.txt'))
        Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','install','--only-binary=:all:','--index-url','https://pypi.org/simple','-c',$constraints,'-r',(Join-Path $projectRoot 'requirements.txt'))
        Invoke-CheckedPython -Executable $venvPython -Arguments @((Join-Path $projectRoot 'environment.py'),'--check')
    }
    Write-Host 'SM7 is ready.'
    if (-not $PrepareOnly) {
        $pythonWindowless = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
        $appProcess = Start-Process -FilePath $pythonWindowless -ArgumentList @("`"$(Join-Path $projectRoot 'gui.py')`"") -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru
        if ($appProcess.ExitCode -ne 0) { throw 'The application stopped. See .runtime/app-error.log.' }
    }
} catch {
    $failure = $_.Exception.Message
    Add-Content -LiteralPath $logPath -Value "ERROR: $failure" -Encoding UTF8
    Write-Host "SM7 could not start: $failure"
    if (-not $PrepareOnly) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show("SM7 could not start.`n$failure`n`nDetails: $logPath", 'SM7') | Out-Null
    }
    exit 1
} finally {
    if ($lockStream) { $lockStream.Dispose() }
}
