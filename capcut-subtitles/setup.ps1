# CapCut 자막 도구 설치 스크립트 (1-setup.bat 이 실행합니다)
# 모든 파일은 이 폴더 안에만 설치됩니다. 폴더를 지우면 깨끗하게 삭제돼요.

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root  = $PSScriptRoot
$Tools = Join-Path $Root 'tools'
$Venv  = Join-Path $Root '.venv'
$Py    = Join-Path $Venv 'Scripts\python.exe'
$Uv    = Join-Path $Tools 'uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
$env:UV_CACHE_DIR          = Join-Path $Tools 'cache'
$env:UV_LINK_MODE          = 'copy'

function Fail([string]$msg) {
    Write-Host ''
    Write-Host "설치 실패: $msg" -ForegroundColor Red
    Write-Host '인터넷 연결과 디스크 여유 공간(약 6GB)을 확인하고 1-setup.bat 을 다시 실행해 보세요.'
    exit 1
}

Write-Host ''
Write-Host '=== CapCut 자막 도구 설치 ===' -ForegroundColor Cyan
Write-Host '처음 한 번만 하면 됩니다. 인터넷 속도에 따라 10~30분 걸려요.'
Write-Host ''

# 1) 설치 도구 uv (파이썬과 라이브러리를 이 폴더 안에 설치해 줌)
if (Test-Path $Uv) {
    Write-Host '[1/4] 설치 도구 준비됨'
} else {
    Write-Host '[1/4] 설치 도구(uv) 내려받는 중...'
    try {
        New-Item -ItemType Directory -Force -Path $Tools | Out-Null
        $zip = Join-Path $Tools 'uv.zip'
        $tmp = Join-Path $Tools 'uv-tmp'
        Invoke-WebRequest -UseBasicParsing -ErrorAction Stop `
            -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' `
            -OutFile $zip
        Expand-Archive -Force -ErrorAction Stop -Path $zip -DestinationPath $tmp
        $found = Get-ChildItem -Path $tmp -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { Fail 'uv.exe 를 찾지 못했습니다.' }
        Copy-Item -ErrorAction Stop $found.FullName $Uv
        Remove-Item -Recurse -Force $zip, $tmp
    } catch {
        Fail $_.Exception.Message
    }
}

# 2) 파이썬 3.12 + 전용 가상환경
if (Test-Path $Py) {
    Write-Host '[2/4] 파이썬 준비됨'
} else {
    Write-Host '[2/4] 파이썬 3.12 설치 중...'
    if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    & $Uv venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Py)) { Fail '파이썬을 설치하지 못했습니다.' }
}

# 3) 음성 인식 라이브러리 (+ NVIDIA 그래픽카드가 있으면 GPU 가속 라이브러리)
Write-Host '[3/4] 음성 인식 프로그램 설치 중...'
& $Uv pip install --python $Py -r (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail '음성 인식 프로그램을 설치하지 못했습니다.' }

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host '      NVIDIA 그래픽카드 감지 → GPU 가속 라이브러리 설치 (약 1.3GB)'
    & $Uv pip install --python $Py 'nvidia-cublas-cu12==12.9.2.10' 'nvidia-cudnn-cu12==9.10.2.21'
    if ($LASTEXITCODE -ne 0) {
        Write-Host '      GPU 라이브러리 설치에 실패해서 CPU로 실행합니다. (느리지만 정확도는 거의 같아요)' -ForegroundColor Yellow
    }
} else {
    Write-Host '      NVIDIA 그래픽카드가 없어서 CPU로 실행합니다. (느리지만 정확도는 거의 같아요)'
}

# 4) 인식 모델 내려받기 + 실제로 돌아가는지 확인
Write-Host '[4/4] 인식 모델(Whisper large-v3, 약 3GB) 내려받고 점검하는 중...'
& $Py -X utf8 (Join-Path $Root 'make_subtitles.py') --check
if ($LASTEXITCODE -ne 0) { Fail '인식 모델을 준비하지 못했습니다.' }

# 설치용 임시 파일 정리 (디스크 공간 확보)
& $Uv cache clean 2>$null | Out-Null

Write-Host ''
Write-Host '설치 완료!' -ForegroundColor Green
Write-Host '이제 영상 파일을 2-make-subtitles.bat 위에 끌어다 놓으면 자막이 만들어집니다.'
exit 0
