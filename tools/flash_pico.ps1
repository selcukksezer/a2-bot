# Gatherion Pico flash helper (UF2 drag / picotool / arduino-cli).
# Does NOT claim success without hardware. No UAC / runas.
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\flash_pico.ps1
#   powershell -ExecutionPolicy Bypass -File tools\flash_pico.ps1 -WaitCom -SetHid

param(
    [switch]$WaitCom,
    [switch]$SetHid,
    [int]$TimeoutSec = 90,
    [string]$Uf2Path = ""
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

$Sketch = Join-Path $Root "vendor\usb_hid\pico\GatherionHidKeyboard"
$Uf2Dir = Join-Path $Root "vendor\usb_hid\pico"
Write-Host "Gatherion Pico flash helper"
Write-Host "Sketch: $Sketch"
Write-Host ""

function Find-RpiRp2Drive {
    Get-Volume -ErrorAction SilentlyContinue |
        Where-Object { $_.FileSystemLabel -eq "RPI-RP2" } |
        ForEach-Object {
            $letter = $_.DriveLetter
            if ($letter) { "{0}:\" -f $letter }
        }
}

function Find-Uf2 {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) { return (Resolve-Path $Explicit).Path }
    $hits = Get-ChildItem -Path $Uf2Dir -Filter "*.uf2" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if ($hits) { return $hits[0].FullName }
    return $null
}

$uf2 = Find-Uf2 -Explicit $Uf2Path
$rp2 = @(Find-RpiRp2Drive)

# 1) UF2 drag-drop when BOOTSEL drive is present
if ($rp2.Count -gt 0) {
    $drive = $rp2[0]
    Write-Host "BOOTSEL drive detected: $drive"
    if ($uf2) {
        Write-Host "Copying UF2 → $drive"
        Copy-Item -LiteralPath $uf2 -Destination $drive -Force
        Write-Host "UF2 copied. Pico should reboot into firmware."
    } else {
        Write-Host "No .uf2 under vendor\usb_hid\pico\ yet."
        Write-Host "Build once with Arduino IDE (Philhower Pico core) → Export Compiled Binary,"
        Write-Host "place the .uf2 next to the sketch folder, re-run this script in BOOTSEL mode."
        Write-Host "Or use Arduino IDE Upload while holding BOOTSEL on first flash."
    }
}
else {
    Write-Host "No RPI-RP2 (BOOTSEL) drive. Normal run modes:"
    # 2) picotool if available + uf2
    $picotool = Get-Command picotool -ErrorAction SilentlyContinue
    if ($picotool -and $uf2) {
        Write-Host "Trying: picotool load -f `"$uf2`""
        & picotool load -f $uf2
        if ($LASTEXITCODE -eq 0) {
            Write-Host "picotool load OK"
        } else {
            Write-Host "picotool failed (exit $LASTEXITCODE) — hold BOOTSEL and retry, or use Arduino IDE."
        }
    }
    elseif ($picotool -and -not $uf2) {
        Write-Host "picotool found but no .uf2 staged. Export UF2 from Arduino IDE first."
    }

    # 3) arduino-cli upload if available
    $acli = Get-Command arduino-cli -ErrorAction SilentlyContinue
    if ($acli) {
        Write-Host "arduino-cli found. Attempting compile+upload (board may need COM / BOOTSEL)…"
        & arduino-cli compile --fqbn "rp2040:rp2040:rpipico" $Sketch
        if ($LASTEXITCODE -eq 0) {
            & arduino-cli upload -p auto --fqbn "rp2040:rp2040:rpipico" $Sketch
        } else {
            Write-Host "arduino-cli compile failed — install Philhower core or use Arduino IDE."
        }
    }

    if (-not $picotool -and -not $acli -and -not $uf2) {
        Write-Host ""
        Write-Host "Manual first flash:"
        Write-Host "  1) Hold BOOTSEL, plug Pico USB → RPI-RP2 drive appears"
        Write-Host "  2) Arduino IDE → Philhower Pico core → open GatherionHidKeyboard.ino → Upload"
        Write-Host "  3) Or drop a built .uf2 onto RPI-RP2"
        Write-Host "Docs: docs\PICO_OPERATOR_TR.md"
    }
}

if ($WaitCom -or $SetHid) {
    Write-Host ""
    Write-Host "Waiting for new COM after flash/plug…"
    $py = Join-Path $Root "tools\set_hid_com.py"
    $args = @($py, "--wait-plug", "--timeout", "$TimeoutSec")
    if (-not $SetHid) { $args += "--dry-run" }
    & python @args
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Next: python tools\set_hid_com.py --wait-plug"
Write-Host "Then: python tools\hid_com_smoke.py --port COMx --keys tab,1,2,3"
exit 0
