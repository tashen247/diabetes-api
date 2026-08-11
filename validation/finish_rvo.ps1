# Finish RVO row in Fundus_AI_Validation.xlsx using C:\AI_Images\RVO
# Run from ANY directory in PowerShell on the Windows GPU host:
#   irm https://raw.githubusercontent.com/tashen247/diabetes-api/cursor/fundus-ai-validation-finish-a330/validation/finish_rvo.ps1 | iex
# Or save/run locally:
#   powershell -ExecutionPolicy Bypass -File finish_rvo.ps1

$ErrorActionPreference = "Stop"
$Branch = "cursor/fundus-ai-validation-finish-a330"
$Base = "https://raw.githubusercontent.com/tashen247/diabetes-api/$Branch/validation"
$Dest = "C:\source_controls\development_sanele\validation"
$DataRoot = "C:\AI_Images\RVO"
$Model = "C:\AI_Images\trained_models\rvo_model.keras"
$JsonOut = "C:\AI_Images\RVO\eval_rvo_results.json"

Write-Host "Creating $Dest"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null

$files = @(
  "eval_rvo_ai_images.py",
  "apply_rvo_results_to_workbook.py",
  "Fundus_AI_Validation.xlsx",
  "run_rvo_eval_ai_images.bat",
  "README.md"
)

foreach ($f in $files) {
  $url = "$Base/$f"
  $out = Join-Path $Dest $f
  Write-Host "Downloading $f ..."
  Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
}

if (-not (Test-Path $Model)) {
  throw "Model not found: $Model"
}
if (-not (Test-Path $DataRoot)) {
  throw "Data root not found: $DataRoot"
}

Write-Host "Installing Python deps (tensorflow/keras 2.12, sklearn, pillow)..."
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.12.0" "keras==2.12.0" scikit-learn pillow numpy openpyxl

Set-Location $Dest
Write-Host "Scoring RVO on $DataRoot ..."
python .\eval_rvo_ai_images.py --data-root $DataRoot --model $Model --batch-size 64 --out $JsonOut

if (-not (Test-Path $JsonOut)) {
  throw "Expected results JSON missing: $JsonOut"
}

Write-Host "Applying metrics into workbook..."
python .\apply_rvo_results_to_workbook.py --json $JsonOut --workbook (Join-Path $Dest "Fundus_AI_Validation.xlsx")

Write-Host ""
Write-Host "DONE. Updated workbook:"
Write-Host "  $(Join-Path $Dest 'Fundus_AI_Validation.xlsx')"
Write-Host "Results JSON:"
Write-Host "  $JsonOut"
Write-Host "Open the Summary sheet RVO row and confirm AUROC is filled."
