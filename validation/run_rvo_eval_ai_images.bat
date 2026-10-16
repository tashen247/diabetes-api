@echo off
REM Score RVO on C:\AI_Images\RVO for Fundus_AI_Validation.xlsx
setlocal
cd /d "%~dp0"

echo Installing deps (keras/tensorflow/sklearn/pillow)...
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.12.0" "keras==2.12.0" scikit-learn pillow numpy

echo.
echo Running RVO eval (data=C:\AI_Images\RVO, model=C:\AI_Images\trained_models\rvo_model.keras)...
python eval_rvo_ai_images.py --data-root "C:\AI_Images\RVO" --model "C:\AI_Images\trained_models\rvo_model.keras" --batch-size 64 --out "C:\AI_Images\RVO\eval_rvo_results.json"

echo.
echo Done. Paste C:\AI_Images\RVO\eval_rvo_results.json into the agent chat
echo or run: python apply_rvo_results_to_workbook.py --json "C:\AI_Images\RVO\eval_rvo_results.json"
endlocal
