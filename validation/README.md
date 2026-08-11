# Fundus AI validation pack

Workbook: `Fundus_AI_Validation.xlsx`

**Status:** incomplete until RVO metrics are scored. All other headline AUROCs are filled.

## RVO finish-up (Windows GPU host)

You are usually in `C:\source_controls\development_sanele` — there is **no** local `validation\` folder until you pull/download it.

### One-paste (recommended)

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/tashen247/diabetes-api/cursor/fundus-ai-validation-finish-a330/validation/finish_rvo.ps1 | iex
```

That downloads scripts into `C:\source_controls\development_sanele\validation`, scores:

- Images: `C:\AI_Images\RVO`
- Model: `C:\AI_Images\trained_models\rvo_model.keras`

and writes AUROC into the workbook.

### Or single Python file in your current folder

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/tashen247/diabetes-api/cursor/fundus-ai-validation-finish-a330/validation/finish_rvo_validation.py -OutFile finish_rvo_validation.py
pip install "tensorflow==2.12.0" "keras==2.12.0" scikit-learn pillow numpy openpyxl
python finish_rvo_validation.py
```

Quote **val** AUROC from the JSON, not `max(train,val)`.
