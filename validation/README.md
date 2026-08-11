# Fundus AI validation pack

Workbook: `Fundus_AI_Validation.xlsx`

## RVO finish-up (Windows GPU host)

Data and weights live on the imaging box:

- Images: `C:\AI_Images\RVO`
- Model: `C:\AI_Images\trained_models\rvo_model.keras`

From this folder:

```bat
run_rvo_eval_ai_images.bat
```

Or:

```bat
python eval_rvo_ai_images.py --data-root C:\AI_Images\RVO --model C:\AI_Images\trained_models\rvo_model.keras --batch-size 64
python apply_rvo_results_to_workbook.py --json C:\AI_Images\RVO\eval_rvo_results.json
```

Quote **val** AUROC from the JSON, not `max(train,val)`.
