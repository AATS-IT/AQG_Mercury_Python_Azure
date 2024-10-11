# AQG Adult Cardiac Model Inference Service

Containerized inference endpoint for CSV-based inference on 8 AQG models trained on fields available in the Mercury Biostats dataset.

## Response Variables (Y)

- opdead: Operative mortality.
- losopl6: Length of stay >6 days postoperatively.
- losopg14: Length of stay >14 days postoperatively.
- ev_print: Major morbidity event.
- ev_dsinf: Deep sternal wound infection event.
- ev_strok: Stroke event.
- ev_reop: Reoperation event.
- ev_renal: Renal failure event.
- ev_mm: Composite of major morbidity and mortality.

## Folder Structure Required

Create a project folder with the following structure:

    /my_inference_api
    │
    ├── /models            # Folder to store your trained models (.pkl)
    │   └── all_lite_classweighting_model.pkl
    │   └── class_prevalence.csv
    ├── /temp_files        # Folder to store uploaded files temporarily
    │
    ├── Dockerfile         # Dockerfile to define how to build the image
    ├── inference_api.py   # FastAPI script with the inference logic
    ├── requirements.txt   # Python dependencies

    •	Models: Place your trained models (all_lite_classweighting_model.pkl and class_prevalence.csv) in the /models folder.
    •	Temporary Files: The folder /temp_files is used to store temporarily uploaded files during inference. The API clears out temp files after processing.
