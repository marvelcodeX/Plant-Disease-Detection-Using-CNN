# Plant Disease Detection

A Flask web app for detecting plant diseases from leaf images using a PyTorch CNN model trained on PlantVillage-style classes. Users can create a local account, upload or capture a leaf image, receive confidence-ranked predictions, and review their private scan history from a dashboard.

This project is built upon the existing open-source repository [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection?tab=readme-ov-file). The current version updates the deployed Flask app with authentication, per-user scan history, top-3 prediction confidence, refreshed UI templates, and local runtime persistence.

## Features

- Classifies leaf images across 39 plant health and disease classes.
- Shows the primary prediction, confidence score, and top-3 ranked alternatives.
- Provides disease descriptions, prevention guidance, and supplement/fertilizer suggestions.
- Supports local user signup, login, logout, and protected AI Engine access.
- Saves each user's scan history to a local SQLite database.
- Includes a dashboard with total scans, healthy/diseased counts, average confidence, common predictions, and recent scans.
- Supports image upload and browser camera capture from the AI Engine.

## Project Structure

```text
.
|-- Flask Deployed App/
|   |-- app.py
|   |-- CNN.py
|   |-- disease_info.csv
|   |-- supplement_info.csv
|   |-- requirements.txt
|   |-- templates/
|   `-- static/uploads/
|-- Model/
|-- demo_images/
|-- test_images/
|-- .gitignore
`-- README.md
```

#`.pt` File Link

[Download .pt file from here](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A)

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv venv
source venv/bin/activate
cd "Flask Deployed App"
pip install -r requirements.txt
```

Download or copy the trained model file into the Flask app folder:

```text
Flask Deployed App/plant_disease_model_1_latest.pt
```

The model can be downloaded from the original project resources linked in `Model/Readme.md`, or replaced with your own checkpoint that matches the `CNN.CNN(39)` architecture.

For local development, set a Flask secret key before running the app:

```bash
export SECRET_KEY="replace-with-a-secure-local-secret"
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

## Runtime Notes

- The SQLite database is created automatically at `Flask Deployed App/plant_disease.db` when the app starts.
- Uploaded images are saved under `Flask Deployed App/static/uploads/`.
- Both locations are ignored by Git because they are runtime state, not source code.
- If you rename the model file, update `MODEL_PATH` in `Flask Deployed App/app.py`.

## Testing Images

Sample images are available in `test_images/`. Use them to validate that the prediction flow, result page, and dashboard history are working after setup.

## Credits

Base project and model workflow: [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection?tab=readme-ov-file)

Original blog reference: [Plant Disease Detection Using Convolutional Neural Networks with PyTorch](https://medium.com/analytics-vidhya/plant-disease-detection-using-convolutional-neural-networks-and-pytorch-87c00c54c88f)
