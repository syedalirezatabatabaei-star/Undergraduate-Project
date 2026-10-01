<p align="center">
  <a href="README.md">English</a>
  &nbsp; | &nbsp;
  <a href="README.fa.md">فارسی</a>
</p>
## Technologies

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://skillicons.dev/icons?i=python" width="50" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://skillicons.dev/icons?i=fastapi" width="50" alt="FastAPI">
  </a>
   <a href="https://pytorch.org/">
    <img src="https://skillicons.dev/icons?i=pytorch" width="50" alt="PyTorch">
  </a>
</p>

<p align="center">
  <code>TorchVision</code>
  <code>CNN</code>
  <code>MNIST</code>
  <code>PyTorch Lightning</code>
</p>
# Handwritten Digit Recognition

An undergraduate project for handwritten digit recognition using the MNIST dataset.

The project includes image preprocessing, model loading, prediction, an API, and a frontend interface for interacting with the trained models.

## Dataset

The project uses the MNIST handwritten digit dataset.

* 10 classes: 0–9
* Grayscale images
* Image size: 28 × 28 pixels

## Project Structure

```text
Undergraduate-Project/
│
├── Frontend/
├── model/
├── models/
│
├── api.py
├── preprocessing.py
├── load.py
├── ensamble.py
│
├── README.md
└── README.fa.md
```

## Main Files

### `api.py`

Provides the API used to receive input data and return model predictions.

### `preprocessing.py`

Contains the preprocessing steps required before an image is passed to the model.

### `load.py`

Handles loading the required data and model resources.

### `ensamble.py`

Contains the ensemble-related implementation used in the project.

### `model/`

Contains model-related code and components.

### `models/`

Contains the model files and related resources.

### `Frontend/`

Contains the frontend part of the application.

## Workflow

The general prediction workflow is:

```text
Input Image
     |
     v
Preprocessing
     |
     v
Trained Model
     |
     v
Prediction
     |
     v
Digit (0-9)
```

## Technologies

* Python
* Machine Learning
* Deep Learning
* Computer Vision
* MNIST
* API
* Frontend

## Installation

Clone the repository:

```bash
git clone https://github.com/syedalirezatabatabaei-star/Undergraduate-Project.git
cd Undergraduate-Project
```

Install the required dependencies for the project environment.

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

## Usage

The project can be used to preprocess a handwritten digit image and pass it to the trained model for classification.

The API provides the interface between the frontend and the prediction system.

## Academic Information

**Project:** Undergraduate Project — Handwritten Digit Recognition

**Student:** Seyed Alireza Tabatabaei

**Supervisor:** Dr. Sanaz Asdinia

**Advisor:** Dr. Hamidreza Sadrarhami

## Future Work

* Improve model performance
* Compare different model architectures
* Add prediction confidence
* Improve the frontend
* Add automated tests
* Add API documentation
* Containerize the application
* Deploy the application

## License

This project was developed for academic purposes.
