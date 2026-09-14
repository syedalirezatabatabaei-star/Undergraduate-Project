import torch

from ensamble import EnsembleModel
from preprocessing import preprocess_image


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



model = EnsembleModel()


def predict_image(path):

    image = preprocess_image(
        path,
        #invert=True
    )

    image = image.to(device)

    prediction, probability = model.predict(image)

    return prediction, probability


if __name__ == "__main__":

    result, prob = predict_image("test.jpg")

    print("Prediction:", result)
    print("Confidence:", prob.max().item())


