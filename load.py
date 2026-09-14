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


'''
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load ensemble
model = EnsembleModel()


def predict_image(path):

    image = Image.open(path)

    image = transform(image)

    # اضافه کردن batch dimension
    image = image.unsqueeze(0).to(device)


    prediction, probability = model.predict(image)


    return prediction, probability



if __name__ == "__main__":

    result, prob = predict_image("test.jpg")
    #image = Image.open(path)

    #print(image.mode)
    #print(image.size)

    #image.show()

    print("Prediction:", result)
    print("Confidence:", prob.max().item())
    '''
