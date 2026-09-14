from PIL import Image, ImageOps
from torchvision import transforms


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Lambda(lambda image: ImageOps.invert(image)),

    transforms.Resize((28, 28)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


def preprocess_image(path):

    image = Image.open(path)

    image = transform(image)
    image = image.unsqueeze(0)

    return image
