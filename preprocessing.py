from PIL import Image, ImageOps
from torchvision import transforms


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    # Frontend:
    # black digit on white background
    #
    # MNIST:
    # white digit on black background
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

    # [1, 28, 28]
    #       ↓
    # [1, 1, 28, 28]
    image = image.unsqueeze(0)

    return image
'''
from PIL import Image, ImageOps
import torch
from torchvision import transforms


# =========================
# MNIST configuration
# =========================

MNIST_SIZE = 28

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_image(path):
    """
    Load image and convert it to grayscale.

    Supports RGB, RGBA, JPG, PNG, etc.
    """

    image = Image.open(path)

    # اگر تصویر RGBA باشد، روی پس‌زمینه سفید قرار می‌دهیم
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image)

    image = image.convert("L")

    return image


def invert_image(image):
    """
    Convert black digit on white background
    to white digit on black background.

    Example:

    Before:
        background = white
        digit      = black

    After:
        background = black
        digit      = white
    """

    return ImageOps.invert(image)


def find_digit_bbox(image, threshold=20):
    """
    Find the bounding box of the digit.

    Pixels brighter than threshold are considered
    part of the digit.

    Returns:
        bbox = (left, top, right, bottom)

    If no digit is detected, returns None.
    """

    # Binary mask
    binary = image.point(
        lambda pixel: 255 if pixel > threshold else 0
    )

    bbox = binary.getbbox()

    return bbox


def crop_digit(image, bbox, padding=2):
    """
    Crop the digit and add a small amount of padding.
    """

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    return image.crop((left, top, right, bottom))


def resize_keep_aspect(image, target_size=20):
    """
    Resize digit while preserving its aspect ratio.

    The resulting image will fit inside target_size x target_size.
    """

    width, height = image.size

    if width == 0 or height == 0:
        raise ValueError("Invalid digit image size.")

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def center_on_mnist_canvas(
    image,
    canvas_size=28
):
    """
    Put the digit in the center of a 28x28 black canvas.
    """

    canvas = Image.new(
        "L",
        (canvas_size, canvas_size),
        color=0
    )

    width, height = image.size

    left = (canvas_size - width) // 2
    top = (canvas_size - height) // 2

    canvas.paste(image, (left, top))

    return canvas


def preprocess_image(
    path,
    invert=True,
    threshold=20,
    padding=2,
    digit_size=20
):
    """
    Complete preprocessing pipeline for MNIST CNN inference.

    Input:
        path: path to input image

    Output:
        Tensor with shape:

        [1, 1, 28, 28]

    """

    # -------------------------
    # 1. Load image
    # -------------------------

    image = load_image(path)

    # -------------------------
    # 2. Invert colors
    # -------------------------

    if invert:
        image = invert_image(image)

    # -------------------------
    # 3. Find digit
    # -------------------------

    bbox = find_digit_bbox(
        image,
        threshold=threshold
    )

    # اگر هیچ عددی پیدا نشد
    if bbox is None:
        raise ValueError(
            "No digit was detected in the image."
        )

    # -------------------------
    # 4. Crop digit
    # -------------------------

    digit = crop_digit(
        image,
        bbox,
        padding=padding
    )

    # -------------------------
    # 5. Resize while preserving
    #    aspect ratio
    # -------------------------

    digit = resize_keep_aspect(
        digit,
        target_size=digit_size
    )

    # -------------------------
    # 6. Center digit in 28x28
    # -------------------------

    image = center_on_mnist_canvas(
        digit,
        canvas_size=MNIST_SIZE
    )

    # -------------------------
    # 7. Convert to Tensor
    # -------------------------

    image = transforms.ToTensor()(image)

    # -------------------------
    # 8. Normalize exactly like MNIST
    # -------------------------

    image = transforms.Normalize(
        (MNIST_MEAN,),
        (MNIST_STD,)
    )(image)

    # -------------------------
    # 9. Add batch dimension
    # -------------------------

    image = image.unsqueeze(0)

    return image


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    image_tensor = preprocess_image(
        "test.jpg"
    )

    print("Tensor shape:", image_tensor.shape)
    print("Tensor dtype:", image_tensor.dtype)
    print(
        "Tensor min:",
        image_tensor.min().item()
    )
    print(
        "Tensor max:",
        image_tensor.max().item()
    )
    
    '''