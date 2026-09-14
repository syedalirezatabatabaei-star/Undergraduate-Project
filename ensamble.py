import torch

from model.modelm3 import CNN1M3
from model.modelm5 import CNN1M5
from model.modelm7 import CNN1M7

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class EnsembleModel:

    def __init__(self):

        self.m3 = CNN1M3().to(device)
        self.m5 = CNN1M5().to(device)
        self.m7 = CNN1M7().to(device)


        self.m3.load_state_dict(
            torch.load("/alireza/models/M3.pth",
                       map_location=device)
        )

        self.m5.load_state_dict(
            torch.load("/alireza/models/M5.pth",
                       map_location=device)
        )

        self.m7.load_state_dict(
            torch.load("/alireza/models/M7.pth",
                       map_location=device)
        )


        self.m3.eval()
        self.m5.eval()
        self.m7.eval()


    def predict(self, image):

        with torch.no_grad():

            out_m3 = self.m3(image)
            out_m5 = self.m5(image)
            out_m7 = self.m7(image)


            prob_m3 = torch.softmax(out_m3, dim=1)
            prob_m5 = torch.softmax(out_m5, dim=1)
            prob_m7 = torch.softmax(out_m7, dim=1)


            final_prob = (
                prob_m3 +
                prob_m5 +
                prob_m7
            ) / 3


            prediction = torch.argmax(
                final_prob,
                dim=1
            )
            prediction1 = torch.argmax(prob_m3, dim=1)
            prediction2 = torch.argmax(prob_m5, dim=1)
            prediction3 = torch.argmax(prob_m7, dim=1)

            print("M3:", prediction1.item())
            print("M5:", prediction2.item())
            print("M7:", prediction3.item())


        return prediction.item(), final_prob
if __name__ == "__main__":

    model = EnsembleModel()

    print("All models loaded successfully")
