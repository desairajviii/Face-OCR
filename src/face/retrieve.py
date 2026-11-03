import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)



def embed_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device))
    return embedding.squeeze(0).cpu()