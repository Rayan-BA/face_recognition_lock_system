from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

# prob not optimal
torch.set_num_threads(cpu_count() - 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()

def collate_fn(x):
    return x[0]

def createEmbeddings(data:str, save_to_path:str):
    print("[INFO] Creating new embeddings...")
    dataset = datasets.ImageFolder(data)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)
    
    aligned = []
    names = []
    for x, y in tqdm(loader):
        x_aligned = mtcnn(x)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    np.savez_compressed(save_to_path, embeddings, names)
    print("[INFO] Embedding done.")
