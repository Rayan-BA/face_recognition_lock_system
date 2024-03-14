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

def create_embeddings(dataset_path, embeddings_path, skip_existing_labels=True):
    print("[INFO] Creating new embeddings...")
    dataset = datasets.ImageFolder(dataset_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)

    try:
        embed = np.load(embeddings_path)
        existing_embedded_data, existing_labels = embed["arr_0"], embed["arr_1"]
    except:
        existing_embedded_data, existing_labels = [], []
    
    # print(existing_labels);exit()
    aligned = []
    new_labels = []
    for x, y in tqdm(loader):
        if skip_existing_labels and np.isin(dataset.idx_to_class[y], dataset.classes): continue
        x_aligned = mtcnn(x)
        if x_aligned is not None:
            aligned.append(x_aligned)
            new_labels.append(dataset.idx_to_class[y])

    if len(aligned) > 0:
        aligned = torch.stack(aligned).to(device)
        new_embeddings = resnet(aligned).detach().cpu()
        
        if skip_existing_labels:
            combined_embeddings = np.concatenate([existing_embedded_data, new_embeddings])
            combined_labels = np.concatenate([existing_labels, new_labels])
            np.savez(embeddings_path, combined_embeddings, combined_labels)
        else:
            np.savez(embeddings_path, new_embeddings, new_labels)

    print("[INFO] Embedding done.")
