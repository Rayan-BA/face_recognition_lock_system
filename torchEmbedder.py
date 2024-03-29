from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

# prob not optimal
torch.set_num_threads(round(cpu_count() * 0.8))

class FaceEmbeddingGenerator:
    def __init__(self, dataset="./dataset", skip_existing_labels=True, batch_size=32, device="cpu"):
        self.dataset = dataset
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2", device=self.device).eval()
        self.skip_existing_labels = skip_existing_labels
        self.batch_size = batch_size
    
    def create_embeddings(self):
        print("[INFO] Creating new embeddings...")
        dataset = datasets.ImageFolder(self.dataset)
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=lambda x: x, batch_size=self.batch_size, shuffle=True)

        try:
            embed = np.load("./models/torch_embeddings.npz")
            existing_embedded_data, existing_labels = embed["arr_0"], embed["arr_1"]
        except:
            existing_embedded_data, existing_labels = [], []

        aligned = []
        new_labels = []
        for batch in tqdm(loader):
            for x, y in tqdm(batch):
                if self.skip_existing_labels and np.isin(dataset.idx_to_class[y], dataset.classes): continue
                batch_x_aligned = self.mtcnn(x)
                if batch_x_aligned is not None:
                    aligned.append(batch_x_aligned)
                    new_labels.append(dataset.idx_to_class[y])
        if len(aligned) > 0:
            aligned = torch.stack(aligned).to(self.device)
            new_embeddings = self.resnet(aligned).detach().cpu()

            if self.skip_existing_labels:
                combined_embeddings = np.concatenate([existing_embedded_data, new_embeddings])
                combined_labels = np.concatenate([existing_labels, new_labels])
                np.savez("./models/torch_embeddings.npz", combined_embeddings, combined_labels)
            else:
                np.savez("./models/torch_embeddings.npz", new_embeddings, new_labels)

        print("[INFO] Embedding done.")

if __name__ == "__main__":
    FaceEmbeddingGenerator(skip_existing_labels=False).create_embeddings()
