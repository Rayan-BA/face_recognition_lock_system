from facenet_pytorch import MTCNN, InceptionResnetV1
# from facenet_pytorch.test import tests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

# prob not optimal
torch.set_num_threads(round(cpu_count() * 0.8))

class FaceEmbeddingGenerator:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2", device=self.device).eval()
        self.batch_size = batch_size
    
    def remove_embed(self, target):
        print("[INFO] Removing an embed...")
        embeds = np.load("./models/torch_embeddings.npz")
        existing_embedded_data, existing_labels = embeds["arr_0"], embeds["arr_1"]
        # for i in existing_embedded_data:
        #     print(i)
        del_indecies = []
        for i, label in enumerate(existing_labels):
            if label == target: del_indecies.append(i)
        if len(del_indecies) > 0:
            existing_labels = np.delete(existing_labels, del_indecies)
            existing_embedded_data = np.delete(existing_embedded_data, del_indecies, axis=0)
            # for i in existing_embedded_data:
            #     print(i)
            np.savez("./models/torch_embeddings.npz", existing_embedded_data, existing_labels)
            print("[INFO] Embed removed.")
        else: print("[INFO] Target not found.")
    
    def create_embeddings(self, skip_existing_labels=True):
        print("[INFO] Creating new embeddings...")
        dataset = datasets.ImageFolder(self.dataset)
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=lambda x: x, batch_size=self.batch_size, shuffle=True)
        
        try:
            embeds = np.load("./models/torch_embeddings.npz")
            existing_embedded_data, existing_labels = embeds["arr_0"], embeds["arr_1"]
            # print(existing_embedded_data)
            # print(dataset.classes)
            # exit()
        except:
            existing_embedded_data, existing_labels = np.asarray([]), []

        aligned , new_labels = [], []
        for batch in tqdm(loader):
            for x, y in tqdm(batch):
                if skip_existing_labels and np.isin(dataset.idx_to_class[y], dataset.classes, assume_unique=True): continue
                batch_x_aligned = self.mtcnn(x)
                if batch_x_aligned is not None:
                    aligned.append(batch_x_aligned)
                    new_labels.append(dataset.idx_to_class[y])
        if len(aligned) > 0:
            aligned = torch.stack(aligned).to(self.device)
            new_embeddings = np.asarray(self.resnet(aligned).detach())
            if skip_existing_labels:
                if len(existing_embedded_data) == 0: existing_embedded_data = np.empty((0, new_embeddings.shape[1]))
                # else: existing_embedded_data = np.expand_dims(existing_embedded_data, axis=0)
                # print(existing_embedded_data)
                # print(new_embeddings)
                # for i in new_embeddings:
                #     print(i)
                print(existing_embedded_data.shape)
                print(new_embeddings.shape)
                combined_embeddings = np.concatenate([existing_embedded_data, new_embeddings])
                combined_labels = np.concatenate([existing_labels, new_labels])
                np.savez("./models/torch_embeddings.npz", combined_embeddings, combined_labels)
            else:
                np.savez("./models/torch_embeddings.npz", new_embeddings, new_labels)
        
        print("[INFO] Embedding done.")

if __name__ == "__main__":
    FaceEmbeddingGenerator("./tmp").create_embeddings(skip_existing_labels=True)
    # FaceEmbeddingGenerator("./tmp").remove_embed("Meli")

