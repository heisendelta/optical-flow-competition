from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torchvision import transforms as tf
import numpy as np

class DimensionalityReduction:
    def __init__(self, n_components, scaler=MinMaxScaler(), red_technique='pca'):
        self.scaler = scaler # if None, then doesn't perform any scaling
        self.n_components = n_components

        match red_technique:
            case 'pca':
                self.technique = PCA(n_components=n_components)
            case 'sparsepca':
                self.technique = SparsePCA(n_components=n_components)
            case 'tsvd':
                self.technique = TruncatedSVD(n_components=n_components)
            case _:
                raise NotImplementedError

    def __call__(self, image_tensor: torch.Tensor):
        # this is for a single image, not a batch of images
        C, H, W = image_tensor.shape

        tensor_reshaped = image_tensor.cpu().numpy().reshape(C, -1).T
        if self.scaler is not None:
            tensor_reshaped = self.scaler.fit_transform(tensor_reshaped)

        reduced_tensor_reshaped = self.technique.fit_transform(tensor_reshaped)
        reduced_tensor_reshaped = reduced_tensor_reshaped.T.reshape(self.n_components, H, W)

        reduced_tensor = torch.tensor(reduced_tensor_reshaped, device=image_tensor.device)
        return reduced_tensor
    
# Needs to have 3 channels or color passed in
# runs into the problem of loss of information in dimensionality reduction
class HistogramEqualization:
    def __call__(self, img):
        return tf.functional.equalize(img)
    
class CombinedTransform:
    def __init__(self, transform=tf.Compose([ tf.ToTensor() ])):
        self.transform = transform

    def __call__(self, flow_dict):
        seed = np.random.randint(2147483647)

        # Just in case I decide to add more feature columns
        feaure_flow_columns = [c for c in flow_dict.keys() if 'event_volume' in c]

        is_train: bool = 'flow_gt' in flow_dict.keys() # Only preprocess for train data

        if is_train:
            for col in feaure_flow_columns:
                torch.manual_seed(seed)

                if type(flow_dict[col]) == list:
                    flow_dict[col] = [ self.transform(img) for img in flow_dict[col] ]
                else:
                    flow_dict[col] = self.transform(flow_dict[col])

        return flow_dict

def combined_transform():
    return CombinedTransform(
        transform=tf.Compose([
    #         tf.GaussianBlur(kernel_size=(5, 5)),
            tf.RandomResizedCrop((480, 640)),
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip()
        ])
    )
