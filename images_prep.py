import torch.hub as hub
import torch.backends.cudnn as cudnn
import torch.nn as nn
import h5py
import resnet
from config import paths, parameters
from utils import get_transform
from dataset import CocoImages, Composite
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import load, no_grad
from tqdm import tqdm

train_path = paths["train_path"]
val_path = paths["val_path"]
preprocessed_dir = paths["preprocessed_dir"]
pretrained_path = paths["pretrained_path"]

image_size = parameters["image_size"]
central_fraction = parameters["central_fraction"]
preprocess_batch_size = parameters["preprocess_batch_size"]
data_workers = parameters["data_workers"]
output_size = parameters["output_size"]
output_features = parameters["output_features"]


class Net(nn.Module):
    """
        Wrapper class for the pre-trained ResNet-152 model
    """

    def __init__(self):
        super(Net, self).__init__()
        #net = resnet.resnet152(pretrained=True)
        #net = resnet.resnet34(pretrained=True)
        net = resnet.resnet18(pretrained=True)
        net.load_state_dict(load(pretrained_path))
        self.model = net

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_loader(*_paths):
    """
        Receives a list containing datasets, applies transforms and returns a compatible data loader
    """

    transform = get_transform(image_size, central_fraction)
    datasets = [CocoImages(path, transform=transform) for path in _paths]
    dataset = Composite(*datasets)
    data_loader = DataLoader(dataset,
                             batch_size=preprocess_batch_size,
                             num_workers=data_workers,
                             shuffle=False,
                             pin_memory=True,
                             )
    return data_loader


def prep_images(device):
    cudnn.benchmark = True

    # Load ResNet-152 pre-trained on ImageNet, to be used for pre-processing the images
    # net = hub.load('pytorch/vision:v0.4.2', 'resnet152', pretrained=True)
    # net = net.to(device)
    net = Net().cuda()
    net.eval()

    # Create loaders for train & validation data sets
    loader = create_loader(train_path, val_path)
    features_shape = (len(loader.dataset),
                      output_features,
                      output_size,
                      output_size
                      )

    # Transfer the images through the model for pre-processing and save as h5py file in the preprocessed directory
    with h5py.File(preprocessed_dir, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        with no_grad():
            for ids, images in tqdm(loader):
                images = Variable(images.cuda(async=True))
                out = net(images)
                j = i + images.size(0)
                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j
