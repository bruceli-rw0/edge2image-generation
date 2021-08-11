import os
from glob import glob
from torch.utils import data
from PIL import Image
from torchvision import transforms

class CustomDataset(data.Dataset):
    def __init__(self, folder, num_data, args):
        super().__init__()

        self.direction = args.direction
        self.edges_and_images = list()

        assert len(folder)
        for data_folder in folder:
            assert os.path.exists(data_folder)
            components = os.path.normpath(data_folder).split(os.sep)

            if any([f in ["edges2shoes", "edges2handbags"] for f in components]):
                self.edges_and_images += \
                    [(path, self._getABImageData) for path in sorted(glob(os.path.join(data_folder, "*.jpg")))]
            
            elif any([f in ["lhq_256", "kaggle_landscape"] for f in components]):
                assert sorted(glob(os.path.join(data_folder, "*s"))) == \
                    [os.path.join(data_folder, "edges"), os.path.join(data_folder, "images")]
                self.edges_and_images += \
                    [((edges, image), self._getImageEdgeData) for edges, image in zip(
                        sorted(glob(os.path.join(data_folder, "edges", "*.jpg"))),
                        sorted(glob(os.path.join(data_folder, "images", "*.jpg")))
                    )]

        if num_data != -1:
            self.edges_and_images = self.edges_and_images[:num_data]
        
        self.transform = transforms.Compose([
            transforms.Resize(args.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.edges_and_images)

    def __getitem__(self, index):
        path, process_fn = self.edges_and_images[index]
        return process_fn(path)

    def _getABImageData(self, path):
        # read image from path
        AB = Image.open(path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        edge = AB.crop((0, 0, w2, h))
        image = AB.crop((w2, 0, w, h))
        return {
            'A': self.transform(edge), 
            'B': self.transform(image),
            'A_path': path
        } if self.direction == 'AtoB' else {
            'A': self.transform(image), 
            'B': self.transform(edge),
            'A_path': path
        }

    def _getImageEdgeData(self, path):
        pathe, pathi = path
        edge = Image.open(pathe).convert('RGB')
        image = Image.open(pathi).convert('RGB')
        return {
            'A': self.transform(edge), 
            'B': self.transform(image),
            'A_path': pathe
        } if self.direction == 'AtoB' else {
            'A': self.transform(image), 
            'B': self.transform(edge),
            'A_path': pathi
        }
