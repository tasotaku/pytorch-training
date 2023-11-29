from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        dir_list = list(dir_path_resolved.glob("*"))
        self.img_list = []
        for dir in dir_list:
            image_path_list = list(dir.glob("*.png"))
            self.img_list += image_path_list

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        return img

    def __len__(self):
        return len(self.img_list)
    
if __name__ == "__main__":
    dataset = MyDataset("./data")
    print("---Number of files in the dataset---")
    print(len(dataset))
    print("---Output the size of the image file---")
    print(dataset[0].size)