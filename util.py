from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def build_dataset_CIFAR100(is_train, data_path):
    transform = build_transform(is_train)
    dataset = datasets.CIFAR100(data_path, train=is_train, transform=transform, download=True)
    nb_classes = 100
    return dataset, nb_classes


def build_dataset_CIFAR10(is_train, data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    dataset = datasets.CIFAR10(data_path, train=is_train, transform=transform, download=True)
    nb_classes = 10

    return dataset, nb_classes


def build_transform(is_train):
    input_size = 224
    eval_crop_ratio = 1.0

    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.0,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def prepare_data(batch_size, data='cifar-100'):

    if data == 'cifar-100':
        train_set, nb_classes = build_dataset_CIFAR100(is_train=True, data_path='./data/cifar100')
        test_set, _ = build_dataset_CIFAR100(is_train=False, data_path='./data/cifar100')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    elif data == 'cifar-10':
        train_set, nb_classes = build_dataset_CIFAR10(is_train=True, data_path='./data/cifar10')
        test_set, _ = build_dataset_CIFAR10(is_train=False, data_path='./data/cifar10')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    else:
        raise NotImplementedError

    return train_loader, test_loader, nb_classes

def evaluate_model(model, data_loader, device):
    # model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy

def train_one_epoch(model, criterion, optimizer, data_loader, device):

    cnt = 0

    for image, target in tqdm(data_loader):
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')