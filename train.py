import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from collections import OrderedDict
from train_input_args import train_input_args
from train_model import train_model
from save_model import save_model

print('Initiating Training...')


def train():
    print('Gathering Arguments...')
    args = train_input_args()

    data_dir = args.data_dir
    print("data_dir:", data_dir)

    save_dir = args.save_dir
    print("save_dir:", save_dir)

    arch = args.arch
    print("arch:", arch)

    learning_rate = args.learning_rate
    print("learning_rate:", learning_rate)

    hidden_units = args.hidden_units
    print("hidden_units:", hidden_units)

    epochs = args.epochs
    print("epochs:", epochs)

    gpu = args.gpu
    print("gpu:", gpu)

    # ---------------------------------------------------------
    print('Setting up transforms...')

    data_types = [
        'train',
        'valid',
        'test'
    ]
    rotation = 30
    resize = 225
    crop_size = 224
    normalize_mean = [
        0.485,
        0.456,
        0.406
    ]
    normalize_std = [
        0.229,
        0.224,
        0.225
    ]

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(
            data_dir + '/' + x,
            transform=data_transforms[x]
        ) for x in data_types
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=32,
            shuffle=True
        ) for x in data_types
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in data_types
    }

    # ---------------------------------------------------------
    print('Setting up Device & Models...')

    # set the device to what the cpu if the user requests it otherwise
    # default try to use cuda if the device is cuda capable.
    device = 'cpu' if gpu == 'cpu' else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    architectures = {
        'resnet': resnet18,
        'alexnet': alexnet,
        'vgg16': vgg16
    }

    model = architectures[arch]

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    # ---------------------------------------------------------

    # Criterion NLLLoss which is recommended with Softmax final layer
    criterion = nn.NLLLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # Set the model to the device
    model.to(device)

    # ---------------------------------------------------------
    print('Training the model...')

    # Train the model
    model_ft = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataset_sizes,
        dataloaders,
        epochs,
        device
    )

    # ---------------------------------------------------------
    print('Saving the model...')

    save_model(
        model_ft,
        image_datasets,
        arch,
        save_dir
    )

    print('Saved model successfully!')


# Call to main function to run the program
train()