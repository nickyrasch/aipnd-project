import torch


def save_model(model, image_datasets, arch, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()

    torch.save(
        {
            'arch': arch,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx
        },
        save_dir + '/checkpoint.pth'
    )