import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from torchvision.utils import save_image
from tqdm import tqdm
from model import discriminatorNN, generatorNN

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="animefacedataset",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=100,
        metavar="N",
        help="Latent size of generated noise",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_generator",
        metavar="E",
        help="folder where generated faces are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)

def train_discriminator(generator, discriminator, real_images, discriminator_optimizer, criterion, batch_size, latent_size, device):
    # Clear discriminator gradients
    discriminator_optimizer.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)

    # Image targets
    real_targets = torch.ones(real_images.size(0), 1, device=device)

    # Compute the loss
    real_loss = criterion(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # We generate false images to train the discriminator to recognize the true ones
    # First we generate noise using torch.randn of size (batch size, latent size, 1, 1)
    image_noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    # Pass the fake images to the generator
    fake_images = generator(image_noise)

    # Create the targets for the false images
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    # Compute the loss
    fake_loss = criterion(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Compute the discriminator full loss
    loss = real_loss + fake_loss
    # Updating the weights
    loss.backward()
    discriminator_optimizer.step()
    return loss.item(), real_score, fake_score

def train_generator(generator, discriminator, generator_optimizer, criterion, batch_size, latent_size, device):
    # Clear generator gradients
    generator_optimizer.zero_grad()

    # Generate random noise
    image_noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    # Generate fake images with noise
    fake_images = generator(image_noise)

    # Try to fool the discriminator
    # Pass your fake images to the discriminator
    preds = discriminator(fake_images)
    # Create labels to fool your discriminator (we say they are true)
    targets = torch.ones(batch_size, 1, device=device)
    loss = criterion(preds, targets)

    # Update generator weights
    loss.backward()
    generator_optimizer.step()

    return loss.item()

def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Updated device setup
    torch.manual_seed(args.seed)

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(device)

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create output folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    torch.cuda.empty_cache()

    # Parameters
    latent_size = args.latent_size
    batch_size = args.batch_size

    # Load Dataset
    train_ds = ImageFolder(args.data, transform=Compose([
        Resize((64, 64)),
        CenterCrop(64),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    
    for real_images, _ in tqdm(train_dl):
        real_images = to_device(real_images, device)

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    discriminator = to_device(discriminatorNN(), device)
    generator = to_device(generatorNN(latent_size), device)

    # Create optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Initialize schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.99)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.99)


    # Loss function
    criterion = F.binary_cross_entropy

    for epoch in range(args.epochs):
        # Initialize models for training
        generator.train()
        discriminator.train()
        epoch_losses_g, epoch_losses_d = [], []

        for real_images, _ in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            real_images = to_device(real_images, device)
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(generator, discriminator, real_images, discriminator_optimizer, criterion, batch_size, latent_size, device)
            epoch_losses_d.append(loss_d)
            # Train generator
            loss_g = train_generator(generator, discriminator, generator_optimizer, criterion, batch_size, latent_size, device)
            epoch_losses_g.append(loss_g)

        avg_loss_g = sum(epoch_losses_g) / len(epoch_losses_g)
        avg_loss_d = sum(epoch_losses_d) / len(epoch_losses_d)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss G: {avg_loss_g:.4f}, Loss D: {avg_loss_d:.4f}")

        scheduler_g.step()
        scheduler_d.step()

    model_file = args.output + "/generator" + ".pth"
    torch.save(generator.state_dict(), model_file)
    print("Training completed. Last model saved to", args.output)

if __name__ == "__main__":
    main()