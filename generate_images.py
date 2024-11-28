import argparse
import os

from torchvision.utils import save_image
import torch

from model import generatorNN

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--model",
        type=str,
        default="model/generator.pth",
        metavar="D",
        help="Trained generator (model) to generate images (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_images",
        metavar="D",
        help="Output of generated images",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1,
        metavar="D",
        help="Number of images to output",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=100,
        metavar="N",
        help="Latent size of generated noise",
    )
    args = parser.parse_args()
    return args

def main() -> None:
    """Main Function."""
    # options
    args = opts()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    output_dir = args.output

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Updated device setup

    # load model and transform
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model = generatorNN(args.latent_size).to(device)
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    with torch.no_grad():
        # Create random latent vectors to feed the generator
        latent_vectors = torch.randn(args.number, args.latent_size, 1 ,1 , device=device)

        gen_images = model(latent_vectors)

        # Rescale images to [0, 1] (assuming generator outputs in [-1, 1])
        gen_images = (gen_images + 1) / 2.0

        # Save each image
        for i, img in enumerate(gen_images):
            img_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            save_image(img, img_path)

    print(f"Generated {args.number} images and saved to '{output_dir}'")

if __name__ == "__main__":
    main()