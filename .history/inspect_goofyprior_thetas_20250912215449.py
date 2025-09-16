import argparse
import matplotlib.pyplot as plt
import torch
from eb_arena_mapper import gen_batch_from_seed

def main():
    parser = argparse.ArgumentParser(description="Inspect GoofyPrior thetas using eb_arena_mapper.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--prior', type=str, default='goofy', help='Prior type (should be "goofy")')
    parser.add_argument('--device', type=str, default='cpu', help='Device for prior (default: cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for prior')
    parser.add_argument('--worst_prior', action='store_true', help='Use worst prior (should be False for GoofyPrior)')
    parser.add_argument('--same_prior', action='store_true', help='Use same prior (should be False for GoofyPrior)')
    parser.add_argument('--uniform_percentage', type=float, default=0.0, help='Uniform percentage (default: 0.0)')
    parser.add_argument('--theta_max', type=float, default=1.0, help='Max theta value (default: 1.0)')
    args = parser.parse_args()

    # Convert dtype string to torch dtype
    if args.dtype == 'float32':
        args.dtype = torch.float32
    elif args.dtype == 'float64':
        args.dtype = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Generate batch and get prior
    (_, labels), prior = gen_batch_from_seed(args, return_prior=True)
    if hasattr(prior, 'thetas'):
        thetas = prior.thetas
        print("GoofyPrior thetas shape:", thetas.shape)
        print("First few thetas:", thetas.flatten()[:10])
        plt.hist(thetas.flatten(), bins=50)
        plt.title("GoofyPrior Thetas Distribution")
        plt.xlabel("Theta value")
        plt.ylabel("Count")
        plt.show()
    else:
        print("Prior does not have a 'thetas' attribute. Inspect the prior object:", type(prior))

if __name__ == "__main__":
    main()
