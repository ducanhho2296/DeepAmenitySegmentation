from train import train
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='Model type (unet or deeplabv3plus)')
parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--gpu', type=int, default=0, help='specific gpu for training')
parser.add_argument('--weight', type=int, default=None, help='Continue training from the last checkpoint')

args = parser.parse_args()

# Set the experiment name
experiment_name = "ducanh_Unet learning rates experiment"
mlflow.set_experiment(experiment_name)

# Define the search space for hyperparameters
learning_rates = [1e-4, 1e-5, 1e-6]
batch_sizes = [4, 8, 16]

# Iterate through hyperparameter combinations
for lr in learning_rates:
    for bs in batch_sizes:
        with mlflow.start_run() as run:
            # Log the hyperparameters
            mlflow.log_params({
                "model": args.model,
                "learning_rate": lr,
                "batch_size": bs,
                "num_epochs": args.epoch,
                "gpu": args.gpu
            })

            # Train the model with the current hyperparameters
            val_loss, train_loss = train(args.model, lr, bs, args.epoch, args.gpu, args.weight)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("train_loss", train_loss)

