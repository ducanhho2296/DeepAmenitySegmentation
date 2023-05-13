import mlflow

# Set the experiment ID or name
# experiment_id = "ducanh_Unet experiment"
# Or use the experiment name
experiment = mlflow.get_experiment_by_name("ducanh_Unet experiment")
experiment_id = experiment.experiment_id

# Get all runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# Find the best run based on the lowest validation loss
best_run = runs.loc[runs["metrics.val_loss"].idxmin()]

# Get the optimized learning rate and batch size
best_learning_rate = best_run["params.learning_rate"]
best_batch_size = int(best_run["params.batch_size"])

print("Best learning rate:", best_learning_rate)
print("Best batch size:", best_batch_size)
