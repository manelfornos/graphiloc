import torch
import torch_geometric
import optuna

import indoorloc_enums as ilenums
import indoorloc_trainer as iltrainer

# Numerical constants
SEED = 42

# Enums
CUDA = ilenums.Devices.cuda.value
CPU = ilenums.Devices.cpu.value


class GNNRegressionOptimizer:
    def __init__(self):
        self.device = torch.device(
            CUDA if torch.cuda.is_available() else CPU
        )

    def _set_gridparams(self, trial, data, model_class):
        if model_class.__name__ in ['SAGERegressor']:
            n_layers = 2
            if hasattr(data, 'train_mask'):
                d = data
            else:
                d = data['train']

            gnn_hidden_dims = []
            gnn_dropouts = []

            for i in range(n_layers):
                gnn_hidden_dims.append(trial.suggest_categorical(f'gnn_hidden_dim_layer_{i}', [128, 256]))
                if i < n_layers - 1:
                    gnn_dropouts.append(trial.suggest_float(f'gnn_dropout_layer_{i}', 0.1, 0.6, step=0.1))

            params = {
                'input_dim': d.num_features,
                'output_dim': d.y.shape[1],
                'gnn_hidden_dims': gnn_hidden_dims,
                'gnn_dropouts': gnn_dropouts,
                'mlp_layers': trial.suggest_categorical('mlp_layers', [2, 4]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                'lr_factor': 0.9,
            }
            return params
    
    def _objective(self, trial, data, model_class, task, max_epochs, patience):
        params = self._set_gridparams(trial, data, model_class)
        model = model_class(**params).to(self.device)
        
        if task == "regression":
            trainer = iltrainer.GNNRegressionTrainer()
        if task == "classification":
            trainer = iltrainer.GNNClassificationTrainer()

        loss = trainer.train_validate(
            data, model, max_epochs, patience, verbose=0, trial=trial
        )
        
        return loss

    def run_optuna_study(
            self,
            data: torch_geometric.data.Data, 
            model_class: type, 
            task,
            study_name,
            direction,
            storage,    
            load_if_exists,
            n_trials,
            max_epochs,
            patience,
            callbacks=None
    ):
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, 
                                               n_warmup_steps=500,
                                               n_min_trials=5),
            sampler=optuna.samplers.TPESampler(seed=SEED,
                n_startup_trials=10
            )
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda trial: self._objective(trial, data, model_class, task, max_epochs, patience),
            n_trials=n_trials,
            n_jobs=1,
            callbacks=callbacks)