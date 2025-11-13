import os
import torch
import optuna

from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from torch.utils.tensorboard.writer import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

class NPEOptunaTraining:
    # USAGE:
    # npe = NPEOptunaTraining(y, x, n_trials, study_name, output_dir, n_jobs, device)
    # study = npe()
    # theta = [N, S]
    # x = A = [N, 1]
    def __init__(self, theta, A, 
                study_name,
                output_dir,
                n_trials=30, 
                n_jobs=1,
                device='cpu',
                val_frac=0.2,
                seed = 42
                ):
       
        self.theta = theta.float().to(device)
        self.A = A.float().to(device)
        self.S = self.theta.shape[1]
        self.device = torch.device(device)
        self.study_name = study_name
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.val_frac = val_frac
        self.seed = seed

        os.makedirs(os.path.join(self.output_dir, self.study_name), exist_ok=True)
        self.storage = f"sqlite:///{output_dir}/{study_name}/{study_name}.db"

        th_min = self.theta.min(dim=0).values - 3.0
        th_max = self.theta.max(dim=0).values + 3.0
        self.prior = BoxUniform(low=th_min, high=th_max, device=self.device)

        # train/val split
        N = len(self.theta)
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=g)
        n_val = max(200, int(self.val_frac * N))
        self.val_idx = idx[:n_val]
        self.train_idx = idx[n_val:]


        self.n_startup_trials = 20
        self.n_blocks_min = 2 
        self.n_blocks_max = 5
        self.n_transf_min = 2
        self.n_transf_max = 5
        self.n_hidden_min = 32
        self.n_hidden_max = 128
        self.n_lr_min = 5e-6
        self.n_lr_max = 1e-3
    
    def objective(self, trial):
        n_blocks = trial.suggest_int("n_blocks", self.n_blocks_min, self.n_blocks_max)
        n_transf = trial.suggest_int("n_transf", self.n_transf_min,  self.n_transf_max)
        n_hidden = trial.suggest_int("n_hidden", self.n_hidden_min, self.n_hidden_max, log=True)
        lr = trial.suggest_float("lr", self.n_lr_min, self.n_lr_max, log=True)

        # writer = SummaryWriter('%s/%s/%s.%i' % 
        #             (self.output_dir, self.study_name, self.study_name, trial.number))
        writer = None

        neural_posterior = posterior_nn('maf', 
                hidden_features=n_hidden, 
                num_transforms=n_transf, 
                num_blocks=n_blocks, 
                use_batch_norm=True)
        
        anpe = SNPE(prior=self.prior,
                density_estimator=neural_posterior,
                device=self.device, 
                summary_writer=writer)
        anpe.append_simulations(self.theta[self.train_idx], self.A[self.train_idx])
        p_theta_x_est = anpe.train(
                training_batch_size=2048,
                learning_rate=lr, 
                show_train_summary=True)
        qphi = anpe.build_posterior(density_estimator=p_theta_x_est)

        save_path = os.path.join(self.output_dir, self.study_name, f"{self.study_name}.{trial.number}.pt")
        torch.save(qphi, save_path)
        best_valid_log_prob = anpe._summary["best_validation_loss"][0]
        return -1 * best_valid_log_prob
    
    def run(self):
        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(
            study_name=self.study_name, 
            storage=self.storage, 
            sampler=sampler, 
            directions='minimize', 
            load_if_exists=True,
        )
        cb = optuna.study.MaxTrialsCallback(self.n_trials)

        study.optimize(self.objective, n_trials=None, n_jobs=self.n_jobs, callbacks=[cb])
        return study
    
    def __call__(self):
        return self.run()


if __name__ == "__main__":
    blob = torch.load('../spender_desi_noise_6latent_space.pt', map_location="cpu")
    theta = blob["latents"].float()   # [N, 6]
    A     = blob["A"].float()         # [N, 1]

    trainer = NPEOptunaTraining(
        theta=theta,
        A=A,
        study_name="desi_noise_flow_optuna",
        output_dir="../optuna_studies",
        n_trials=100,
        n_jobs=1,
        device='cuda',
        val_frac=0.2,
        seed=42
    )
    study = trainer()

    trials_stored = sorted(study.trials, key=lambda t: t.values[0])
    top_paths = [os.path.join(trainer.output_dir, trainer.study_name, f"{trainer.study_name}.{trial.number}.pt") for trial in trials_stored[:5]]
    for i, path in enumerate(top_paths):
        print(f"Top {i+1} model path: {path}")
    