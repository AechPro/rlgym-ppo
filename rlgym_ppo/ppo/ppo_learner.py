import os
import time

import numpy as np
import torch

from rlgym_ppo.ppo import ContinuousPolicy, DiscreteFF, MultiDiscreteFF, ValueEstimator


class PPOLearner(object):
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        policy_type,
        policy_layer_sizes,
        critic_layer_sizes,
        continuous_var_range,
        batch_size,
        n_epochs,
        policy_lr,
        critic_lr,
        clip_range,
        ent_coef,
        mini_batch_size,
        device,
    ):
        self.device = device

        assert (
            batch_size % mini_batch_size == 0
        ), "MINIBATCH SIZE MUST BE AN INTEGER MULTIPLE OF BATCH SIZE"

        if policy_type == 2:
            self.policy = ContinuousPolicy(
                obs_space_size,
                act_space_size * 2,
                policy_layer_sizes,
                device,
                var_min=continuous_var_range[0],
                var_max=continuous_var_range[1],
            ).to(device)
        elif policy_type == 1:
            self.policy = MultiDiscreteFF(
                obs_space_size, policy_layer_sizes, device
            ).to(device)
        else:
            self.policy = DiscreteFF(
                obs_space_size, act_space_size, policy_layer_sizes, device
            ).to(device)
        self.value_net = ValueEstimator(obs_space_size, critic_layer_sizes, device).to(
            device
        )
        self.mini_batch_size = mini_batch_size

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=critic_lr
        )
        self.value_loss_fn = torch.nn.MSELoss()

        # Calculate parameter counts
        policy_params = self.policy.parameters()
        critic_params = self.value_net.parameters()

        trainable_policy_parameters = filter(lambda p: p.requires_grad, policy_params)
        policy_params_count = sum(p.numel() for p in trainable_policy_parameters)

        trainable_critic_parameters = filter(lambda p: p.requires_grad, critic_params)
        critic_params_count = sum(p.numel() for p in trainable_critic_parameters)

        total_parameters = policy_params_count + critic_params_count

        # Display in a structured manner
        print("Trainable Parameters:")
        print(f"{'Component':<10} {'Count':<10}")
        print("-" * 20)
        print(f"{'Policy':<10} {policy_params_count:<10}")
        print(f"{'Critic':<10} {critic_params_count:<10}")
        print("-" * 20)
        print(f"{'Total':<10} {total_parameters:<10}")
        
        print(f"Current Policy Learning Rate: {policy_lr}")
        print(f"Current Critic Learning Rate: {critic_lr}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.cumulative_model_updates = 0

    def learn(self, exp):
        """
        Compute PPO updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.

        Returns:
            dict: Dictionary containing training report metrics.
        """

        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []

        # Save parameters before computing any updates.
        policy_before = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.value_net.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(self.batch_size)
            for batch in batches:
                (
                    batch_acts,
                    batch_old_probs,
                    batch_obs,
                    batch_target_values,
                    batch_advantages,
                ) = batch
                batch_acts = batch_acts.view(self.batch_size, -1)
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = start + self.mini_batch_size

                    acts = batch_acts[start:stop].to(self.device)
                    obs = batch_obs[start:stop].to(self.device)
                    advantages = batch_advantages[start:stop].to(self.device)
                    old_probs = batch_old_probs[start:stop].to(self.device)
                    target_values = batch_target_values[start:stop].to(self.device)

                    # Compute value estimates.
                    vals = self.value_net(obs).view_as(target_values)

                    # Get policy log probs & entropy.
                    log_probs, entropy = self.policy.get_backprop_data(obs, acts)
                    log_probs = log_probs.view_as(old_probs)

                    # Compute PPO loss.
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )

                    # Compute KL divergence & clip fraction using SB3 method for reporting.
                    with torch.no_grad():
                        log_ratio = log_probs - old_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item()

                        # From the stable-baselines3 implementation of PPO.
                        clip_fraction = (
                            torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                            .cpu()
                            .item()
                        )
                        clip_fractions.append(clip_fraction)

                    policy_loss = -torch.min(
                        ratio * advantages, clipped * advantages
                    ).mean()
                    value_loss = self.value_loss_fn(vals, target_values)
                    ppo_loss = (
                        (policy_loss - entropy * self.ent_coef)
                        * self.mini_batch_size
                        / self.batch_size
                    )

                    ppo_loss.backward()
                    value_loss.backward()

                    mean_val_loss += value_loss.cpu().detach().item()
                    mean_divergence += kl
                    mean_entropy += entropy.cpu().detach().item()
                    n_minibatch_iterations += 1

                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1

        if n_minibatch_iterations == 0:
            n_minibatch_iterations = 1

        # Compute averages for the metrics that will be reported.
        mean_entropy /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
        if len(clip_fractions) == 0:
            mean_clip = 0
        else:
            mean_clip = np.mean(clip_fractions)

        # Compute magnitude of updates made to the policy and value estimator.
        policy_after = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.value_net.parameters()
        ).cpu()
        policy_update_magnitude = (policy_before - policy_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        # Assemble and return report dictionary.
        self.cumulative_model_updates += n_iterations

        report = {
            "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,
            "Value Function Loss": mean_val_loss,
            "SB3 Clip Fraction": mean_clip,
            "Policy Update Magnitude": policy_update_magnitude,
            "Value Function Update Magnitude": critic_update_magnitude,
        }
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        return report

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(folder_path, "PPO_POLICY.pt"))
        torch.save(
            self.value_net.state_dict(), os.path.join(folder_path, "PPO_VALUE_NET.pt")
        )
        torch.save(
            self.policy_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"),
        )
        torch.save(
            self.value_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"),
        )

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), "PPO LEARNER CANNOT FIND FOLDER {}".format(
            folder_path
        )

        self.policy.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY.pt"))
        )
        self.value_net.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET.pt"))
        )
        self.policy_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"))
        )
        self.value_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"))
        )
