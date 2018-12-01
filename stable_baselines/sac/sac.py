import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.sac.policies import SACPolicy


def get_vars(scope):
    """
    Alias
    """
    return tf_util.get_trainable_vars(scope)


class SAC(OffPolicyRLModel):
    """Soft Actor-Critic"""

    def action_probability(self, observation, state=None, mask=None):
        pass

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-3, buffer_size=50000,
                 learning_starts=1000, train_freq=1, batch_size=32,
                 tau=0.001, reward_scale=1, target_update_interval=1, gradient_steps=4,
                 verbose=0, tensorboard_log=None, _init_setup_model=True):
        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        self.reward_scale = reward_scale
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma

        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.mu = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None

        self.obs_target = None
        self.target_policy = None
        self.action_train_ph = None
        self.critic_target = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None)

                    self.next_observations_ph = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph
                    # TODO; check ph between processed or not
                    self.observations_ph = self.policy_tf.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                    # self.observations_ph = self.policy_tf.processed_x
                    # self.next_observations_ph = self.target_policy.processed_x

                with tf.variable_scope("model", reuse=False):
                    mu, pi, logp_pi = self.policy_tf.make_actor(self.observations_ph)
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.observations_ph, self.actions_ph)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.observations_ph,
                                                                   self.actions_ph,
                                                                   reuse=True)
                    self.qf1, self.qf2, = qf1, qf2

                with tf.variable_scope("target", reuse=False):
                    _, _, value_target = self.target_policy.make_critics(self.next_observations_ph,
                                                                        self.target_policy.make_actor(
                                                                            self.next_observations_ph)[1])
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Min Double-Q
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Targets for Q and V regression
                    q_backup = tf.stop_gradient(
                        self.reward_scale * self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)

                    # policy_regularization_losses = tf.get_collection(
                    #     tf.GraphKeys.REGULARIZATION_LOSSES,
                    #     scope=self._policy.name)
                    # policy_regularization_loss = tf.reduce_sum(
                    #     policy_regularization_losses)

                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    # (control dep of policy_train_op because sess.run otherwise evaluates in nondeterministic order)
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    values_params = get_vars('model/values_fn')

                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        # Polyak averaging for target variables
                        # (control flow because sess.run otherwise evaluates in nondeterministic order)
                        with tf.control_dependencies([train_values_op]):
                            source_params = get_vars("model/values_fn/vf")
                            target_params = get_vars("target/values_fn/vf")

                            target_update_op = [
                                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                                for target, source in zip(target_params, source_params)
                            ]

                            # All ops to call during one training step
                            self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                             value_loss, qf1, qf2, value_fn, logp_pi,
                                             policy_train_op, train_values_op, target_update_op]

                            # Initializing targets to match source variables
                            target_init_op = [
                                tf.assign(target, source)
                                for target, source in zip(target_params, source_params)
                            ]

                        # tf.summary.scalar('actor_loss', self.actor_loss)
                        # tf.summary.scalar('critic_loss', self.critic_loss)
                    # tf.summary.scalar('critic_target', tf.reduce_mean(self.critic_target))
                    # tf.summary.histogram('critic_target', self.critic_target)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(self.rewards_ph))
                    # tf.summary.histogram('rewards', self.rewards_ph)

                # IMPORTANT: are the target variables also saved ?
                self.params = find_trainable_variables("model")

                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                # self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, log=False):
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards,
            self.terminals_ph: batch_dones,
        }

        out = self.sess.run(self.step_ops, feed_dict)
        policy_loss, qf1_loss, qf2_loss, value_loss, qf1, qf2, value_fn, logp_pi = out

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="SAC"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            episode_rewards = [0.0]
            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                action = self.policy_tf.step(obs, deterministic=False)
                new_obs, reward, done, _ = self.env.step(action * np.abs(self.action_space.low))
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                if writer is not None:
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, step)

                for _ in range(self.gradient_steps):
                    if step < self.batch_size:
                        break
                    self._train_step(step, writer)
                    # if step % self.target_update_interval == 0:
                    #     # Run target ops here.
                    #     self.sess.run(self.target_ops)

                episode_rewards[-1] += reward
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def save(self, save_path):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "reward_scale": self.reward_scale,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
