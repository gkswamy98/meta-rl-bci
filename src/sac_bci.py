import tensorflow as tf
import numpy as np
from tf2rl.algos.sac_discrete import SACDiscrete
from tf2rl.algos.sac_discrete import SAC
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.misc.huber_loss import huber_loss
from .eegnet import EEGNet


class SACBCI(SACDiscrete):
    def __init__(
            self,
            state_shape,
            action_dim,
            *args,
            actor_fn=None,
            critic_fn=None,
            target_update_interval=None,
            **kwargs):
        kwargs["name"] = "SAC_discrete"
        self.actor_fn = EEGNet
        self.critic_fn = lambda: EEGNet(softmax=False)
        self.target_hard_update = target_update_interval is not None
        self.target_update_interval = target_update_interval
        self.n_training = tf.Variable(0, dtype=tf.int32)
        SAC.__init__(self=self, state_shape=state_shape, action_dim=action_dim, *args, **kwargs)
        if self.auto_alpha:
            self.target_alpha = -np.log((1.0 / action_dim)) * 0.98

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        # The output of actor is categorical distribution
        self.actor = self.actor_fn()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        self.qf1 = self.critic_fn()
        self.qf2 = self.critic_fn()
        self.qf1_target = self.critic_fn()
        self.qf2_target = self.critic_fn()
        update_target_variables(self.qf1_target.weights,
                                self.qf1.weights, tau=1.)
        update_target_variables(self.qf2_target.weights,
                                self.qf2.weights, tau=1.)
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        
    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim
        
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        if len(state.shape) == 3:
            print(state.shape)
        action = self._get_action_body(tf.constant(state), test)
        # TODO: should this be sampling?
        return np.argmax(action.numpy(), axis=-1) 

        
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            batch_size = states.shape[0]
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.int32)
            
            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        actions], axis=1)
            
            with tf.GradientTape(persistent=True) as tape:
                # Compute critic loss
                next_action_prob = self.actor(next_states)
                next_action_logp = tf.math.log(next_action_prob + 1e-8)
                next_q = tf.minimum(
                    self.qf1_target(next_states), self.qf2_target(next_states))

                # Compute state value function V by directly computes expectation
                target_q = tf.expand_dims(tf.einsum(
                    'ij,ij->i', next_action_prob, next_q - self.alpha * next_action_logp), axis=1)  # Eq.(10)
                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * target_q)

                current_q1 = self.qf1(states)
                
                current_q2 = self.qf2(states)

                td_loss1 = tf.reduce_mean(huber_loss(
                    target_q - tf.expand_dims(tf.gather_nd(current_q1, indices), axis=1),
                    delta=self.max_grad) * weights)
                td_loss2 = tf.reduce_mean(huber_loss(
                    target_q - tf.expand_dims(tf.gather_nd(current_q2, indices), axis=1),
                    delta=self.max_grad) * weights)  # Eq.(7)

                # Compute actor loss
                current_action_prob = self.actor(states)
                current_action_logp = tf.math.log(current_action_prob + 1e-8)

                policy_loss = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob,
                              self.alpha * current_action_logp - tf.stop_gradient(
                                  tf.minimum(current_q1, current_q2))) * weights)  # Eq.(12)
                mean_ent = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob, current_action_logp)) * (-1)

                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.log_alpha * tf.stop_gradient(current_action_logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            if self.target_hard_update:
                if self.n_training % self.target_update_interval == 0:
                    update_target_variables(self.qf1_target.weights,
                                            self.qf1.weights, tau=1.)
                    update_target_variables(self.qf2_target.weights,
                                            self.qf2.weights, tau=1.)
            else:
                update_target_variables(self.qf1_target.weights,
                                        self.qf1.weights, tau=self.tau)
                update_target_variables(self.qf2_target.weights,
                                        self.qf2.weights, tau=self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

        return (td_loss1 + td_loss2) / 2., policy_loss, mean_ent, \
            tf.reduce_min(current_action_logp), tf.reduce_max(current_action_logp), \
            tf.reduce_mean(current_action_logp)