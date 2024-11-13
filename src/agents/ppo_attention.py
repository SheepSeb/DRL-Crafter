import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output, attention_probs
    
    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PPOAttentionPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, attention_dim=256, num_heads=4):
        super().__init__()
        
        # CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        conv_out_size = self._get_conv_out_size(obs_shape)
        
        # Project CNN features to attention dimension
        self.feature_projection = nn.Linear(64, attention_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(attention_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(attention_dim, num_heads)
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm(attention_dim)
        self.layer_norm2 = nn.LayerNorm(attention_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 4),
            nn.ReLU(),
            nn.Linear(attention_dim * 4, attention_dim)
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(attention_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(attention_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))
    
    def forward(self, x):
        # CNN encoding
        batch_size = x.size(0)
        conv_features = self.conv(x)
        
        # Reshape features for attention
        h, w = conv_features.shape[-2:]
        features = conv_features.view(batch_size, 64, -1).transpose(1, 2)  # (batch_size, h*w, channels)
        
        # Project to attention dimension
        features = self.feature_projection(features)  # (batch_size, h*w, attention_dim)
        
        # Add positional encoding
        features = self.positional_encoding(features)
        
        # Self-attention
        attended_features = self.attention(features, features, features)
        attended_features = self.layer_norm1(features + attended_features)  # Residual connection
        
        # Feedforward
        ff_output = self.feedforward(attended_features)
        ff_output = self.layer_norm2(attended_features + ff_output)  # Residual connection
        
        # Pool attention outputs (mean pooling)
        pooled_features = ff_output.mean(dim=1)  # (batch_size, attention_dim)
        
        # Policy and value heads
        action_logits = self.actor(pooled_features)
        value = self.critic(pooled_features)
        
        return action_logits, value
    
    def get_action(self, state, deterministic=False):
        action_logits, value = self(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, value
    
    def evaluate_actions(self, state, action):
        action_logits, value = self(state)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

class PPOAttentionAgent:
    def __init__(self, 
                 obs_shape,
                 action_dim,
                 device,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 num_epochs=4,
                 batch_size=64,
                 attention_dim=256,
                 num_heads=4):
        
        self.policy = PPOAttentionPolicy(
            obs_shape=obs_shape,
            action_dim=action_dim,
            attention_dim=attention_dim,
            num_heads=num_heads
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Initialize buffers
        self.reset_buffers()
    
    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def act(self, state, eval=False):
        with torch.no_grad():
            state = state.unsqueeze(0)
            action, value = self.policy.get_action(state, deterministic=eval)
            if not eval:
                # Store transition only during training
                log_prob = Categorical(logits=self.policy(state)[0]).log_prob(action)
                self.states.append(state)
                self.actions.append(action)
                self.values.append(value)
                self.log_probs.append(log_prob)
        
        return action.cpu().item()
    
    def learn(self, state, action, reward, next_state, done):
        # Store reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done:
            self._update()
            self.reset_buffers()
    
    def _compute_advantages(self):
        # Convert lists to tensors
        rewards = torch.tensor(self.rewards).float().to(self.device)
        values = torch.cat(self.values).squeeze()
        dones = torch.tensor(self.dones).float().to(self.device)
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * lastgaelam
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update(self):
        # Compute advantages and returns
        advantages, returns = self._compute_advantages()
        
        # Convert buffers to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # PPO update for multiple epochs
        for _ in range(self.num_epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                # Get minibatch indices
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # Get minibatch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_states, batch_actions)
                
                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                total_loss = (policy_loss + 
                            self.value_loss_coef * value_loss + 
                            self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()