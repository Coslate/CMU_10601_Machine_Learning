"""
Rewritten solution, based on starter code given to students
"""

import torch
from torch import nn, optim, Tensor
from typing import Optional
import json
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt

# DO NOT CHANGE THIS LINE!
# And DO NOT reset the torch seed anywhere else in your code!
torch.manual_seed(10601)

# for Q5.3 empirical question
'''
from rnn import *
lm = torch.load("model_q5_4_large_stories.pt")
tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
'''


class SentenceDataset:
    def __init__(self, a):
        with open(a) as f:
            data = json.load(f)
            data = [torch.tensor(seq) for seq in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # TODO: Initialize weights 
        self.i2h = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.h2h = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # See here for PyTorch activation functions
        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activation = nn.ReLU()

    def forward(self, input: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input at timestep t
                - shape: (batch_size, input_dim,)
            hidden_state (Tensor): Hidden state from timestep t-1
                - shape: (batch_size, hidden_dim,)

        Returns:
            Tensor: Next hidden state at timestep t
                - shape: (batch_size, hidden_dim)
        """
        # TODO: fill this in
        out = self.activation(self.i2h(input) + self.h2h(hidden_state))

        return out
    

class RNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # TODO: Initialie the RNNCell Class
        self.cell = RNNCell(input_dim=input_dim, hidden_dim=hidden_dim)

        # TODO: Initialize the weights
        self.out = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def step(self, input: Tensor, hidden_prev: Optional[Tensor] = None) -> Tensor:
        """
        Compute hidden and output states for a single timestep

        Args:
            input (Tensor): input at current timestep t
                - shape: (batch_size, input_dim,)
            hidden_prev (Tensor): hidden states of preceding timesteps [1, t-1]
                If there are no previous hidden states (i.e. we are at t=1), then
                this may be None and we will initialize the previous hidden state
                to all zeros.
                - shape: (batch_size, t-1, hidden_dim)

        Returns:
            Tensor: RNN hidden state at current timestep t
                - shape: (batch_size, hidden_dim,)
            Tensor: RNN output at current timestep t.
                RNN output state at current timestep t 
                - shape: (batch_size, hidden_dim,)
        """
        if hidden_prev is None:
            # If this is the first timestep and there is no previous hidden state,
            # create a dummy hidden state of all zeros
            
            # TODO: Fill this in (After you intialize, make sure you add .to(input))
            last_hidden_state = torch.zeros(input.size(0), 1, self.hidden_dim).to(input)
        else:
            # TODO: fill this in
            last_hidden_state = hidden_prev

        # Call the RNN cell and apply the transform to get a prediction
        next_hidden_state = self.cell(input, last_hidden_state[:, -1, :])
        next_output_state = self.out(next_hidden_state)

        #return next_hidden_state[:, -1, :], next_output_state[:, -1, :]
        return next_hidden_state, next_output_state

    def forward(self, sequence: Tensor) -> Tensor:
        """
        Compute hidden and output states for all timesteps over input sequence

        Args:
            sequence (Tensor): inputs to RNN over t timesteps
                - shape (batch_size, t, input_dim)

        Returns:
            Tensor: hidden states over t timesteps
                - shape (batch_size, t, hidden_dim)
            Tensor: output states over t timesteps
                - shape (batch_size, t, hidden_dim)
        """
        hidden_states = None
        output_states = []
        b, t, _ = sequence.shape
        #prev_hidden_state = None

        for i in range(t):
            # TODO: Extract the current input 
            inp = sequence[:, i, :]

            # TODO: Call step() to get the next hidden/output states
            #next_hidden_state, next_output_state = self.step(input=inp, hidden_prev=prev_hidden_state)
            next_hidden_state, next_output_state = self.step(input=inp, hidden_prev=hidden_states)
            #prev_hidden_state = next_hidden_state
            next_hidden_state = next_hidden_state.unsqueeze(1)

            # TODO: Concatenate the newest hidden state to to all previous ones
            if hidden_states is None:
                hidden_states = next_hidden_state
            else:
                hidden_states = torch.cat([hidden_states, next_hidden_state], dim=1)  # Concatenate along time dim 

            # TODO: Append the next output state to the list
            output_states.append(next_output_state)
            

        # TODO: torch.stack all of the output states over the timestep dim
        output_states = torch.stack(output_states, dim=1) # Shape: (batch_size, t, hidden_dim)

        return hidden_states, output_states


class SelfAttention(nn.Module):
    """Scaled dot product attention from original transformers paper"""

    def __init__(self, hidden_dim, key_dim, value_dim):
        """
        hidden_dim (int): Hidden dimension of RNN
        key_dim (int): Dimension of attention key and query vectors
        value_dim (int): Dimension of attention value vectors
        """
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # TODO: Initialize Query, Key, and Value transformations
        self.query_transform = nn.Linear(in_features=hidden_dim, out_features=key_dim)
        self.key_transform = nn.Linear(in_features=hidden_dim, out_features=key_dim)
        self.value_transform = nn.Linear(in_features=hidden_dim, out_features=value_dim)

        # Output projection within the Attention Layer (NOT the LM head)
        self.output_transform = nn.Linear(in_features=value_dim, out_features=hidden_dim)

    def step(self, y_all: Tensor) -> Tensor:
        """
        Compute attention for **current** timestep t

        Args:
            y_all (Tensor): Predictions up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for current timestep
                - shape (batch_size, hidden_dim,)
        """
        last_hidden_state = y_all[:, -1].unsqueeze(1) #(B, 1, H)
        batch_size, _, _hidden_dim = last_hidden_state.shape
        assert _hidden_dim == self.hidden_dim

        # TODO: Compute the QKV values
        query = self.query_transform(last_hidden_state) #(B, 1, d)
        keys = self.key_transform(y_all) #(B, t, d)
        values = self.value_transform(y_all) #(B, t, vd)

        scaling = self.key_dim ** (0.5)
        query = query / scaling

        # TODO: Compute attention weights over values
        # Remember to divide raw attention scores by scaling factor
        # These scores should then be normalized using softmax
        # Hint: use torch.softmax
        weights = torch.bmm(query, keys.transpose(1, 2))  #(B, 1, t)
        weights = torch.softmax(weights, dim=-1) #(B, 1, t)

        # TODO: Compute weighted sum of values based on attention weights
        output_state = torch.bmm(weights, values) #(B, 1, vd)

        # Apply output projection back to hidden dimension
        output_state = self.output_transform(output_state).squeeze(1) #(B, hidden_dim)
        
        return output_state
    

    def forward(self, y_all) -> Tensor:
        """
        Compute attention for all timesteps

        Args:
            y_all (Tensor): Predictions up to current timestep t
                - shape (batch_size, t, hidden_dim)

        Returns:
            Tensor: Attention output for all timesteps
                - shape (batch_size, t, hidden_dim)
        """
        t = y_all.shape[1]
        output_states = []

        for i in range(t):
            # TODO: Perform a step of SelfAttention and unsqueeze the result,
            # Then add it to the output states
			# HINT: use self.step()
            output_state = self.step(y_all[:, :i+1, :]) #Should not pass token that is ahead of current i time-step. (not look ahead)
            output_states.append(output_state.unsqueeze(1))
            
        # TODO: torch.cat() all of the outputs in the list
        # across the sequence length dimension (t)
        output_states = torch.cat(output_states, dim=1)

        return output_states

class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        key_dim=None,
        value_dim=None,
    ):
        """
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of RNN hidden states
        vocab_size (int): Number of (sub)words in model vocabulary
        """
        super(RNNLanguageModel, self).__init__()

        # TODO: Initialize word embeddings (HINT: use nn.Embedding)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # TODO: RNN backbone
        self.rnn = RNN(input_dim=embed_dim, hidden_dim=hidden_dim)

        # TODO: Self Attention Layer
        self.attention = SelfAttention(hidden_dim=hidden_dim, key_dim=key_dim, value_dim=value_dim)

        # TODO: Final projection from RNN output state to next token logits
        self.lm_head = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Computes next-token logits and hidden states for each token in tokens

        Args:
            tokens (Tensor): Input tokens IDs
                - shape (batch_size, t,)

        Returns:
            Tensor: Next-token logits for each token from the LM head
                - shape (batch_size, t, vocab_size)
            Tensor: RNN hidden states for each token
                - shape (batch_size, t, hidden_dim)
            Tensor: RNN output states for each token
                - shape (batch_size, t, hidden_dim)
        """
        # TODO: Apply embeddings, rnns, and lm_head sequentially
        x_embed = self.embeddings(tokens) #(batch_size, t, embed_dim)
        x_hidden_rnn, x_output_rnn = self.rnn(x_embed) #(batch_size, t, hidden_dim)
        x_attn_out = self.attention(x_output_rnn) #(batch_size, t, hidden_dim)
        x_head_out = self.lm_head(x_attn_out) #(batch_size, t, vocab_size)

        return x_head_out, x_hidden_rnn, x_output_rnn

    def select_token(self, token_logits: Tensor, temperature: float) -> int:
        """
        Selects (or samples) next token from token_logits

        Args:
            token_logits (Tensor): Next token logits
                - shape (batch_size, vocab_size,)
            temperature (float): Sampling temperature. If 0, do greedy decoding.

        Returns:
            index (int): ID of next token selected
        """
        if temperature == 0:
            # Greedy Decoding
            return torch.argmax(token_logits, dim=-1)
        else:
            # Temperature Sampling
            token_logits = token_logits / temperature
            token_probs = torch.softmax(token_logits, dim=-1)
            index = torch.multinomial(token_probs, 1)[0]
            return index

    def generate(self, tokens: Tensor, max_tokens=10, temperature=0.0) -> Tensor:
        """
        Generates new tokens given `tokens` as a prefix.

        Args:
            tokens (Tensor): Input tokens
                - shape: (1, input_length,)
            max_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature

        Returns:
            Tensor: generated tokens
                - shape: (max_tokens,)
        """
        # Get hidden states for input tokens by calling forward
        #bs, T = tokens.shape
        token_logits, hidden_states, attn_inputs = self(tokens)
        next_token_logits = token_logits[0, -1]

        new_tokens = []
        step = 0

        # Now, start generating new tokens
        # While we could in theory repeatedly call self(tokens) here, we don't since
        # that's an order of magnitude more inefficient as we would be repeatedly re-encoding
        # the prefix. Instead, here, we repeatedly compute the hidden state and next token
        # logits for the *latest* token.
        while True:
            step += 1

            # Select next token based on next_token_logits
            next_token = self.select_token(next_token_logits, temperature)
            new_tokens.append(next_token.item())

            # Stop generating once we reach max_tokens
            if step >= max_tokens:
                break

            # Get next input embedding
            embed = self.embeddings(next_token)
            #embed_reshape = embed.reshape((bs, embed.shape[-1]))

            # Get next hidden state and next attn input state from RNN
            next_hidden_state, next_attn_input = self.rnn.step(embed, hidden_states)
            #next_hidden_state, next_attn_input = self.rnn.step(embed, hidden_states[:, -1, :])

            # Update hidden states
            hidden_states = torch.cat(
                [hidden_states, next_hidden_state.unsqueeze(1)], dim=1
            )

            # Update attention inputs
            attn_inputs = torch.cat(
                [attn_inputs, next_attn_input.unsqueeze(1)], dim=1
            )
            
            # Call attention 
            next_output_state = self.attention.step(attn_inputs)

            # Generate the token to be used in the next step of generation
            next_token_logits = self.lm_head(next_output_state)


        return torch.tensor(new_tokens)


def train(lm, train_data, valid_data, loss_fn, optimizer, num_sequences, batch_size, return_elapsed_time=False):
    """
    Run one epoch of language model training

    Args:
        lm (RNNLanguageModel): RNN language model
        dataset (list[Tensor]): Train dataset
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function
        optimizer: PyTorch Adam optimizer
        num_sequences: The total number of sequences to train on
        batch_size: Number of sequences we process in one step

    Returns:
        List: Training losses
        List: Validation Losses
    """
    # Set the model to training model
    lm.train()
    max_grad_norm = 1.0

    train_batch_losses = []
    train_batch_loss = 0.0
    valid_batch_losses = []

    # DO NOT change the next line
    dataset = train_data
    start_time = time.time()

    # Run validation everytime we process around 10% of the training data
    val_frequency = 0.1
    val_index = int(num_sequences * val_frequency) // batch_size
    if val_index == 0:
        val_index = 1
      
    print(f"val_index = {val_index}")
    # Loop over the dataset
    for idx, sequence in enumerate(dataset):
        time_elapsed = round((time.time() - start_time) / 60, 6)

        # Move the sequence to the device
        sequence = sequence.to(device)

        # Stop training when we hit the num_sequences limit
        if idx == num_sequences // batch_size:
            print(f"idx = {idx}, num_sequences = {num_sequences}, batch_size = {batch_size}, num_sequences//batch_size = {num_sequences//batch_size}")
            break

        # TODO: Zero gradients
        optimizer.zero_grad()


        # TODO: Forward pass through model
        batch_size, t = sequence.shape
        token_logits, hidden_states, attn_inputs = lm(sequence)


        # TODO: Compute next-token classification loss

        # Hint 1: The Token logits should be of shape (batch_size, t, vocab_size), 
        # and the sequence should be of shape (batch_size, t). 
        # If we want to compute the loss of the nth logit token, 
        # which token in the sequence should I compare it with?
        target = sequence[:, 1:] #shift by one position for prediction

        # Hint 2: We will need to permute the token_logits to the 
        # correct shape before passing into loss function
        token_logits = token_logits[:, :-1, :] #remove the last one
        token_logits = token_logits.permute(0, 2, 1) #(batch_size, vocab_size, t-1) for loss_fn format
        loss = loss_fn(token_logits, target) 

        # TODO: Backward pass through model
        loss.backward()


        # DO NOT change this - clip gradient norm to avoid exploding gradients
        nn.utils.clip_grad_norm_(lm.parameters(), max_grad_norm)

        # TODO: Update weights
        optimizer.step()


        # DO NOT change any of the code below
        train_batch_loss += loss.detach().cpu().item()

        if idx % val_index == 0:
            # Calculate train/val loss as normal
            train_batch_loss = (
                round(train_batch_loss / val_index, 6)
                if idx != 0
                else round(train_batch_loss, 6)
            )

            # Append to the batch loss and reset to 0
            train_batch_losses.append(train_batch_loss)
            train_batch_loss = 0.0

            print(f"Batch: {idx} | Sequence Length: {(sequence.shape[1])} | Elapsed time (minutes): {time_elapsed}")

            # Append to the validation loss
            valid_loss = round(validate(lm, valid_data, loss_fn), 6)
            valid_batch_losses.append(valid_loss)


    print(f"Train Batch Losses: {train_batch_losses}")
    print(f"len(train_batch_losses) = {len(train_batch_losses)}")

    if return_elapsed_time:
        return train_batch_losses, valid_batch_losses, time_elapsed
    else:
        return train_batch_losses, valid_batch_losses


@torch.no_grad()
def validate(lm, dataset, loss_fn):
    """
    Args:
        lm (RNNLanguageModel):
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function

    Returns:
        float: Average validation loss
    """
    # Set the model to eval mode
    lm.eval()

    mean_loss = 0.0
    num_batches = 1

    for i, sequence in enumerate(dataset):        
        if i < num_batches:
            # Move the sequence to the device
            sequence = sequence.to(device)

            # TODO: Perform forward pass through the model
            batch_size, t = sequence.shape
            token_dists, _, _ = lm(sequence)

            # TODO: Compute loss (Same as in train)
            target = sequence[:, 1:] #shift by one position for prediction (batch_size, t-1)
            token_dists = token_dists[:, :-1, :] #remove the last one
            token_dists = token_dists.permute(0, 2, 1) #(batch_size, vocab_size, t-1) for loss_fn format
            loss = loss_fn(token_dists, target) 

            # DO NOT change this line
            mean_loss += loss.detach().cpu().item()

    return mean_loss / num_batches

def plotQuestion5_4a(train_dataset, valid_dataset, embed_dim, vocab_size, dk, dv, num_sequences, batch_size, device):
    print(f"batch_size = {batch_size}")
    print(f"num_sequences = {num_sequences}")
    print(f"dk = {dk}")
    print(f"dv = {dv}")
    # Define embed_dim configurations

    # Store losses for plotting
    train_losses = {}
    valid_losses = {}

    # Loop over different configurations
    num_sequences_dict = {}
    total_time = {}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    # Create the RNN Language Model with the given dimensions
    lm = RNNLanguageModel(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            vocab_size=vocab_size,
            key_dim=dk,
            value_dim=dv,
    ).to(device)
    
    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

        # Train the model and get losses
    train_loss, valid_loss, elapsed_time = train(
            lm,
            train_dataloader,
            valid_dataloader,
            loss_fn,
            optimizer,
            num_sequences,
            batch_size,
            return_elapsed_time=True
    )
    
        # Store losses
    train_losses = train_loss[-1]
    valid_losses = valid_loss[-1]
    total_time   = elapsed_time
    print(f"final train_loss = {train_losses}")
    print(f"final valid_loss = {valid_losses}")
    print(f"time = {elapsed_time*60} (sec)")

def plotQuestion5_3(train_data, valid_data, vocab_size, dk, dv, embed_dim, batch_size):
    print(f"batch_size = {batch_size}")
    print(f"embed_dim = {embed_dim}")
    print(f"dk = {dk}")
    print(f"dv = {dv}")

    num_sequences = [10000, 20000, 50000, 100000]

    # Store losses for plotting
    train_losses = []
    valid_losses = []

    # Loop over different configurations
    for num_sequence in num_sequences:
        print(f"> Training/num_sequence = {num_sequence}...")
    
    # Create the RNN Language Model with the given dimensions
        lm = RNNLanguageModel(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            vocab_size=vocab_size,
            key_dim=dk,
            value_dim=dv,
        ).to(device)
    
        # Optimizer and Loss Function
        optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Train the model and get losses
        train_loss, valid_loss = train(
            lm,
            train_data,
            valid_data,
            loss_fn,
            optimizer,
            num_sequence,
            batch_size
        )
    
        # Store losses
        train_losses.append(train_loss[-1])
        valid_losses.append(valid_loss[-1])

    # Generate plots

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(num_sequences, train_losses, label=f'Training Loss vs different num_sequences')
    plt.xlabel('num_sequence')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different num_sequences')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(num_sequences, valid_losses, label=f'Validation Loss vs different num_sequences')
    plt.xlabel('num_sequence')
    plt.ylabel('Loss')
    plt.title('Validation Loss for Different num_sequences')
    plt.legend()
    plt.grid()
    plt.show()

def plotQuestion5_2(train_dataset, valid_dataset, embed_dim, vocab_size, dk, dv, num_sequences, device):
    print(f"num_sequences = {num_sequences}")
    print(f"dk = {dk}")
    print(f"dv = {dv}")
    # Define embed_dim configurations
    batch_sizes = [32, 64, 128, 256]

    # Store losses for plotting
    train_losses = {}
    valid_losses = {}

    # Loop over different configurations
    num_sequences_dict = {}
    total_time = {}
    for batch_size in batch_sizes:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )

        print(f"> Training/batch_size = {batch_size}...")
    
    # Create the RNN Language Model with the given dimensions
        lm = RNNLanguageModel(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            vocab_size=vocab_size,
            key_dim=dk,
            value_dim=dv,
        ).to(device)
    
        # Optimizer and Loss Function
        optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Train the model and get losses
        train_loss, valid_loss, elapsed_time = train(
            lm,
            train_dataloader,
            valid_dataloader,
            loss_fn,
            optimizer,
            num_sequences,
            batch_size,
            return_elapsed_time=True
        )
    
        # Store losses
        train_losses[batch_size] = train_loss
        valid_losses[batch_size] = valid_loss
        num_sequences_dict[batch_size] = [batch_size * idx for idx in range(1, len(train_loss) + 1)]
        total_time[batch_size] = elapsed_time

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        print(f"Plotting training batch_size = {batch_size}")
        plt.plot(num_sequences_dict[batch_size], train_losses[batch_size], label=f'Training Loss with batch_size={batch_size}')
        print(f"batch_size = {batch_size}, elapsed_time = {total_time[batch_size]*60} secs")
    plt.xlabel('Number of Sequences')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Batch Sizes')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        print(f"Plotting validation batch_size = {batch_size}")
        plt.plot(num_sequences_dict[batch_size], valid_losses[batch_size], label=f'Validation Loss with batch_size={batch_size}')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Loss')
    plt.title('Validation Loss for Different Batch Sizes')
    plt.legend()
    plt.grid()
    plt.show()

def plotQuestion5_1(train_data, valid_data, vocab_size, dk, dv, num_sequences, batch_size):
    print(f"batch_size = {batch_size}")
    print(f"num_sequences = {num_sequences}")
    print(f"dk = {dk}")
    print(f"dv = {dv}")
    # Define embed_dim configurations
    embed_dims = [64, 128, 256, 512]

    # Store losses for plotting
    train_losses = {}
    valid_losses = {}

    # Loop over different configurations
    for embed_dim in embed_dims:
        print(f"> Training/embed_dim = {embed_dim}...")
    
    # Create the RNN Language Model with the given dimensions
        lm = RNNLanguageModel(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            vocab_size=vocab_size,
            key_dim=dk,
            value_dim=dv,
        ).to(device)
    
        # Optimizer and Loss Function
        optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Train the model and get losses
        train_loss, valid_loss = train(
            lm,
            train_data,
            valid_data,
            loss_fn,
            optimizer,
            num_sequences,
            batch_size
        )
    
        # Store losses
        train_losses[embed_dim] = train_loss
        valid_losses[embed_dim] = valid_loss

    # Generate plots
    num_sequences = [batch_size * idx for idx in range(1, len(train_losses[embed_dims[0]]) + 1)]
    print(f"batch_size = {batch_size}")
    print(f"train_losses[embed_dims[0]] = {train_losses[embed_dims[0]]}")
    print(f"num_sequences = {num_sequences}")

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    for embed_dim in embed_dims:
        print(f"Plotting training embed_dim = {embed_dim}")
        plt.plot(num_sequences, train_losses[embed_dim], label=f'Training Loss with embed_dim={embed_dim}')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Embedding Dimensions')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    for embed_dim in embed_dims:
        print(f"Plotting validation embed_dim = {embed_dim}")
        plt.plot(num_sequences, valid_losses[embed_dim], label=f'Validation Loss with embed_dim={embed_dim}')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Loss')
    plt.title('Validation Loss for Different Embedding Dimensions')
    plt.legend()
    plt.grid()
    plt.show()


@torch.no_grad()
def complete(lm, tokenizer, prefix: str, num_tokens=64, temperature=0.0, device=None):
    """
    Generates text completion from language model given text prefix.
    This function has been implemented for you.

    Args:
        prefix (str):
        num_tokens (int): Number of new tokens to generate
        temperature (float): Sampling temperature

    Returns:
        str: Text completion
    """
    lm.eval()

    input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input = input.to(device)
    output = lm.generate(input, max_tokens=num_tokens, temperature=temperature)
    
    return tokenizer.decode(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--metrics_out", type=str)
    parser.add_argument("--train_losses_out", type=str)
    parser.add_argument("--val_losses_out", type=str)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dk", type=int)
    parser.add_argument("--dv", type=int)
    parser.add_argument("--num_sequences", type=int)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    # Initialize torch device to use cuda if we have a gpu
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
    vocab_size = tokenizer.vocab_size

    # Initialize LM 
    lm = RNNLanguageModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        key_dim=args.dk,
        value_dim=args.dv,
    )
    lm = lm.to(device)

    print(lm)
    print(
        "Number of Parameters: ",
        sum(p.numel() for p in lm.parameters() if p.requires_grad),
    )
    print("Loading data")

    train_data = SentenceDataset(args.train_data)
    print(f"len(train_data) = {len(train_data)}")

    valid_data = SentenceDataset(args.val_data)
    print(f"len(valid_data) = {len(valid_data)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )

    print("Finished Loading Dataset")

    '''
    # Initialize PyTorch cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.parameters(), lr=1e-3)

    ### BEGIN: Training Loop
    start = time.time()
    train_loss, valid_loss = train(
        lm,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimizer,
        args.num_sequences,
        args.batch_size,
    )
    end = time.time()
    time_taken = end - start
    ### END: Training Loop

    results = {
        "Train Losses": train_loss,
        "Valid Losses": valid_loss,
        "Final Train Loss": train_loss[-1],
        "Final Valid Loss": valid_loss[-1],
        "Time": time_taken,
    }

    for key, value in results.items():
        print(key, value)

    print("Final Train Loss: ", train_loss[-1])
    print("Final Valid Loss: ", valid_loss[-1])

    # Saves your trained model weights(Please comment when submitting to gradescope)
    torch.save(lm, "model_q5_4_large_stories.pt")
    '''

    #plotQuestion5_1(train_dataloader, valid_dataloader, vocab_size, args.dk, args.dv, args.num_sequences, args.batch_size)
    #plotQuestion5_2(train_data, valid_data, args.embed_dim, vocab_size, args.dk, args.dv, args.num_sequences, device)
    #plotQuestion5_3(train_dataloader, valid_dataloader, vocab_size, args.dk, args.dv, args.embed_dim, args.batch_size)
    #plotQuestion5_4a(train_data, valid_data, args.embed_dim, vocab_size, args.dk, args.dv, args.num_sequences, args.batch_size, device)



    # You can later load back in your model in a separate Python file by running:
    # >>> from rnn import *
    # >>> lm = torch.load("model.pt")
    # This may be helpful for Empirical Question 5.4, where
    # training the model may take up to 45 minutes.
    from rnn import *
    lm = torch.load("model_q5_4_large_stories.pt")

    """
    # Example code for generating text with your LM

    test_str = ["Once upon a time there was a"]
    
    for ts in test_str:
        completion = complete(ts, num_tokens=64, temperature=0.3)
        print("  Test prefix:", ts)
        print("  Test output:", completion)
    """

    # # Greedy Sampling(Please comment out when submitting to gradescope)
    # test_strs = ["Once upon a time there was a "]
    # for ts in test_strs:
    #     completion = complete(ts, num_tokens=128, temperature=0.0)
    #     print("  Test prefix:", ts)
    #     print("  Test output:", completion)


    # Looping through all temperature values for empirical questions
    # Please comment out when submitting to gradescope
    test_strs = ["Once upon a time there was a "]
    print("----------------")
    num_tokens = 128
    samples_per_setting = 20
    for temperature in [0, 0.3, 0.8]:
        for _ in range(samples_per_setting):
            completion = complete(lm, tokenizer, prefix=test_strs[0], num_tokens=num_tokens, temperature=temperature, device=device)
            print(f"temperature = {temperature}")
            print("  Test prefix:", test_strs[0])
            print("  Test output:", completion)
        print("----------------")

    # Save your metrics
    '''
    with open(args.train_losses_out, "w") as f:
        for loss in train_loss:
            f.write(str(loss) + "\n")

    with open(args.val_losses_out, "w") as f:
        for loss in valid_loss:
            f.write(str(loss) + "\n")

    with open(args.metrics_out, "w") as f:
        f.write("Final Train Loss: " + str(train_loss[-1]) + "\n")
        f.write("Final Valid Loss: " + str(valid_loss[-1]) + "\n")
    '''