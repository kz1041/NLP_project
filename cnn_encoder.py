class EncoderCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.1, for_attention=False):
        super(EncoderCNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # Calculate padding so we have same number of outputs as inputs
        self.padding = int((kernel_size-1)/2)
        print("PADDING", self.padding)
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Create convolutional layers
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.kernel_size, padding=self.padding)
        
        self.dropout1 = nn.Dropout(dropout)
        
        # Create linear layers
        if for_attention:
            self.nn_for_outputs = nn.Linear(hidden_size, 2 * hidden_size)
            self.nn_for_hidden = nn.Linear(hidden_size, 2 * hidden_size)
        else:
            self.nn_for_outputs = nn.Linear(hidden_size, hidden_size)
            self.nn_for_hidden = nn.Linear(hidden_size, hidden_size)
        
        
    def forward(self, source_sentences, source_lengths):
        batch_size = source_sentences.shape[0]
        # Pass into embedding
        res = self.embedding(source_sentences)
        
        # Pass into first convolutional layer
        outputs = self.conv1(res.transpose(1,2)).transpose(1,2)
        # Nonlinear activation function
        outputs = F.relu(outputs)
        
        # Pass into dropout layer
        outputs = self.dropout1(outputs)
        
        # Pass into second convolutional layer
        outputs = self.conv2(outputs.transpose(1,2)).transpose(1,2)
        outputs = F.tanh(outputs)
        
        hidden, _ = torch.max(outputs, dim=1)
        
        hidden = self.nn_for_hidden(hidden)
        hidden = F.tanh(hidden)
        hidden = hidden.view(1, batch_size, -1)
        
        outputs = self.nn_for_outputs(outputs)
        outputs = F.tanh(outputs)
        
        
        return outputs, hidden