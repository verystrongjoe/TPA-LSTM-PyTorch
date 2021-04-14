import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7) 168
        self.original_columns = data.original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one

        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features, requires_grad=True, device='cuda'))

        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC, requires_grad=True, device='cuda'))

        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True, device='cuda'))

        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features, requires_grad=True, device='cuda'))
        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;

        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;

        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, input):
        if self.use_cuda:
            x = input.cuda()

        """
        Step 1. First step is to feed this information to LSTM and find out the hidden states 
        General info about LSTM:
        """
        input_to_lstm = x.permute(1, 0, 2).contiguous()  # (T, B, C) -> (B, T, C)
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)    # (T, B, NUM_DIRECTION*H), ((NUM OF LAYERS *  NUM_DIRECTION, B, H), (NUM OF LAYERS *  NUM_DIRECTION, B, H))
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))  # (1, B, H)

        """
        Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()    # (B, T, H)
        hn = hn.permute(1, 0, 2).contiguous()   # (B, 1, H)
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features)  # (B, 1, T, H)
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer))   # (B, HC, T, 1)
        convolution_output = self.dropout(convolution_output);   # (B, HC, T, 1)


        """
        Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)  # (B, HC, T)

        """
        In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any 
        problem while matrix multiplication padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)   # (B, 1, H)
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)  # (B, HC, T)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn  # (B, 1, H)
            final_convolution_output = convolution_output # (B, HC, T)

        """
        final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()   # (B, T, HC)
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()  # (B, 1, H) -> (B, H, 1)
        convolution_output_for_scoring = convolution_output_for_scoring.cuda()  # (B, T, HC)
        final_hn_realigned = final_hn_realigned.cuda()   # (B, H, 1)

        # attention_matrix shape : (B, HC, H)
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()   # (B, T, HC) * (B, HC, H) -> (B, T, H)
        scoring_function = torch.bmm(mat1, final_hn_realigned)  # (B, T, H) * (B, H, 1)  = (B, T, 1)
        alpha = torch.nn.functional.sigmoid(scoring_function)  # (B, T, 1)
        context_vector = alpha * convolution_output_for_scoring  # (B, T, HC)
        context_vector = torch.sum(context_vector, dim=1)  # (B, HC)

        """
        Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)  # (B, HC, 1) = V_t
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)  # (B, H, H) * (B, H, 1) + (B, H, HC) * (B, HC, 1)  = (B, H, 1) + (B, H, 1) = (B, H, 1)
        result = torch.bmm(self.final_matrix, h_intermediate)  # (B, C, H) * (B, H, 1) = (B, C, 1)
        result = result.permute(0, 2, 1).contiguous()  # (B, 1, C)
        result = result.squeeze()  # (B, C)

        """
        Remove from result the extra result points which were added as a result of padding 
        """
        final_result = result[:result.size(0) - diff]  # (B, C)

        """
        Adding highway network to it
        """
        if self.hw > 0:
            z = x[:, -self.hw:, :]  # (B, T, C)  -> (B, HW, C)
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)  # (B, C, HW) -> (B * C, HW)
            z = self.highway(z)  # (B * C, 1)
            z = z.view(-1, self.original_columns)  # (B, C)
            res = final_result + z  # (B, C)
        return torch.sigmoid(res)

