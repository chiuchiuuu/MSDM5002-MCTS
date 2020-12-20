import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class PolicyValueNet(nn.Module):

    def __init__(self, size):
        super(PolicyValueNet, self).__init__()

        self.size = size

        ## convolution
        # padding=1 to remain the size of the board
        self.conv1 = nn.Conv2d(5, 16, 3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        ## action probability layers
        self.prob_conv = nn.Conv2d(64, 5, 1)
        self.prob_fc = nn.Linear(5*self.size*self.size, self.size*self.size)

        ## evalutation layers
        self.eval_conv = nn.Conv2d(64, 2, 1)
        self.eval_fc1 = nn.Linear(2*size*size, 32)
        self.eval_fc2 = nn.Linear(32, 1)


    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # probablity
        prob = F.relu(self.prob_conv(x))
        prob = prob.view(-1, 5*self.size*self.size)
        prob = F.softmax(self.prob_fc(prob))

        # evaluation
        eval = F.relu(self.eval_conv(x))
        eval = eval.view(-1, 2*self.size*self.size)
        eval = F.relu(self.eval_fc1(eval))
        eval = F.tanh(self.eval_fc2(eval))

        return prob, eval


class Training:
    def __init__(self, size, lr):
        self.size = size

        self.model = PolicyValueNet(size)

        self.lr = 0.1
        self.l2_const = 1e-4

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                        weight_decay=self.l2_const)

    def train(self, states, probs, winners, lr):

        self.model.train()

        self.optimizer.zero_grad()
        self.adjust_learning(self.optimizer, lr)

        output = self.model(states)
        loss = self.loss()

    def adjust_learning(self, optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr




    
