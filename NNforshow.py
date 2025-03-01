import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy

class CustomNet(nn.Module):
    def __init__(self, layer_num, inputn, nnumber1, nnumber2, nnumber3=None):
        super(CustomNet, self).__init__()
        self.layer_num = layer_num

        self.fc1_W1 = nn.Parameter(torch.zeros(nnumber1, inputn))
        self.fc2_W1 = nn.Parameter(torch.zeros(nnumber2, nnumber1))
        self.fc1_W2 = nn.Parameter(torch.zeros(nnumber1, inputn) )
        self.fc2_W2 = nn.Parameter(torch.zeros(nnumber2, nnumber1) )
        
        self.fc1_bias = nn.Parameter(torch.zeros(nnumber1))
        self.fc2_bias = nn.Parameter(torch.zeros(nnumber2))
        if self.layer_num>2:
            self.fc3_W1 = nn.Parameter(torch.zeros(nnumber3, nnumber2))
            self.fc3_W2 = nn.Parameter(torch.zeros(nnumber3, nnumber2) )
            self.fc3_bias = nn.Parameter(torch.zeros(nnumber3))      


        self.fc1_gradients = None
        self.fc2_gradients = None
        self.fc3_gradients = None
        self.optimizer = None
        self.trainhis = []
        self.testhis = []
        self.teststdhis = []

    def reset_w(self, fc1_W, fc2_W, fc3_W=None):
        self.fc1_W1 = copy.deepcopy(fc1_W)
        self.fc2_W1 = copy.deepcopy(fc2_W)
        if self.layer_num>2: 
            self.fc3_W1 = copy.deepcopy(fc3_W)

           
    # 重载 to() 方法
    def to(self, device):
        # 调用父类的 to() 方法以移动模型的参数
        super(CustomNet, self).to(device)
        
        # 自动找到所有的自定义张量并移动到设备
        for name, tensor in self.__dict__.items():
            if torch.is_tensor(tensor):  # 检查是否是 torch.Tensor
                setattr(self, name, tensor.to(device))  # 将张量移动到设备
        
        return self
    
    def forward(self, x):
        
        self.input = x

        wdif = self.fc1_W1 - self.fc1_W2
        x =  torch.mm(x, wdif.t()) +self.fc1_bias


        if self.training:
            x.register_hook(self.save_grad1)
        x = F.relu(x)

        self.input2 = x

        wdif2 = self.fc2_W1 - self.fc2_W2
        x = torch.mm(x, wdif2.t()) + self.fc2_bias


        if self.training:
            x.register_hook(self.save_grad2)

        if self.layer_num>2: 
            x = F.relu(x)
            self.input3=x

            wdif3 = self.fc3_W1-self.fc3_W2
            x = torch.mm(x, wdif3.t()) + self.fc3_bias
           
            if self.training:
                x.register_hook(self.save_grad3)

        
        return x


    def save_grad1(self, grad):
        self.fc1_gradients = grad*(self.input2>0)

    def save_grad2(self, grad):
        self.fc2_gradients = grad
        if self.layer_num>2: 
            self.fc2_gradients*=(self.input3>0)

    def save_grad3(self, grad):
        self.fc3_gradients = grad

    def update_weights_custom(self, lr):
        with torch.no_grad():
            
            updaterate1 = lr * self.fc1_gradients.T.clamp(max=0).abs()
            updaterate2 = lr * self.fc1_gradients.T.clamp(min=0)

            updated_W1 = self.fc1_W1 * (1 - updaterate1) + (self.input)* updaterate1
            updated_W2 = self.fc1_W2 * (1 - updaterate2) + (self.input)* updaterate2


            self.fc1_W1 = nn.Parameter(updated_W1)
            self.fc1_W2 = nn.Parameter(updated_W2)


            updaterate1_fc2 = lr * self.fc2_gradients.T.clamp(max=0).abs()
            updaterate2_fc2 = lr * self.fc2_gradients.T.clamp(min=0)

            updated_fc2W1 = self.fc2_W1 * (1 - updaterate1_fc2) + (self.input2)* updaterate1_fc2
            updated_fc2W2 = self.fc2_W2 * (1 - updaterate2_fc2) + (self.input2)* updaterate2_fc2

            self.fc2_W1 = nn.Parameter(updated_fc2W1)
            self.fc2_W2 = nn.Parameter(updated_fc2W2)

            if self.layer_num>2: 
                updaterate1_fc3 = lr * self.fc3_gradients.T.clamp(max=0).abs() 
                updaterate2_fc3 = lr * self.fc3_gradients.T.clamp(min=0)

                updated_fc3W1 = self.fc3_W1 * (1 - updaterate1_fc3) + (self.input3)* updaterate1_fc3
                updated_fc3W2 = self.fc3_W2 * (1 - updaterate2_fc3) + (self.input3)* updaterate2_fc3
                

                self.fc3_W1 = nn.Parameter(updated_fc3W1)
                self.fc3_W2 = nn.Parameter(updated_fc3W2)


class CustomNet_Stable(CustomNet):
    def __init__(self, layer_num, inputn, nnumber1, nnumber2, nnumber3=None):
        super(CustomNet_Stable, self).__init__(layer_num, inputn, nnumber1, nnumber2, nnumber3)

        self.fc1_avggrad = torch.ones(1,nnumber1)
        self.fc2_avggrad = torch.ones(1,nnumber2)
        if self.layer_num>2: 
            self.fc3_avggrad = torch.ones(1,nnumber3)

        # self.fc1_avggrad2 = torch.ones(1,nnumber1)
        # self.fc2_avggrad2 = torch.ones(1,nnumber2)
        # self.fc3_avggrad2 = torch.ones(1,outputn)

    def update_weights_custom(self, lr=0.001):
        with torch.no_grad():
            updaterate1 = lr * (self.fc1_gradients.T.clamp(max=0).abs()/ self.fc1_avggrad.T *0.1).clamp(max=1)
            updaterate2 = lr * (self.fc1_gradients.T.clamp(min=0)/ self.fc1_avggrad.T*0.1 ).clamp(max=1)

            updated_W1 = self.fc1_W1 * (1 - updaterate1) + (self.input)* updaterate1
            updated_W2 = self.fc1_W2 * (1 - updaterate2) + (self.input)* updaterate2


            self.fc1_W1 = nn.Parameter(updated_W1)
            self.fc1_W2 = nn.Parameter(updated_W2)


            updaterate1_fc2 = lr * (self.fc2_gradients.T.clamp(max=0).abs()/ self.fc2_avggrad.T*0.1 ).clamp(max=1)
            updaterate2_fc2 = lr * (self.fc2_gradients.T.clamp(min=0)/ self.fc2_avggrad.T*0.1 ).clamp(max=1)

            updated_fc2W1 = self.fc2_W1 * (1 - updaterate1_fc2) + (self.input2)* updaterate1_fc2
            updated_fc2W2 = self.fc2_W2 * (1 - updaterate2_fc2) + (self.input2)* updaterate2_fc2

            self.fc2_W1 = nn.Parameter(updated_fc2W1)
            self.fc2_W2 = nn.Parameter(updated_fc2W2)

            if self.layer_num>2: 
                updaterate1_fc3 = lr * (self.fc3_gradients.T.clamp(max=0).abs() / self.fc3_avggrad.T *0.1).clamp(max=1)
                updaterate2_fc3 = lr * (self.fc3_gradients.T.clamp(min=0) / self.fc3_avggrad.T *0.1).clamp(max=1)

                updated_fc3W1 = self.fc3_W1 * (1 - updaterate1_fc3) + (self.input3)* updaterate1_fc3
                updated_fc3W2 = self.fc3_W2 * (1 - updaterate2_fc3) + (self.input3)* updaterate2_fc3

                self.fc3_W1 = nn.Parameter(updated_fc3W1)
                self.fc3_W2 = nn.Parameter(updated_fc3W2)

            self.update_avggrad( lr)

    def update_avggrad(self, lr):
        self.fc1_avggrad = self.fc1_avggrad*(1-lr*(self.fc1_gradients!=0))+(self.fc1_gradients!=0)*self.fc1_gradients.abs()*lr
        self.fc2_avggrad = self.fc2_avggrad*(1-lr*(self.fc2_gradients!=0))+(self.fc2_gradients!=0)*self.fc2_gradients.abs()*lr
        if self.layer_num>2: 
            self.fc3_avggrad = self.fc3_avggrad*(1-lr*(self.fc3_gradients!=0))+(self.fc3_gradients!=0)*self.fc3_gradients.abs()*lr
        
        # self.fc1_avggrad2 = self.fc1_avggrad2*(1-lr*(self.fc1_gradients>0))+(self.fc1_gradients>0)*self.fc1_gradients*lr
        # self.fc2_avggrad2 = self.fc2_avggrad2*(1-lr*(self.fc2_gradients>0))+(self.fc2_gradients>0)*self.fc2_gradients*lr
        # self.fc3_avggrad2 = self.fc3_avggrad2*(1-lr*(self.fc3_gradients>0))+(self.fc3_gradients>0)*self.fc3_gradients*lr


class SimpleNN(nn.Module):
    def __init__(self, layer_num, inputn, nnumber1, nnumber2, nnumber3=None):
        super(SimpleNN, self).__init__()
        self.layer_num = layer_num

        # 初始化权重和偏置
        self.fc1_W = nn.Parameter(torch.zeros(nnumber1, inputn))
        self.fc2_W = nn.Parameter(torch.zeros(nnumber2, nnumber1))
        if self.layer_num>2: 
            self.fc3_W = nn.Parameter(torch.zeros(nnumber3, nnumber2))
            self.fc3_bias = nn.Parameter(torch.zeros(nnumber3))

        self.fc1_bias = nn.Parameter(torch.zeros(nnumber1))
        self.fc2_bias = nn.Parameter(torch.zeros(nnumber2))
        self.optimizer = None
        self.L2punishment = None
        self.trainhis = []
        self.testhis = []
        self.teststdhis = []

    def forward(self, x):
        x = F.linear(x, self.fc1_W, self.fc1_bias)
        x = F.relu(x)
        x = F.linear(x, self.fc2_W, self.fc2_bias)
        if self.layer_num>2: 
            x = F.relu(x)
            x = F.linear(x, self.fc3_W, self.fc3_bias)
        return x
    
    def reset_w(self, fc1_W, fc2_W, fc3_W=None):
        self.fc1_W = copy.deepcopy(fc1_W)
        self.fc2_W = copy.deepcopy(fc2_W)
        if self.layer_num>2: 
            self.fc3_W = copy.deepcopy(fc3_W)

    def update_weights_custom(self,lr):
        pass


class Regression():
    def __init__(self, MAXNUM, inputn, train_num, device):
        self.MAXNUM = MAXNUM
        self.inputn = inputn
        self.device = device
        self.criterion = nn.MSELoss()  # 均方误差作为损失函数

        
        np.random.seed(5)
        self.numlist = np.random.randint(0,high=MAXNUM,size=train_num)

    def build_model(self, layer_num, inputn, nnumber1, nnumber2, nnumber3=None):                
        fc1_W = nn.Parameter( (torch.randn(nnumber1, inputn) * torch.sqrt(torch.tensor(2. / inputn))).to(self.device) )
        fc2_W = nn.Parameter( (torch.randn(nnumber2, nnumber1) * torch.sqrt(torch.tensor(2. / nnumber1))).to(self.device) )
        if layer_num>2:
            fc3_W = nn.Parameter((torch.randn(nnumber3,nnumber2) * torch.sqrt(torch.tensor(2. / nnumber2))).to(self.device))
        else:
            fc3_W = None

        self.model_my = CustomNet(layer_num, inputn, nnumber1, nnumber2, nnumber3 ).to(self.device)
        self.model_mystable = CustomNet_Stable(layer_num, inputn, nnumber1, nnumber2, nnumber3 ).to(self.device)
        
        self.model_base = SimpleNN(layer_num, inputn, nnumber1, nnumber2, nnumber3).to(self.device)
        self.model_L2light = SimpleNN(layer_num, inputn, nnumber1, nnumber2, nnumber3).to(self.device)
        self.model_L2heavy = SimpleNN(layer_num, inputn, nnumber1, nnumber2, nnumber3).to(self.device)

        self.model_L2light.L2punishment = 0.01
        self.model_L2heavy.L2punishment = 0.1

        self.models = [self.model_my, self.model_mystable, self.model_base, self.model_L2light, self.model_L2heavy]
        for m in self.models:
            m.reset_w(fc1_W, fc2_W, fc3_W)

        for m in [self.model_my, self.model_mystable]:
            # 选择 model 中所有 bias 参数
            bias_params = [param for name, param in m.named_parameters() if 'bias' in name]
            # 将 bias_params 注册给 optimizer
            m.optimizer = torch.optim.SGD(bias_params, lr=0.001)

        for m in [self.model_base, self.model_L2light, self.model_L2heavy]:
            m.optimizer = torch.optim.SGD(m.parameters(), lr=0.001)  # 使用SGD优化器



    def test_step_noise(self, model, X, Y):
        total_loss = 0
        all_losses = []
        allrun = 0
        model.eval()
        
        for n in range(self.MAXNUM):
            num = n
            if num not in self.numlist:  
                # 添加噪声到输入
                myinput = X[num].reshape(1, -1).float() + torch.randn(1, self.inputn).to(self.device) * 0.3
                tag = Y[num]

                with torch.no_grad():  
                    output = model(myinput)

                loss = self.criterion(output.squeeze(), tag.float())
                total_loss += loss
                all_losses.append(loss.item())  # 将每个 loss 记录到列表中
                allrun += 1

        mean_loss = total_loss / allrun * 10000  # 平均 loss
        std_dev_loss = torch.std(torch.tensor(all_losses)) * 10000  # 标准差（单位一致）

        return mean_loss, np.array(all_losses)

        
    def test_step(self, model, X, Y):
        total_loss = 0
        all_losses = []
        allrun = 0
        model.eval() 
        
        for n in range(self.MAXNUM):
            num = n
            if num not in self.numlist:  
                myinput = X[num].reshape(1, -1).float()
                tag = Y[num]

                with torch.no_grad():  
                    output = model(myinput)

                loss = self.criterion(output.squeeze(), tag.float())
                total_loss += loss
                all_losses.append(loss.item())  # 将每个 loss 记录到列表中
                allrun += 1

        mean_loss = total_loss / allrun * 10000  # 平均 loss
        std_dev_loss = torch.std(torch.tensor(all_losses)) * 10000  # 标准差（单位一致）

        return mean_loss, np.array(all_losses)
      
    def train_step(self, model, X, Y):
        model.train()
        error=0
        for n in range(10000):
            num = np.random.choice(self.numlist) 
            myinput = X[num].reshape(1,-1).float()
            tag = Y[num]

            # Zero the gradients
            model.optimizer.zero_grad()
            # Forward + backward pass and get the gradients
            output = model(myinput)

    #         训练
            loss = self.criterion(output.squeeze(), tag.float())  # 计算损失
            error+=loss
            
            loss.backward()
            # 更新W1，W2
            model.update_weights_custom( lr=0.001)
            # 更新bias
            model.optimizer.step()

        return error

    def train_step_L2(self, model, X, Y):
        model.train()
        error=0
        for n in range(10000):
            num = np.random.choice(self.numlist) 
            myinput = X[num].reshape(1,-1).float()
            tag = Y[num]

            # Zero the gradients
            model.optimizer.zero_grad()
            # Forward + backward pass and get the gradients
            output = model(myinput)

    #         训练
            # 手动添加L2正则化项
            l2_reg = torch.tensor(0.).to(self.device)
            for name, param in model.named_parameters():
                if "W" in name:  # 只对名称包含 "W" 的参数进行正则化
                    l2_reg += torch.norm(param)

            loss = self.criterion(output.squeeze(), tag.float())  
            error += loss
            loss += l2_reg* model.L2punishment # 计算损失

            
            
            
            loss.backward()
            # 更新W1，W2, bias
            model.optimizer.step()

        return error

  

class Classification(Regression):
    def __init__(self, MAXNUM, inputn, train_num, device):
        self.MAXNUM = MAXNUM
        self.inputn = inputn
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  

        
        np.random.seed(5)
        self.numlist = np.random.randint(0,high=MAXNUM,size=train_num)
        
    def train_step(self, model, X, Y):
        model.train()
        right=0
        for n in range(10000):
            num = np.random.choice(self.numlist) 
            myinput = X[num].flatten().reshape(1,-1).float()
            tag = Y[num]

            # Zero the gradients
            model.optimizer.zero_grad()
            # Forward + backward pass and get the gradients
            output = model(myinput)

    #         训练
            #print(output, tag.float())
            loss = self.criterion(output, tag.unsqueeze(0))  # 计算损失
            loss.backward()
            # 更新W1，W2
            model.update_weights_custom( lr=0.001)
            # 更新bias
            model.optimizer.step()

            if output.argmax()== tag:
                right+=1

        return right

    def train_step_L2(self, model, X, Y):
        model.train()
        right=0
        for n in range(10000):
            num = np.random.choice(self.numlist) 
            myinput = X[num].flatten().reshape(1,-1).float()
            tag = Y[num]

            # Zero the gradients
            model.optimizer.zero_grad()
            # Forward + backward pass and get the gradients
            output = model(myinput)

    #         训练
            # 手动添加L2正则化项
            l2_reg = torch.tensor(0.).to(self.device)
            for name, param in model.named_parameters():
                if "W" in name:  # 只对名称包含 "W" 的参数进行正则化
                    l2_reg += torch.norm(param)
            
            loss = self.criterion(output, tag.unsqueeze(0))  + l2_reg* model.L2punishment # 计算损失
            loss.backward()
            # 更新W1，W2, bias
            model.optimizer.step()

            if output.argmax()== tag:
                right+=1

        return right
    
    def test_step(self, model, X, Y):
        all_correct = []  # 用于记录每个样本的正确性
        allrun = 0
        model.eval()
        
        for n in range(self.MAXNUM):
            num = n
            if num not in self.numlist:  
                myinput = X[num].flatten().reshape(1, -1).float()
                tag = Y[num]

                with torch.no_grad():  
                    output = model(myinput)

                # 判断是否正确，并记录结果
                is_correct = int(output.argmax() == tag)  # 1 表示正确，0 表示错误
                all_correct.append(is_correct)
                allrun += 1

        accuracy = sum(all_correct) / allrun * 10000  # 平均正确率
        std_dev_accuracy = torch.std(torch.tensor(all_correct).float()) * 10000  # 标准差

        return accuracy

    
    def test_step_noise(self, model, X, Y):
        all_correct = []  # 用于记录每个样本的正确性
        allrun = 0
        model.eval()
        
        for n in range(self.MAXNUM):
            num = n
            if num not in self.numlist:  
                myinput = X[num].flatten().reshape(1, -1).float() + torch.randn(1, self.inputn).to(self.device) * 0.3
                myinput = myinput.clamp(0, 1)  # 限制输入在 [0,1] 范围
                tag = Y[num]

                with torch.no_grad():  
                    output = model(myinput)

                # 判断是否正确，并记录结果
                is_correct = int(output.argmax() == tag)  # 1 表示正确，0 表示错误
                all_correct.append(is_correct)
                allrun += 1

        accuracy = sum(all_correct) / allrun * 10000  # 平均正确率
        std_dev_accuracy = torch.std(torch.tensor(all_correct).float()) * 10000  # 标准差

        return accuracy
