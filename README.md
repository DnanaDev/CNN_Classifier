# Parkzap Labs Machine Learning Associate Task - Anand

 Task : Object Classification model to identify the model of a car from an image.
 
 <b>Completed tasks 1 and 2 and created dataset (2000 Images) and trained car classifer that achieves 85% accuracy when classifying between Swift/Innova.</b><br>
 
# Binary Car classifier that classifies images of Maruti/Suzuki Swift and Toyota Innova.

 #### A. Findings presented in this readme using model created in PyTorch (conversion of <i>convnet_img_dim_test.ipynb notebook</i>)
 #### B. For just inferencing test using best model - Section 5 of Notebook.
 
## Files of Interest :
1. Unprocessed images - CarImages folder.
2. Best Model - convnet_img_dim_test.ipynb
3. Preprocessed Dataset - Car_classifier_data_120_val.npy
4. Saved model/Checkpoint - checkpoint_acc_0.867_img_size120_batch_50_epochs_20.pt
5. Validation metrics on older model - convnet-Validation_metrics.ipynb
6. Validation using your files - Validate folder and section 5.

# 0. Gathering the data
 
1. Used <b> google_images_download </b> python web scraping module that uses a Selenium and BeautifulSoup backend to get images from google.
2. Download around 1500 images for each class - Swift, Innova.
3. Manually removed irrelevant examples.
4. Final division stood around a <b>1000</b> images for each(Balanced Dataset.)

# 1 Preparing Dataset 
1. Created Dataset and saved in the form of numpy array <b> "Car_classifier_data_120.npy" </b>.
2. <b>Resized images to 120*120 due to GPU VRAM Restrictions.</b>

## 1.1 Loading Images and creating dataset with labels.

1. <b> Suffled data </b> when creating dataset.
2. Classification Ouputs Label in the form of Tensor index.<br>
Tensor [Swift, Innova]



```python
import os
import cv2
import numpy as np
from tqdm import tqdm    # for progress meter in loop


REBUILD_DATA = True   # Set to false if you don't want to rebuild train, test sets for each execution. 
IMG_SIZE = 120

class SwiftVsInnova():
    IMG_SIZE = 120
    Swift = "CarImages/Swift"
    Innova = "CarImages/Innova"
    labels = {Swift : 0, Innova : 1}    
    training_data = []
    swiftcount = 0
    innovacount = 0
    
    def Make_Training_Data(self) :
        for label in self.labels :
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])
                        
                        if label == self.Swift:
                            self.swiftcount +=1
                        elif label == self.Innova:
                            self.innovacount +=1
                    except Exception as e:
                        pass
                        #print(str(e))
                
        np.random.shuffle(self.training_data)
        np.save("Car_classifier_data_120.npy", self.training_data)
        # To check balance of data in terms of target variables.
        print("Innova Count :", self.innovacount)
        print("Swift Count :", self.swiftcount)

if REBUILD_DATA:
    swiftvinnova = SwiftVsInnova()
    swiftvinnova.Make_Training_Data()

```

      1%|▏         | 16/1068 [00:00<00:06, 158.29it/s]

    CarImages/Swift


    100%|██████████| 1068/1068 [00:06<00:00, 165.97it/s]
      2%|▏         | 16/1014 [00:00<00:09, 107.57it/s]

    CarImages/Innova


    100%|██████████| 1014/1014 [00:06<00:00, 125.97it/s]


    Innova Count : 1001
    Swift Count : 1037



```python
training_data = np.load("Car_classifier_data_120.npy", allow_pickle=True)
print(len(training_data))
print(training_data[0][0])
```

    2038
    [[ 78  76  78 ...  72  74  76]
     [ 86  81  81 ...  74  76  90]
     [ 87  84  39 ...  80 111 160]
     ...
     [104 100  92 ... 125 119 117]
     [104  98  93 ... 128 125 121]
     [ 95  96  94 ... 128 127 122]]


## 1.2 Checking dataset 
1. Plotting images from dataset to check conversion, displayed image number.<br>
2. Outputs label in tensor [Swift, Innova]


```python
#%matplotlib notebook
import matplotlib.pyplot as plt
num = np.random.randint(1291)
print(num)
plt.imshow(training_data[num][0], cmap = "gray")
plt.show()
print(training_data[num][1]) 
```

    704



![png](output_6_1.png)


    [1. 0.]


# 2. Building the Model/Passing data in feed forward (Using PyTorch)

The architecture of the network is printed along with the number of parameters.
## Model:
1. 3 Covolution layers, with maxpool and Relu
2. 2 Fully Connected layers
3. Softmax Activation function


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.core.debugger import set_trace # For debugging in Python

# Selecting whether to run on GPU (if available) or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class Net(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # Passing random data x 
        x = torch.randn(120,120).view(-1,1,120,120)
        self._to_linear = None
        self.convs(x)   # Partial forward pass  
        self.fc1 = nn.Linear(self._to_linear, 512) # Flattening , from calculation should be 2*2, 4
        self.fc2 = nn.Linear(512, 2)   
    def convs(self, x):           # Passing data through conv, using Relu activation and maxpool layer 
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,(2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
 # Feed forward propogation   
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
net = Net().to(device)
# Network/Model Architecture 
print(net)
```

    Running on the CPU
    Net(
      (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
      (conv3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=15488, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=2, bias=True)
    )


# 3. Training - Loss and Optimizing parameters
1. Passing the Training data through the Model and calculating loss<b> (Mean Squared Error)</b>. Then using<b> Adam optimizer</b> to adjust parameters/weights of the model.<br>
2. <b>Default learning rate = .001 </b><br>
3. Split into test and train sets 90:10. <b> 1835 images in train, 203 in test </b>. Already shuffled.
4. Trained with<b> batch size 50 (limited by GPU) for 20 EPOCHS</b>.
5. Loss output at each Epoch below.


```python
import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = .001)  #Passing model to optimizer and setting Learning rate
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 120, 120)
X = X/255.0    # Pixel values are between 0 and 255, to transform to 0 - 1 range
y = torch.Tensor([i[1] for i in training_data])

# Creating validation set/ Test set
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT) # coverting to int as it will be used as a number to slice with
print(val_size)
```

    203



```python
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))
```

    1835
    203



```python
BATCH_SIZE = 50 # First thing to reduce if you run into memory errors. If lower than 8, tweak the model
EPOCHS = 20

def train (net) :
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)) :  
            # Splitting in Batches
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,120,120)
            batch_y = train_y[i:i+BATCH_SIZE]
            # Data is a batch of featuremaps and labels
            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # Sending train,test batch to device (GPU if available)
            net.zero_grad()                          # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            outputs = net(batch_X)                   # Passing batch through the network
            loss = loss_function(outputs, batch_y)   # Calculate the loss on output of network and labels Y, using MSE
            loss.backward()                          # Backward propagation of loss/Reducing loss through partial derivatives
            optimizer.step()                         # Adjusts the weight

        print(f"Epoch: {epoch}. Loss: {loss}")

train(net)    

```

    100%|██████████| 37/37 [00:04<00:00,  8.80it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.35it/s]

    Epoch: 0. Loss: 0.24880678951740265


    100%|██████████| 37/37 [00:04<00:00,  8.84it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.29it/s]

    Epoch: 1. Loss: 0.23515813052654266


    100%|██████████| 37/37 [00:04<00:00,  8.85it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.20it/s]

    Epoch: 2. Loss: 0.17049643397331238


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 11.83it/s]

    Epoch: 3. Loss: 0.13410940766334534


    100%|██████████| 37/37 [00:04<00:00,  8.58it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.16it/s]

    Epoch: 4. Loss: 0.12306389212608337


    100%|██████████| 37/37 [00:04<00:00,  8.84it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.23it/s]

    Epoch: 5. Loss: 0.09291809052228928


    100%|██████████| 37/37 [00:04<00:00,  8.84it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.21it/s]

    Epoch: 6. Loss: 0.045314181596040726


    100%|██████████| 37/37 [00:04<00:00,  8.55it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.15it/s]

    Epoch: 7. Loss: 0.029772350564599037


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.20it/s]

    Epoch: 8. Loss: 0.026872942224144936


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.23it/s]

    Epoch: 9. Loss: 0.023383140563964844


    100%|██████████| 37/37 [00:04<00:00,  8.59it/s]
      5%|▌         | 2/37 [00:00<00:02, 11.82it/s]

    Epoch: 10. Loss: 0.07413015514612198


    100%|██████████| 37/37 [00:04<00:00,  8.75it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.20it/s]

    Epoch: 11. Loss: 0.016286632046103477


    100%|██████████| 37/37 [00:04<00:00,  8.84it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.14it/s]

    Epoch: 12. Loss: 0.022982312366366386


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.18it/s]

    Epoch: 13. Loss: 0.0047156005166471004


    100%|██████████| 37/37 [00:04<00:00,  8.82it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.20it/s]

    Epoch: 14. Loss: 0.0023736346047371626


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.16it/s]

    Epoch: 15. Loss: 0.0001548502768855542


    100%|██████████| 37/37 [00:04<00:00,  8.83it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.17it/s]

    Epoch: 16. Loss: 0.00015510951925534755


    100%|██████████| 37/37 [00:04<00:00,  8.69it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.01it/s]

    Epoch: 17. Loss: 8.530705963494256e-05


    100%|██████████| 37/37 [00:04<00:00,  8.81it/s]
      5%|▌         | 2/37 [00:00<00:02, 12.18it/s]

    Epoch: 18. Loss: 3.468523209448904e-05


    100%|██████████| 37/37 [00:04<00:00,  8.80it/s]

    Epoch: 19. Loss: 2.887372647819575e-05


    


# 4. Accuracy - Validating/Testing Model on validation set 

1. <b>Accuracy on test set around 85%. </b>
2. Tried multiple batch sizes, image sizes and training model for more epochs, but model couldn't improve further .
3. Possible regularization problem, either use more data, use dropout etc, better models like resnet. (Not tried in interest of time)


```python
# moving testing tensors to the GPU if available
test_X.to(device)
test_y.to(device)

def test(net):
    correct = 0 
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))) :
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1,1,120,120).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class :
                correct += 1
            total += 1

    accuracy = round(correct/total,3)
    print("Accuracy :", accuracy)
    return accuracy 

accuracy =  test(net)
```

    100%|██████████| 203/203 [00:00<00:00, 554.02it/s]

    Accuracy : 0.867


    


# 4.1 Validation Metrics (*)

1. *From another run of the model, using different parameters.
2. created using <b>convnet-Validation_metrics.ipynb notebook.</b><br>
3. <b>Overfitting can be seen to occur around the 10th EPOCH.</b> <br>
4. True accuracy should be around high 70's or mid 80's.


```python
#%matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name = "model-1579336059" # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.


def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(model_name)
```


![png](output_16_0.png)


# 4.2 Manual Validation

1. Manually validating examples from Test set.<br>
Output tensor index 0 - Maruti Swift <br>
Output tensor index 1 - Toyota Innova
2. Run again for random sample.


```python
## import matplotlib.pyplot as plt
#plt.imshow(X[2].view(50,50))
#%matplotlib notebook

num = np.random.randint(203)
print(num)
plt.imshow(test_X[num].view(120,120), cmap = "gray")
plt.show()
net_out = net(test_X[num].view(-1,1,120,120).to(device))[0]
print(torch.argmax(net_out))
```

    73



![png](output_18_1.png)


    tensor(1)


# 4.3 Print Model Parameters and Save Checkpoint 
1. Printing entire models' learned parameters/weights. 
2. Saved model in the format <b> "checkpoint_acc_img_size_batch_epochs.pt"</b>.
3. Can be used for inferencing directly (section 5.)


```python
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print('')
print('')
print('')

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
torch.save(net, 'checkpoint' + '_acc_' + str(accuracy) + '_img_size' + str(IMG_SIZE) + '_batch_' + str(BATCH_SIZE) + '_epochs_' + str(EPOCHS)+ '.pt')

```

    Model's state_dict:
    conv1.weight 	 torch.Size([32, 1, 5, 5])
    conv1.bias 	 torch.Size([32])
    conv2.weight 	 torch.Size([64, 32, 5, 5])
    conv2.bias 	 torch.Size([64])
    conv3.weight 	 torch.Size([128, 64, 5, 5])
    conv3.bias 	 torch.Size([128])
    fc1.weight 	 torch.Size([512, 15488])
    fc1.bias 	 torch.Size([512])
    fc2.weight 	 torch.Size([2, 512])
    fc2.bias 	 torch.Size([2])
    
    
    
    Optimizer's state_dict:
    state 	 {139973667684784: {'step': 740, 'exp_avg': tensor([[[[-5.1570e-05, -5.2179e-05, -5.4757e-05, -5.7405e-05, -5.3250e-05],
              [-4.3612e-05, -4.7713e-05, -4.9241e-05, -5.0971e-05, -5.1977e-05],
              [-3.3950e-05, -3.3776e-05, -3.1475e-05, -3.0689e-05, -3.1893e-05],
              [-3.0492e-05, -2.9264e-05, -2.7720e-05, -2.3026e-05, -2.1231e-05],
              [-3.7036e-05, -3.6876e-05, -3.2146e-05, -3.0765e-05, -2.6543e-05]]],
    
    
            [[[-3.1953e-05, -3.1850e-05, -3.1031e-05, -3.3735e-05, -3.1728e-05],
              [-3.1160e-05, -3.0947e-05, -3.2521e-05, -3.5592e-05, -3.3885e-05],
              [-3.6008e-05, -3.6112e-05, -3.7442e-05, -3.7781e-05, -3.5542e-05],
              [-3.8356e-05, -3.9452e-05, -3.8130e-05, -3.9742e-05, -3.6608e-05],
              [-3.3919e-05, -3.5543e-05, -3.6788e-05, -3.3122e-05, -3.2196e-05]]],
    
    
            [[[-4.9680e-05, -4.4164e-05, -4.2662e-05, -4.4981e-05, -4.9401e-05],
              [-7.7436e-05, -8.2757e-05, -9.4765e-05, -9.9969e-05, -1.0276e-04],
              [-7.7354e-05, -8.4235e-05, -8.6343e-05, -8.5746e-05, -8.2300e-05],
              [-4.3959e-05, -4.1191e-05, -3.7284e-05, -3.9534e-05, -4.3892e-05],
              [-4.8268e-05, -4.4388e-05, -4.2680e-05, -4.3559e-05, -4.7693e-05]]],
    
    
            [[[-6.3868e-06, -8.8266e-06, -9.7634e-06, -1.0720e-05, -9.7080e-06],
              [-4.9316e-06, -6.7875e-06, -6.3884e-06, -8.2253e-06, -1.2150e-05],
              [-1.7558e-05, -2.1733e-05, -2.1649e-05, -1.9202e-05, -1.6638e-05],
              [-1.3638e-05, -1.5652e-05, -1.4009e-05, -1.0791e-05, -9.9255e-06],
              [-1.0926e-05, -1.3589e-05, -1.3578e-05, -1.3373e-05, -1.5154e-05]]],
    
    
            [[[-5.2088e-05, -5.3658e-05, -5.2304e-05, -5.3009e-05, -5.1498e-05],
              [-4.7512e-05, -5.1843e-05, -5.0327e-05, -5.2175e-05, -4.9700e-05],
              [-4.4986e-05, -4.7007e-05, -4.6917e-05, -4.7029e-05, -4.2645e-05],
              [-3.3281e-05, -3.2757e-05, -3.0891e-05, -2.8814e-05, -2.6324e-05],
              [-3.2224e-05, -3.2300e-05, -3.0627e-05, -2.8392e-05, -2.6055e-05]]],
    
    
            [[[-3.7023e-05, -3.8088e-05, -3.9866e-05, -3.6763e-05, -3.8795e-05],
              [-3.3791e-05, -3.5417e-05, -3.7636e-05, -3.4757e-05, -3.4973e-05],
              [-3.5277e-05, -3.4171e-05, -3.4651e-05, -3.2777e-05, -3.3789e-05],
              [-3.4894e-05, -3.6177e-05, -3.6245e-05, -3.4060e-05, -3.2033e-05],
              [-3.4643e-05, -3.4501e-05, -3.3787e-05, -3.2112e-05, -2.8400e-05]]],
    
    
            [[[-7.8731e-06, -7.5746e-06, -1.0443e-05, -1.1969e-05, -1.0715e-05],
              [-3.7540e-06, -3.3642e-06, -4.8247e-06, -8.8487e-06, -8.1844e-06],
              [-1.4053e-06, -1.0007e-07,  1.6123e-06, -1.2898e-06, -1.8803e-06],
              [-5.7183e-06, -1.8956e-06, -1.1101e-06, -2.9597e-06, -1.4965e-06],
              [-2.7777e-06, -3.6586e-06, -2.3691e-07, -2.4638e-06, -1.0423e-06]]],
    
    
            [[[-6.1031e-05, -6.9512e-05, -7.0725e-05, -7.7755e-05, -7.5589e-05],
              [-3.0073e-05, -2.8666e-05, -2.9339e-05, -3.1495e-05, -3.3097e-05],
              [-1.3337e-05, -8.6776e-06, -7.5775e-06, -1.3866e-05, -1.7234e-05],
              [-4.1569e-05, -3.7745e-05, -3.5959e-05, -3.0426e-05, -2.5877e-05],
              [-4.4786e-05, -4.6160e-05, -4.3878e-05, -4.2795e-05, -4.4690e-05]]],
    
    
            [[[-1.1700e-04, -1.1456e-04, -1.1106e-04, -1.1055e-04, -1.1287e-04],
              [-1.1582e-04, -1.3290e-04, -1.3483e-04, -1.3188e-04, -1.3208e-04],
              [-7.8789e-05, -7.1351e-05, -6.9935e-05, -6.4625e-05, -6.8494e-05],
              [-5.0172e-05, -4.5543e-05, -4.7036e-05, -5.2767e-05, -5.7877e-05],
              [-8.8715e-05, -9.3175e-05, -9.2790e-05, -9.8722e-05, -9.3178e-05]]],
    
    
            [[[-1.5113e-05, -1.3997e-05, -1.0859e-05, -8.7135e-06, -7.0783e-06],
              [-1.3755e-05, -1.3332e-05, -9.3624e-06, -6.8288e-06, -6.1235e-06],
              [-7.8392e-06, -8.1340e-06, -4.4928e-06, -2.7965e-06, -1.2821e-06],
              [-6.0724e-06, -4.5207e-06, -3.1365e-06, -8.5038e-07, -5.7958e-07],
              [-5.7892e-06, -4.3573e-06, -2.8439e-06, -2.2715e-06, -1.9010e-06]]],
    
    
            [[[-1.3436e-05, -8.8935e-06, -4.8515e-06, -5.9297e-06, -4.7222e-06],
              [-9.6722e-06, -3.4030e-06, -6.0519e-07, -3.2481e-06, -2.1518e-06],
              [-9.0946e-06, -5.3565e-06, -3.0100e-06, -3.6346e-06, -2.6511e-06],
              [-8.7099e-06, -4.3414e-06, -4.2137e-06, -5.3026e-06, -4.0597e-06],
              [-6.6346e-06, -5.0890e-06, -4.8864e-06, -6.7218e-06, -4.5625e-06]]],
    
    
            [[[-5.9639e-05, -6.0262e-05, -6.0290e-05, -6.1249e-05, -5.6801e-05],
              [-2.9865e-05, -2.7461e-05, -2.8518e-05, -2.9368e-05, -2.9789e-05],
              [-3.4636e-05, -3.6435e-05, -3.3775e-05, -3.0125e-05, -2.6631e-05],
              [-3.2914e-05, -3.0945e-05, -2.3772e-05, -2.1817e-05, -2.1590e-05],
              [-2.5049e-05, -2.6257e-05, -2.5088e-05, -2.9027e-05, -2.9808e-05]]],
    
    
            [[[ 1.3298e-05,  7.3201e-06,  9.3310e-06,  5.9583e-06,  5.9204e-06],
              [ 1.3041e-05,  1.4060e-05,  1.0091e-05,  1.1989e-05,  5.7663e-06],
              [ 1.2602e-05,  1.3903e-05,  1.0756e-05,  1.2151e-05,  8.5551e-06],
              [ 1.3498e-05,  1.1701e-05,  8.5436e-06,  9.3790e-06,  5.2823e-06],
              [ 1.8095e-05,  2.0416e-05,  1.7547e-05,  1.6274e-05,  1.2808e-05]]],
    
    
            [[[-4.2686e-05, -4.1483e-05, -4.6457e-05, -4.2411e-05, -4.0753e-05],
              [-1.7129e-05, -1.8632e-05, -2.4155e-05, -2.2155e-05, -2.6923e-05],
              [-1.8637e-05, -2.2882e-05, -2.4863e-05, -2.1590e-05, -2.2857e-05],
              [-1.2674e-05, -1.4555e-05, -1.0283e-05, -9.5342e-06, -1.2221e-05],
              [-2.0761e-05, -2.4357e-05, -2.0179e-05, -1.9724e-05, -1.9229e-05]]],
    
    
            [[[-1.2686e-05, -1.3799e-05, -1.2500e-05, -8.7925e-06, -6.8648e-06],
              [-9.8980e-06, -1.0330e-05, -7.4928e-06, -6.6269e-06, -4.7254e-06],
              [-9.5995e-06, -8.5545e-06, -5.5004e-06, -2.6240e-06, -4.5935e-06],
              [-9.3942e-06, -8.1006e-06, -4.7191e-06, -3.0192e-06, -2.2622e-06],
              [-7.3857e-06, -5.5391e-06, -4.4403e-06, -2.0701e-06, -2.3733e-06]]],
    
    
            [[[-1.6730e-05, -1.5985e-05, -1.6453e-05, -1.7625e-05, -1.7690e-05],
              [-1.8829e-05, -1.7915e-05, -1.8307e-05, -1.8646e-05, -1.4570e-05],
              [-2.2965e-05, -2.3177e-05, -2.2449e-05, -2.0678e-05, -1.6555e-05],
              [-1.8255e-05, -1.8520e-05, -2.0209e-05, -2.0126e-05, -1.8042e-05],
              [-2.1280e-05, -2.2178e-05, -1.9702e-05, -2.0124e-05, -1.7524e-05]]],
    
    
            [[[-5.9814e-05, -6.1516e-05, -6.3830e-05, -6.6738e-05, -6.7770e-05],
              [-5.7527e-05, -5.3503e-05, -5.5711e-05, -4.9148e-05, -4.7684e-05],
              [-2.7571e-05, -2.7752e-05, -3.1286e-05, -3.4706e-05, -4.0563e-05],
              [-2.3541e-05, -2.3742e-05, -2.4260e-05, -2.9897e-05, -3.1349e-05],
              [-3.3989e-05, -3.6413e-05, -3.5872e-05, -3.4584e-05, -3.2798e-05]]],
    
    
            [[[-1.6064e-05, -1.7772e-05, -1.6675e-05, -1.5559e-05, -1.5158e-05],
              [-1.5294e-05, -1.6403e-05, -1.4543e-05, -1.7489e-05, -1.6624e-05],
              [-9.5138e-06, -7.8648e-06, -8.6803e-06, -1.1259e-05, -1.0653e-05],
              [-6.0656e-06, -6.1702e-06, -5.4802e-06, -8.0462e-06, -6.5285e-06],
              [-5.8676e-06, -5.1015e-06, -3.8768e-06, -5.3647e-06, -5.1144e-06]]],
    
    
            [[[-7.4252e-06, -8.1753e-06, -4.2477e-06, -3.7480e-06, -2.3417e-06],
              [-1.1636e-05, -1.1474e-05, -8.8461e-06, -5.6458e-06,  7.3338e-07],
              [-7.3133e-06, -4.6274e-06, -3.2564e-06,  4.5703e-07,  2.8500e-06],
              [-4.0392e-06, -9.1392e-07, -1.5952e-06,  3.4716e-06,  4.1762e-06],
              [-8.1162e-06, -2.9613e-06,  6.3828e-08,  2.6764e-06,  6.0992e-06]]],
    
    
            [[[-5.0641e-05, -5.2561e-05, -4.8040e-05, -4.3144e-05, -4.5144e-05],
              [-6.1331e-05, -6.0827e-05, -5.5810e-05, -5.8778e-05, -6.3083e-05],
              [-5.7250e-05, -6.8240e-05, -7.0410e-05, -6.8028e-05, -6.7829e-05],
              [-3.2635e-05, -3.2535e-05, -3.0438e-05, -3.2861e-05, -3.7519e-05],
              [-2.9022e-05, -2.6102e-05, -2.2322e-05, -1.8501e-05, -2.3121e-05]]],
    
    
            [[[-2.6633e-06, -3.0411e-06, -2.0402e-06, -1.9379e-07, -1.6330e-06],
              [-1.8006e-06, -2.5852e-06, -2.3485e-06,  4.5356e-07,  7.2488e-07],
              [-3.4423e-06, -3.3614e-06, -2.2029e-06, -1.6644e-06,  3.5169e-07],
              [-1.2382e-06, -3.9281e-07, -1.6987e-08,  2.1204e-07,  1.1093e-06],
              [-1.7776e-06, -2.0604e-06, -1.4858e-06,  9.4220e-08, -1.1648e-07]]],
    
    
            [[[ 6.2155e-06,  6.7323e-06,  5.8806e-06,  6.3903e-06,  4.7489e-06],
              [ 6.7004e-06,  6.8864e-06,  8.3435e-06,  7.9753e-06,  6.4084e-06],
              [ 6.7783e-06,  4.7186e-06,  3.6292e-06,  3.5796e-06,  5.1519e-06],
              [ 4.7611e-06,  1.6893e-06,  3.0969e-06,  2.9099e-06,  4.9839e-06],
              [ 1.0384e-05,  7.3768e-06,  8.4131e-06,  7.0984e-06,  8.3050e-06]]],
    
    
            [[[-9.7645e-06, -1.0799e-05, -1.2201e-05, -1.3593e-05, -1.2756e-05],
              [-9.4007e-06, -1.1195e-05, -1.5160e-05, -1.4607e-05, -1.4043e-05],
              [-9.1464e-06, -1.0435e-05, -1.2557e-05, -1.2910e-05, -1.2938e-05],
              [-6.4364e-06, -6.2748e-06, -7.8997e-06, -6.4866e-06, -5.3828e-06],
              [-5.0865e-06, -6.5950e-06, -5.8195e-06, -4.9676e-06, -3.6268e-06]]],
    
    
            [[[-4.0576e-06, -1.4035e-07,  4.8351e-06,  6.7419e-06,  3.6191e-06],
              [ 3.6789e-07,  5.0944e-06,  5.6845e-06,  6.9331e-06,  5.7588e-06],
              [-7.1437e-06, -5.0437e-06,  4.7642e-06,  5.7943e-06,  7.9304e-06],
              [-6.2642e-07,  2.6331e-06,  2.5421e-06,  1.9616e-06,  2.4216e-06],
              [ 5.4748e-06,  6.2457e-06,  7.8013e-06,  7.1084e-06,  1.0207e-05]]],
    
    
            [[[-6.9561e-05, -7.1170e-05, -6.4680e-05, -6.2555e-05, -6.2667e-05],
              [-8.4947e-05, -8.7709e-05, -8.5960e-05, -8.8145e-05, -9.3955e-05],
              [-7.9863e-05, -9.0277e-05, -9.8563e-05, -1.0065e-04, -9.9737e-05],
              [-4.2784e-05, -4.3015e-05, -4.1045e-05, -4.1905e-05, -5.0193e-05],
              [-5.2472e-05, -5.3272e-05, -5.3438e-05, -5.3603e-05, -5.4432e-05]]],
    
    
            [[[-3.4515e-05, -3.5289e-05, -3.5125e-05, -3.0959e-05, -3.1330e-05],
              [-3.0493e-05, -2.7105e-05, -2.5415e-05, -2.2195e-05, -2.4196e-05],
              [-1.8947e-05, -1.7955e-05, -1.9183e-05, -1.5779e-05, -1.5146e-05],
              [-1.4862e-05, -1.3042e-05, -1.2882e-05, -1.1458e-05, -1.0238e-05],
              [-1.2663e-05, -1.1398e-05, -1.1860e-05, -1.1359e-05, -9.8522e-06]]],
    
    
            [[[-1.0762e-04, -1.0773e-04, -1.0838e-04, -1.0795e-04, -1.0654e-04],
              [-1.2768e-04, -1.2888e-04, -1.3203e-04, -1.3043e-04, -1.2927e-04],
              [-1.5223e-04, -1.5572e-04, -1.5852e-04, -1.6078e-04, -1.6290e-04],
              [-1.4046e-04, -1.4350e-04, -1.4124e-04, -1.4107e-04, -1.3581e-04],
              [-1.0206e-04, -9.8420e-05, -1.0021e-04, -1.0146e-04, -1.0571e-04]]],
    
    
            [[[-6.3976e-05, -6.1289e-05, -5.9124e-05, -5.4608e-05, -5.1726e-05],
              [-5.1103e-05, -4.9075e-05, -5.0566e-05, -4.7619e-05, -5.1124e-05],
              [-3.6592e-05, -3.6017e-05, -3.2486e-05, -2.9784e-05, -3.4006e-05],
              [-2.2451e-05, -2.0026e-05, -1.6897e-05, -1.5883e-05, -2.0863e-05],
              [-3.4337e-05, -3.1213e-05, -2.8177e-05, -2.7726e-05, -2.6766e-05]]],
    
    
            [[[ 2.8822e-06,  2.0491e-06,  9.0827e-07,  1.0442e-06,  2.9427e-07],
              [ 1.5296e-06,  2.6608e-06,  8.6365e-07,  2.0374e-06,  2.3498e-06],
              [ 1.3835e-06,  1.0451e-06, -1.1965e-07,  1.6149e-06,  3.6045e-06],
              [-2.2152e-07,  4.0430e-07, -9.2480e-07,  2.0382e-06,  2.1938e-06],
              [-1.7062e-06, -3.9146e-07, -4.3139e-07, -6.4998e-07,  4.8895e-07]]],
    
    
            [[[ 9.2681e-06,  1.0865e-05,  1.3129e-05,  1.3945e-05,  1.7250e-05],
              [ 1.0005e-05,  1.3652e-05,  1.6059e-05,  1.3594e-05,  1.4304e-05],
              [-8.6044e-07,  2.1141e-07,  2.0087e-07, -4.4679e-07,  4.5894e-06],
              [ 3.6983e-06,  5.9267e-06,  8.8556e-06,  1.2017e-05,  1.1263e-05],
              [ 4.8838e-06,  4.4850e-06,  5.6334e-06,  6.7387e-06,  9.9663e-06]]],
    
    
            [[[-2.7509e-05, -2.8213e-05, -2.8374e-05, -2.7318e-05, -2.4989e-05],
              [-1.6956e-05, -1.6482e-05, -1.6974e-05, -1.5323e-05, -1.5437e-05],
              [-1.6802e-05, -1.6230e-05, -1.4565e-05, -1.4640e-05, -1.3342e-05],
              [-1.2173e-05, -1.0555e-05, -9.2907e-06, -9.1478e-06, -8.9926e-06],
              [-1.0346e-05, -9.3883e-06, -8.8061e-06, -7.7167e-06, -8.2877e-06]]],
    
    
            [[[-5.2608e-05, -4.9034e-05, -4.6178e-05, -3.9026e-05, -3.9463e-05],
              [-3.9046e-05, -3.2223e-05, -2.9456e-05, -2.9525e-05, -3.4017e-05],
              [-2.7260e-05, -2.7314e-05, -1.9916e-05, -2.0091e-05, -2.3169e-05],
              [-3.4262e-05, -3.5909e-05, -3.1532e-05, -3.3909e-05, -3.2187e-05],
              [-2.8231e-05, -2.7104e-05, -2.7922e-05, -3.0027e-05, -3.2257e-05]]]],
           device='cuda:0'), 'exp_avg_sq': tensor([[[[1.1266e-04, 1.0634e-04, 1.1809e-04, 1.2625e-04, 1.1833e-04],
              [7.5742e-05, 8.8020e-05, 9.6825e-05, 1.0724e-04, 1.1116e-04],
              [4.9834e-05, 5.5794e-05, 4.2165e-05, 4.0945e-05, 4.1732e-05],
              [4.7955e-05, 3.4461e-05, 3.4674e-05, 2.4655e-05, 2.2309e-05],
              [6.0973e-05, 4.9895e-05, 4.3143e-05, 3.8853e-05, 3.0476e-05]]],
    
    
            [[[1.2971e-04, 1.1757e-04, 1.2601e-04, 1.2463e-04, 1.5329e-04],
              [1.4199e-04, 1.4852e-04, 1.5342e-04, 1.6433e-04, 1.6262e-04],
              [1.5322e-04, 1.8454e-04, 1.9354e-04, 1.8784e-04, 2.0486e-04],
              [1.5199e-04, 1.7050e-04, 1.8715e-04, 1.8512e-04, 2.0486e-04],
              [1.8514e-04, 1.8004e-04, 1.7515e-04, 1.9091e-04, 2.0232e-04]]],
    
    
            [[[4.0289e-04, 3.7106e-04, 3.6577e-04, 3.5188e-04, 4.0347e-04],
              [5.4234e-04, 6.7802e-04, 7.3082e-04, 7.8166e-04, 7.9048e-04],
              [5.5433e-04, 6.7282e-04, 6.8103e-04, 5.4795e-04, 5.4931e-04],
              [3.5106e-04, 2.7358e-04, 2.3264e-04, 1.8350e-04, 2.2587e-04],
              [3.4255e-04, 2.3747e-04, 1.8580e-04, 1.8136e-04, 2.0613e-04]]],
    
    
            [[[5.3321e-05, 4.6650e-05, 4.8426e-05, 5.9235e-05, 5.6722e-05],
              [4.0397e-05, 3.4247e-05, 3.4799e-05, 3.7997e-05, 5.0115e-05],
              [5.2973e-05, 4.9074e-05, 5.2570e-05, 4.9739e-05, 4.5146e-05],
              [4.4770e-05, 4.3989e-05, 3.8402e-05, 3.5054e-05, 3.6716e-05],
              [4.6051e-05, 4.4023e-05, 4.5290e-05, 4.8642e-05, 4.6371e-05]]],
    
    
            [[[7.8459e-05, 8.9493e-05, 9.1500e-05, 9.5140e-05, 9.4333e-05],
              [6.7484e-05, 8.1049e-05, 8.4331e-05, 7.4874e-05, 7.9258e-05],
              [5.8723e-05, 6.0490e-05, 6.4883e-05, 6.8707e-05, 6.2134e-05],
              [3.1524e-05, 2.6298e-05, 2.1566e-05, 2.3457e-05, 2.3378e-05],
              [2.6986e-05, 2.5730e-05, 2.2631e-05, 2.9120e-05, 2.2898e-05]]],
    
    
            [[[1.0346e-04, 1.1259e-04, 1.2684e-04, 1.2149e-04, 1.0133e-04],
              [1.2077e-04, 1.1802e-04, 1.1946e-04, 1.2419e-04, 1.0156e-04],
              [1.0627e-04, 1.2293e-04, 1.3097e-04, 1.2325e-04, 9.8585e-05],
              [8.7878e-05, 1.0008e-04, 9.2461e-05, 8.3443e-05, 9.0558e-05],
              [8.5805e-05, 9.5827e-05, 1.0355e-04, 9.4866e-05, 9.9566e-05]]],
    
    
            [[[2.3418e-05, 1.8465e-05, 2.5950e-05, 2.7314e-05, 3.1285e-05],
              [1.5332e-05, 1.4052e-05, 1.6458e-05, 2.1131e-05, 1.9921e-05],
              [1.1221e-05, 6.9014e-06, 6.1291e-06, 9.0288e-06, 9.0778e-06],
              [1.3839e-05, 8.2363e-06, 7.4141e-06, 1.0109e-05, 8.9125e-06],
              [8.0127e-06, 8.9644e-06, 7.2994e-06, 1.0866e-05, 1.4401e-05]]],
    
    
            [[[8.6141e-04, 9.3604e-04, 8.3688e-04, 9.6903e-04, 9.9155e-04],
              [3.9053e-04, 3.7227e-04, 4.4306e-04, 4.0002e-04, 4.5468e-04],
              [2.1900e-04, 2.0335e-04, 2.1107e-04, 2.0587e-04, 2.2443e-04],
              [3.7503e-04, 3.5125e-04, 3.4824e-04, 2.9042e-04, 2.7757e-04],
              [3.7189e-04, 4.7117e-04, 4.6584e-04, 4.0174e-04, 4.2383e-04]]],
    
    
            [[[2.0066e-03, 1.7436e-03, 1.5768e-03, 1.5519e-03, 1.6382e-03],
              [1.3743e-03, 1.8520e-03, 1.8833e-03, 1.8725e-03, 1.8774e-03],
              [6.6769e-04, 7.3278e-04, 9.1199e-04, 7.4090e-04, 8.1300e-04],
              [4.0935e-04, 4.2822e-04, 4.9837e-04, 6.0023e-04, 6.3958e-04],
              [6.7495e-04, 7.9637e-04, 8.4305e-04, 1.0029e-03, 9.3485e-04]]],
    
    
            [[[1.1293e-05, 1.1443e-05, 8.4505e-06, 9.8206e-06, 9.8483e-06],
              [1.7698e-05, 1.7568e-05, 1.2314e-05, 8.4222e-06, 8.9841e-06],
              [1.9307e-05, 1.4163e-05, 9.0551e-06, 7.9501e-06, 1.1409e-05],
              [1.7650e-05, 1.5098e-05, 1.4261e-05, 1.2397e-05, 1.2191e-05],
              [2.7018e-05, 2.0761e-05, 2.1584e-05, 2.1350e-05, 2.0767e-05]]],
    
    
            [[[3.7369e-05, 8.1114e-06, 3.6369e-06, 3.6896e-06, 3.5409e-06],
              [3.8054e-05, 1.4255e-05, 8.1347e-06, 4.4799e-06, 4.4313e-06],
              [1.1598e-05, 6.4821e-06, 4.0200e-06, 3.4549e-06, 2.7699e-06],
              [1.1563e-05, 4.5294e-06, 2.9868e-06, 4.0334e-06, 2.6964e-06],
              [8.0983e-06, 6.1320e-06, 5.7900e-06, 7.3672e-06, 4.3235e-06]]],
    
    
            [[[2.8224e-04, 2.7558e-04, 2.7489e-04, 3.0463e-04, 3.2998e-04],
              [1.1835e-04, 1.3167e-04, 1.3199e-04, 1.4665e-04, 1.4118e-04],
              [1.3298e-04, 1.7484e-04, 1.3408e-04, 1.0526e-04, 1.0514e-04],
              [1.1268e-04, 1.0101e-04, 7.2041e-05, 6.3462e-05, 7.7104e-05],
              [7.0680e-05, 8.2567e-05, 9.0144e-05, 1.1880e-04, 9.7893e-05]]],
    
    
            [[[9.9299e-05, 1.2731e-04, 1.1519e-04, 1.5771e-04, 1.6030e-04],
              [5.5854e-05, 5.8192e-05, 6.4604e-05, 9.0888e-05, 1.3423e-04],
              [5.7911e-05, 5.0099e-05, 5.8781e-05, 5.5771e-05, 7.4354e-05],
              [6.1595e-05, 6.8531e-05, 6.3750e-05, 6.6130e-05, 9.8600e-05],
              [1.2994e-04, 1.0408e-04, 1.2589e-04, 1.1884e-04, 1.1971e-04]]],
    
    
            [[[1.3272e-04, 1.0903e-04, 1.3887e-04, 1.1860e-04, 1.0903e-04],
              [4.7292e-05, 5.1572e-05, 6.7635e-05, 5.2975e-05, 5.9963e-05],
              [3.9032e-05, 4.9921e-05, 5.6070e-05, 4.7870e-05, 4.4983e-05],
              [2.9281e-05, 3.9140e-05, 3.1935e-05, 2.9770e-05, 3.0458e-05],
              [4.8848e-05, 6.6348e-05, 5.9971e-05, 5.5719e-05, 4.8745e-05]]],
    
    
            [[[4.0363e-05, 4.9006e-05, 4.9945e-05, 4.8840e-05, 4.7176e-05],
              [3.6490e-05, 4.3226e-05, 4.0200e-05, 4.2759e-05, 4.4080e-05],
              [3.9271e-05, 3.7683e-05, 3.5212e-05, 3.1121e-05, 3.5530e-05],
              [2.9958e-05, 2.9833e-05, 2.7922e-05, 2.4429e-05, 2.6220e-05],
              [2.7924e-05, 2.3626e-05, 2.5766e-05, 2.4125e-05, 2.5988e-05]]],
    
    
            [[[1.2910e-04, 1.3298e-04, 1.3710e-04, 1.1662e-04, 1.0890e-04],
              [1.0757e-04, 1.2433e-04, 1.1972e-04, 1.0982e-04, 1.3050e-04],
              [9.8854e-05, 1.1153e-04, 1.0034e-04, 1.0061e-04, 1.0845e-04],
              [1.1904e-04, 1.1579e-04, 1.0988e-04, 9.2123e-05, 9.1667e-05],
              [9.9848e-05, 9.3939e-05, 9.2969e-05, 9.5587e-05, 9.5530e-05]]],
    
    
            [[[5.0398e-04, 4.5311e-04, 5.2092e-04, 5.8625e-04, 5.4045e-04],
              [4.2020e-04, 3.7179e-04, 3.7498e-04, 3.2842e-04, 2.9331e-04],
              [1.4354e-04, 1.6748e-04, 1.7910e-04, 2.1883e-04, 2.8066e-04],
              [8.6767e-05, 9.0671e-05, 1.0528e-04, 1.6973e-04, 2.0127e-04],
              [1.4566e-04, 2.1541e-04, 1.8694e-04, 1.9317e-04, 2.0387e-04]]],
    
    
            [[[2.2190e-05, 2.5210e-05, 2.5924e-05, 2.9127e-05, 2.6324e-05],
              [1.5524e-05, 1.8643e-05, 2.1891e-05, 2.3091e-05, 1.9357e-05],
              [1.0196e-05, 7.6815e-06, 9.0718e-06, 1.2283e-05, 1.2000e-05],
              [6.0635e-06, 4.3351e-06, 5.0031e-06, 9.6355e-06, 1.1317e-05],
              [3.3335e-06, 3.0496e-06, 3.6642e-06, 7.0122e-06, 8.0355e-06]]],
    
    
            [[[2.3593e-04, 2.2130e-04, 2.4079e-04, 1.8967e-04, 1.3361e-04],
              [1.8061e-04, 1.8005e-04, 1.4221e-04, 1.4290e-04, 1.0589e-04],
              [1.6959e-04, 1.9081e-04, 1.7214e-04, 1.2614e-04, 9.6766e-05],
              [1.1017e-04, 1.0381e-04, 1.1637e-04, 1.0747e-04, 6.6757e-05],
              [7.7310e-05, 8.8579e-05, 7.2489e-05, 6.5591e-05, 7.8479e-05]]],
    
    
            [[[4.1560e-04, 4.2239e-04, 3.8712e-04, 3.3618e-04, 3.1451e-04],
              [4.5739e-04, 4.5611e-04, 3.7338e-04, 4.1320e-04, 4.8517e-04],
              [3.2202e-04, 4.4809e-04, 4.3497e-04, 3.8382e-04, 4.5682e-04],
              [1.6953e-04, 1.5274e-04, 1.3563e-04, 1.7882e-04, 2.2490e-04],
              [1.2553e-04, 1.0412e-04, 7.4063e-05, 6.6547e-05, 9.4924e-05]]],
    
    
            [[[1.0229e-05, 9.5804e-06, 1.2034e-05, 4.5379e-06, 3.1305e-06],
              [9.9207e-06, 1.5787e-05, 1.4415e-05, 8.2810e-06, 5.5222e-06],
              [8.5352e-06, 1.1508e-05, 1.9835e-05, 1.5042e-05, 1.0176e-05],
              [1.6842e-05, 1.2794e-05, 1.0630e-05, 1.5393e-05, 8.7141e-06],
              [7.3023e-06, 5.5072e-06, 5.4394e-06, 3.6147e-06, 1.9821e-06]]],
    
    
            [[[2.8480e-05, 1.6574e-05, 1.6638e-05, 2.1446e-05, 1.7401e-05],
              [4.7057e-05, 5.0754e-05, 4.9636e-05, 3.7762e-05, 3.1092e-05],
              [9.6231e-05, 6.1119e-05, 5.9463e-05, 4.4653e-05, 5.0949e-05],
              [9.1539e-05, 9.8549e-05, 9.7175e-05, 1.0357e-04, 1.0210e-04],
              [1.5711e-04, 1.3024e-04, 1.8108e-04, 1.6716e-04, 1.2370e-04]]],
    
    
            [[[1.4380e-05, 1.2433e-05, 1.2027e-05, 1.2661e-05, 8.8891e-06],
              [7.6688e-06, 9.1385e-06, 1.0342e-05, 1.0815e-05, 9.5433e-06],
              [8.3455e-06, 9.1362e-06, 1.1004e-05, 1.0461e-05, 8.1692e-06],
              [7.3394e-06, 8.9874e-06, 9.1207e-06, 6.7858e-06, 5.5429e-06],
              [7.1147e-06, 9.1885e-06, 8.2338e-06, 7.1205e-06, 5.1023e-06]]],
    
    
            [[[4.5671e-04, 4.3982e-04, 2.1216e-04, 1.7725e-04, 1.9687e-04],
              [4.3679e-04, 3.7520e-04, 1.8580e-04, 1.2755e-04, 1.2927e-04],
              [2.1307e-04, 2.2539e-04, 1.4292e-04, 2.0155e-04, 1.2111e-04],
              [2.5712e-04, 3.8609e-04, 2.6758e-04, 1.4151e-04, 1.0796e-04],
              [5.5584e-04, 6.1482e-04, 3.2372e-04, 3.7019e-04, 2.5152e-04]]],
    
    
            [[[1.0668e-03, 8.2876e-04, 6.8130e-04, 6.8363e-04, 6.9883e-04],
              [1.0707e-03, 1.0571e-03, 8.7804e-04, 9.2732e-04, 1.0343e-03],
              [6.8047e-04, 8.4046e-04, 9.5116e-04, 1.0713e-03, 1.0677e-03],
              [3.1693e-04, 3.8145e-04, 2.9287e-04, 3.1435e-04, 4.6606e-04],
              [3.4826e-04, 4.2612e-04, 3.5258e-04, 3.0359e-04, 4.1230e-04]]],
    
    
            [[[2.4746e-05, 2.6901e-05, 2.9269e-05, 2.6880e-05, 2.8647e-05],
              [2.2334e-05, 1.9631e-05, 1.8280e-05, 1.4668e-05, 1.7159e-05],
              [1.2283e-05, 1.0686e-05, 1.2320e-05, 9.0542e-06, 8.7967e-06],
              [8.3148e-06, 5.8293e-06, 6.9359e-06, 7.4120e-06, 5.7400e-06],
              [5.7416e-06, 5.3947e-06, 6.8738e-06, 6.5161e-06, 6.2870e-06]]],
    
    
            [[[5.0618e-04, 6.1250e-04, 5.3542e-04, 5.0987e-04, 4.1756e-04],
              [6.7943e-04, 7.5690e-04, 7.1590e-04, 6.4986e-04, 5.6931e-04],
              [8.3584e-04, 1.0834e-03, 1.1142e-03, 9.5702e-04, 9.2070e-04],
              [6.2807e-04, 7.8313e-04, 7.2191e-04, 6.9180e-04, 5.6897e-04],
              [3.0312e-04, 2.6749e-04, 3.0575e-04, 3.0918e-04, 3.2121e-04]]],
    
    
            [[[3.3297e-04, 3.0598e-04, 3.0207e-04, 2.9377e-04, 3.0195e-04],
              [2.4421e-04, 1.8982e-04, 2.3511e-04, 2.2366e-04, 2.5869e-04],
              [1.3108e-04, 1.1901e-04, 1.0769e-04, 1.0279e-04, 1.5355e-04],
              [5.8549e-05, 4.9093e-05, 5.5339e-05, 5.3621e-05, 8.6583e-05],
              [1.1467e-04, 1.0459e-04, 9.3439e-05, 9.0948e-05, 9.3365e-05]]],
    
    
            [[[2.5226e-05, 2.1667e-05, 2.4174e-05, 2.8049e-05, 3.9781e-05],
              [2.7751e-05, 2.5396e-05, 3.9886e-05, 4.1679e-05, 3.3934e-05],
              [1.2113e-05, 1.1481e-05, 1.3450e-05, 1.1839e-05, 1.2810e-05],
              [1.7353e-05, 2.1493e-05, 3.0192e-05, 1.7562e-05, 1.8048e-05],
              [2.5384e-05, 1.8966e-05, 1.8463e-05, 1.7054e-05, 1.4467e-05]]],
    
    
            [[[2.4533e-04, 2.2768e-04, 2.3973e-04, 2.2176e-04, 2.5511e-04],
              [2.0314e-04, 2.2961e-04, 2.2587e-04, 2.1359e-04, 2.1406e-04],
              [1.5925e-04, 1.5889e-04, 1.3151e-04, 1.3620e-04, 1.6822e-04],
              [1.3960e-04, 1.5791e-04, 1.7428e-04, 1.7963e-04, 1.5748e-04],
              [1.5537e-04, 1.3760e-04, 1.2842e-04, 1.5254e-04, 1.6482e-04]]],
    
    
            [[[5.1825e-05, 5.5768e-05, 5.3185e-05, 4.9347e-05, 5.0884e-05],
              [2.5449e-05, 2.9021e-05, 2.7439e-05, 2.2912e-05, 2.6763e-05],
              [2.2416e-05, 2.5946e-05, 2.0956e-05, 2.1011e-05, 2.6408e-05],
              [1.3004e-05, 1.7830e-05, 1.4845e-05, 1.3362e-05, 1.8795e-05],
              [1.3889e-05, 1.8317e-05, 1.4151e-05, 1.3114e-05, 1.4451e-05]]],
    
    
            [[[1.3328e-04, 1.2629e-04, 1.1106e-04, 8.7099e-05, 8.9130e-05],
              [9.4815e-05, 7.9949e-05, 7.2197e-05, 7.4259e-05, 8.6396e-05],
              [7.0104e-05, 7.1557e-05, 6.1332e-05, 5.9697e-05, 6.1651e-05],
              [8.3431e-05, 9.3442e-05, 8.8700e-05, 9.2470e-05, 8.2330e-05],
              [7.9784e-05, 7.9167e-05, 8.8693e-05, 8.3630e-05, 8.1308e-05]]]],
           device='cuda:0')}, 139973667684856: {'step': 740, 'exp_avg': tensor([-7.9799e-05, -2.1480e-05, -1.3800e-04, -1.5647e-05, -8.3889e-05,
            -3.4340e-05, -1.3140e-05, -8.7334e-05, -2.2785e-04,  3.3146e-06,
            -8.7209e-05, -9.6058e-05,  3.1596e-05, -6.5494e-05, -2.4494e-05,
            -1.9000e-05, -1.0589e-04, -3.9420e-05, -4.5874e-06, -1.2316e-04,
            -2.6975e-06,  1.6979e-05, -1.3662e-05, -3.4447e-05, -1.5522e-04,
            -5.6988e-05, -2.3558e-04, -1.0878e-04, -4.5769e-05,  1.2883e-05,
            -3.8347e-05, -9.0137e-05], device='cuda:0'), 'exp_avg_sq': tensor([2.9157e-04, 6.1778e-04, 3.4338e-03, 2.9030e-04, 3.1029e-04, 4.1914e-04,
            3.0401e-04, 4.3652e-03, 8.1713e-03, 9.3819e-05, 5.4442e-04, 1.0035e-03,
            1.7687e-03, 7.4427e-04, 2.2259e-04, 4.9283e-04, 1.6554e-03, 1.0319e-04,
            6.3148e-04, 1.7870e-03, 4.9489e-05, 3.8759e-04, 7.5012e-05, 6.8815e-03,
            3.4972e-03, 1.3243e-04, 2.8643e-03, 1.1973e-03, 1.2679e-03, 1.0819e-03,
            1.9215e-04, 7.2794e-04], device='cuda:0')}, 139973667684928: {'step': 740, 'exp_avg': tensor([[[[-1.3017e-40,  1.0379e-39,  1.1065e-39,  7.7513e-40,  1.0801e-39],
              [ 1.1671e-41,  2.8165e-40,  7.9976e-40,  6.0221e-40,  2.0683e-40],
              [ 4.7381e-40,  8.5584e-40,  3.3612e-40,  5.2462e-40,  1.9351e-41],
              [ 7.1868e-40,  1.2927e-40,  1.1003e-39,  8.1668e-40,  9.0573e-40],
              [ 1.6044e-39,  2.0372e-39,  2.3748e-39,  1.9679e-39,  1.9639e-39]],
    
             [[ 1.5175e-39,  2.7005e-39,  5.8051e-39,  7.1674e-39,  5.3070e-39],
              [ 4.9560e-39,  6.6073e-39,  7.1170e-39,  6.3059e-39,  5.2763e-39],
              [ 5.9423e-39,  6.1978e-39,  6.7896e-39,  6.5981e-39,  7.3944e-39],
              [ 7.4403e-39,  6.9715e-39,  5.5321e-39,  6.5758e-39,  7.2848e-39],
              [ 7.3981e-39,  6.4277e-39,  6.1097e-39,  5.6136e-39,  7.7875e-39]],
    
             [[ 1.6003e-39,  2.3878e-39,  2.8766e-39,  3.0332e-39,  2.4151e-39],
              [ 3.3217e-39,  2.9495e-39,  2.5174e-39,  2.5773e-39,  1.5448e-39],
              [ 1.6252e-39,  2.0962e-39,  2.6319e-39,  2.4615e-39,  2.0987e-39],
              [ 1.7941e-39,  1.7961e-39,  2.1625e-39,  1.9835e-39,  1.1507e-39],
              [ 2.0778e-39,  1.9583e-39,  2.1591e-39,  1.2864e-39,  2.0523e-39]],
    
             ...,
    
             [[ 3.7341e-38,  3.5959e-38,  3.7449e-38,  3.9891e-38,  4.0988e-38],
              [ 3.8980e-38,  3.9323e-38,  4.0111e-38,  4.1530e-38,  4.3540e-38],
              [ 4.1950e-38,  4.2357e-38,  4.2636e-38,  4.3036e-38,  4.4404e-38],
              [ 4.3690e-38,  4.4763e-38,  4.4172e-38,  4.4829e-38,  4.6257e-38],
              [ 4.6747e-38,  4.5810e-38,  4.5947e-38,  4.5739e-38,  4.6522e-38]],
    
             [[ 6.8719e-39,  5.9080e-39,  7.1665e-39,  8.7346e-39,  1.0001e-38],
              [ 5.4537e-39,  4.8429e-39,  4.6155e-39,  7.6902e-39,  8.5740e-39],
              [ 7.1835e-39,  7.1527e-39,  5.6052e-39,  7.3974e-39,  7.8800e-39],
              [ 7.9648e-39,  7.9595e-39,  7.8996e-39,  8.2719e-39,  9.2436e-39],
              [ 8.2832e-39,  7.9332e-39,  8.2839e-39,  8.1033e-39,  9.5073e-39]],
    
             [[ 8.9725e-39,  8.2496e-39,  6.2404e-39,  5.9072e-39,  5.6301e-39],
              [ 6.8025e-39,  5.8175e-39,  5.4543e-39,  5.5491e-39,  6.0928e-39],
              [ 6.2136e-39,  6.6331e-39,  5.3251e-39,  4.5145e-39,  4.9341e-39],
              [ 5.3506e-39,  5.0898e-39,  3.5753e-39,  4.0050e-39,  4.6279e-39],
              [ 5.1802e-39,  4.7753e-39,  4.8643e-39,  4.9856e-39,  4.9389e-39]]],
    
    
            [[[-2.3329e-40,  2.5938e-40,  3.3883e-40,  3.4141e-40,  6.6923e-41],
              [ 8.6949e-41,  6.2021e-42,  1.6432e-40, -6.4741e-41, -1.5544e-40],
              [ 2.8079e-40,  4.9772e-40,  5.7827e-41, -1.3919e-40, -4.1623e-41],
              [ 3.6633e-40,  1.4762e-40, -2.0857e-41, -6.8783e-41,  8.1654e-42],
              [ 3.2714e-40, -4.2106e-41,  1.8035e-41, -7.3078e-42,  4.3593e-41]],
    
             [[ 6.9746e-40,  7.9374e-40,  5.3751e-40,  7.4240e-40,  4.9219e-40],
              [ 1.0716e-39,  6.1477e-40,  1.8335e-40,  1.5195e-40,  4.0749e-40],
              [ 6.9012e-40,  3.0681e-41, -1.1119e-40, -6.7531e-41,  2.7236e-40],
              [ 4.0325e-40, -2.6978e-40, -1.5513e-40,  4.3195e-41,  2.6050e-40],
              [ 3.3414e-40, -3.0223e-41,  9.8982e-41, -1.7499e-41, -4.0778e-42]],
    
             [[ 4.8622e-41,  9.7805e-41,  4.7279e-40,  2.7356e-40, -9.5648e-41],
              [ 3.1614e-40,  2.4094e-40,  4.4906e-40,  3.7347e-40,  3.4288e-41],
              [ 3.2898e-41,  2.6512e-40, -4.9620e-42, -9.1325e-41, -1.0953e-40],
              [ 3.9901e-40,  3.2897e-40,  4.4494e-41,  1.3499e-40,  1.2278e-40],
              [ 3.1125e-40, -4.7189e-41,  2.4656e-40,  8.8422e-42,  2.0481e-40]],
    
             ...,
    
             [[ 3.4768e-39,  3.4594e-39,  3.1654e-39,  3.4807e-39,  3.5993e-39],
              [ 3.5080e-39,  3.1223e-39,  2.8912e-39,  2.6242e-39,  2.9552e-39],
              [ 3.4824e-39,  2.9936e-39,  2.7789e-39,  2.3916e-39,  2.8374e-39],
              [ 2.6511e-39,  2.3416e-39,  1.8059e-39,  1.6128e-39,  1.9506e-39],
              [ 2.9817e-39,  2.4707e-39,  1.9771e-39,  1.5299e-39,  1.8712e-39]],
    
             [[ 8.0991e-41,  7.2119e-40,  5.9778e-40,  1.1077e-39,  7.3862e-40],
              [ 3.3211e-40,  5.5296e-40,  5.5264e-40,  7.1912e-40,  3.3798e-40],
              [ 7.8255e-40,  9.9451e-40,  1.0591e-39,  8.3093e-40,  9.7714e-40],
              [ 2.8326e-40,  6.7864e-40,  4.9226e-40,  3.1823e-40,  5.5142e-40],
              [ 2.8716e-40,  7.3420e-40,  2.9063e-40,  9.6468e-41,  1.0081e-40]],
    
             [[ 5.0172e-40,  6.4223e-40,  3.9304e-40,  6.3966e-40,  8.7602e-40],
              [ 1.4063e-40,  2.3620e-40,  1.2269e-40,  5.0100e-40,  2.1171e-40],
              [ 5.3354e-40,  2.8703e-40,  6.8455e-40,  7.0419e-40,  3.0808e-40],
              [ 2.1889e-40,  4.6243e-40,  8.5378e-40,  5.4143e-40,  4.0176e-40],
              [ 1.6357e-40,  1.2711e-39,  7.2682e-40,  6.1730e-40,  1.6587e-40]]],
    
    
            [[[-1.6753e-40, -1.5888e-40, -2.0343e-40, -9.5491e-41, -4.1597e-40],
              [ 3.5977e-40,  9.3385e-41,  5.4454e-40,  5.2817e-40,  1.9897e-40],
              [-1.6513e-40,  1.9349e-40,  6.7795e-40,  1.2357e-40, -4.4943e-40],
              [ 1.1387e-40,  2.5351e-41,  2.6312e-40, -1.0949e-40, -1.9237e-40],
              [-1.6039e-40,  7.1776e-41, -2.7402e-40, -3.6148e-40, -2.2030e-40]],
    
             [[ 9.9835e-40,  9.9705e-40,  4.0143e-40,  1.1756e-40, -3.8315e-40],
              [ 4.8509e-40,  4.2281e-40, -7.8791e-41, -7.3134e-40, -8.5606e-40],
              [ 4.0687e-40,  5.0955e-40, -1.7012e-40, -5.1403e-40, -1.6152e-40],
              [ 3.2355e-40,  7.5480e-40, -3.2352e-40, -2.4311e-40, -2.2180e-41],
              [ 2.3988e-40,  7.5251e-40,  3.4833e-40,  5.3604e-40, -2.8851e-40]],
    
             [[ 5.9218e-40,  2.8136e-40,  4.7603e-40,  4.5203e-40,  1.7704e-40],
              [ 9.5352e-40,  9.0927e-40,  3.9501e-40,  2.7491e-40,  4.6798e-40],
              [ 8.1302e-40,  5.5141e-40,  9.1993e-40,  8.6871e-40,  5.8405e-41],
              [ 1.0006e-39,  7.5367e-40,  5.6078e-40,  7.5442e-40,  1.0580e-40],
              [ 1.5072e-40,  3.5002e-40,  2.4577e-40,  3.2641e-40,  2.0108e-40]],
    
             ...,
    
             [[ 4.7314e-39,  4.7701e-39,  4.3734e-39,  3.2179e-39,  2.5495e-39],
              [ 4.9810e-39,  4.6980e-39,  4.7051e-39,  3.3983e-39,  2.5485e-39],
              [ 4.8755e-39,  4.3491e-39,  3.8990e-39,  3.0324e-39,  2.5480e-39],
              [ 3.8422e-39,  4.4682e-39,  4.0238e-39,  3.2805e-39,  2.3676e-39],
              [ 4.1068e-39,  4.7485e-39,  4.4639e-39,  3.8562e-39,  3.4026e-39]],
    
             [[ 8.6484e-40,  5.6847e-40,  5.1112e-40,  8.2172e-40, -8.7071e-41],
              [ 1.1121e-39,  1.0052e-39,  1.5194e-39,  8.2572e-40,  6.9318e-40],
              [ 2.2191e-39,  1.8154e-39,  2.1530e-39,  1.3686e-39,  9.5270e-40],
              [ 1.1245e-39,  1.2466e-39,  1.5465e-39,  5.5486e-40,  1.9764e-40],
              [ 7.4045e-40,  6.9788e-40,  1.1704e-39,  1.7191e-40, -4.3492e-40]],
    
             [[ 1.1989e-39,  1.5462e-39,  7.6913e-40,  7.0874e-40,  1.3661e-39],
              [ 1.2017e-39,  1.4380e-39,  1.0032e-39,  1.5225e-39,  1.1823e-39],
              [ 1.3575e-39,  1.6226e-39,  1.4214e-39,  1.4500e-39,  1.7474e-39],
              [ 1.1581e-39,  7.5857e-40,  1.3588e-39,  1.3202e-39,  1.2137e-39],
              [ 1.3571e-39,  7.0390e-40,  1.2001e-39,  1.0686e-39,  8.9103e-40]]],
    
    
            ...,
    
    
            [[[ 5.8928e-39,  6.7046e-39,  4.6468e-39,  2.9079e-39,  3.3898e-39],
              [ 4.7966e-39,  3.9767e-39,  2.1433e-39,  1.5807e-39,  2.8248e-39],
              [ 3.2371e-39,  2.4685e-39,  1.9564e-39,  1.7366e-39,  3.9733e-39],
              [ 1.8927e-39,  1.6033e-39,  2.2447e-39,  3.7832e-39,  4.0819e-39],
              [ 2.1601e-39,  2.5901e-39,  4.4462e-39,  6.4014e-39,  4.8703e-39]],
    
             [[ 1.4018e-38,  1.3487e-38,  1.4963e-38,  1.6410e-38,  1.6531e-38],
              [ 1.2373e-38,  1.1413e-38,  1.6523e-38,  2.1169e-38,  2.0607e-38],
              [ 1.1434e-38,  1.1849e-38,  1.9105e-38,  2.5354e-38,  2.1172e-38],
              [ 1.1079e-38,  1.2456e-38,  1.8560e-38,  2.1249e-38,  1.6191e-38],
              [ 1.1676e-38,  1.3200e-38,  1.5705e-38,  1.5335e-38,  1.2711e-38]],
    
             [[ 6.1400e-39,  6.4469e-39,  4.4169e-39,  2.3402e-39,  2.8745e-39],
              [ 4.5604e-39,  3.0290e-39,  9.4813e-40,  7.7477e-40,  2.5678e-39],
              [ 4.5740e-39,  4.6391e-39,  4.3785e-39,  3.6666e-39,  5.0874e-39],
              [ 3.6641e-39,  3.3012e-39,  3.7462e-39,  2.9975e-39,  3.2154e-39],
              [ 2.9867e-39,  4.1722e-39,  5.2437e-39,  4.7437e-39,  3.7601e-39]],
    
             ...,
    
             [[ 1.1462e-37,  1.0998e-37,  1.0936e-37,  1.1027e-37,  1.1311e-37],
              [ 1.0989e-37,  1.0562e-37,  1.0748e-37,  1.1417e-37,  1.1685e-37],
              [ 1.0586e-37,  1.0262e-37,  1.0658e-37,  1.1623e-37,  1.2085e-37],
              [ 1.0405e-37,  1.0128e-37,  1.0644e-37,  1.1497e-37,  1.1875e-37],
              [ 1.0217e-37,  1.0113e-37,  1.0440e-37,  1.0738e-37,  1.1088e-37]],
    
             [[ 2.4395e-38,  2.3898e-38,  2.2218e-38,  2.1565e-38,  2.1384e-38],
              [ 2.4852e-38,  2.4337e-38,  2.2808e-38,  2.2189e-38,  2.0843e-38],
              [ 2.3206e-38,  2.0263e-38,  1.9253e-38,  1.9538e-38,  2.2931e-38],
              [ 2.1760e-38,  1.8603e-38,  2.0436e-38,  2.3817e-38,  2.7171e-38],
              [ 2.0462e-38,  1.8653e-38,  1.9697e-38,  2.3696e-38,  2.4650e-38]],
    
             [[ 1.0972e-38,  1.1054e-38,  1.0574e-38,  1.0210e-38,  9.2649e-39],
              [ 1.2660e-38,  1.4831e-38,  1.3371e-38,  1.1889e-38,  9.7202e-39],
              [ 1.3731e-38,  1.2221e-38,  9.3267e-39,  6.1001e-39,  6.9830e-39],
              [ 1.3054e-38,  1.2379e-38,  9.3985e-39,  6.1158e-39,  7.4434e-39],
              [ 1.3684e-38,  1.2835e-38,  9.8525e-39,  7.3010e-39,  9.7432e-39]]],
    
    
            [[[ 4.8290e-41,  6.7697e-42, -1.3691e-42,  1.4435e-40,  8.1675e-41],
              [ 6.6256e-41,  3.9466e-41, -5.8181e-41, -5.5822e-41, -1.3172e-41],
              [-1.7555e-41, -8.2468e-41,  2.4411e-42,  9.3458e-41,  6.2094e-41],
              [-9.6535e-42, -1.1701e-41,  9.8276e-41,  7.3431e-41,  6.6161e-41],
              [-1.9711e-40, -6.3780e-41,  1.1744e-40,  2.7290e-41,  2.2278e-41]],
    
             [[-3.1510e-41, -4.9243e-41,  9.7766e-41, -2.8799e-41, -1.0840e-41],
              [-9.6314e-41,  1.6350e-41,  9.7170e-41,  2.5278e-41,  8.6918e-41],
              [-2.1763e-40, -1.0448e-41,  1.2012e-40,  8.5273e-41,  1.7898e-40],
              [-2.9183e-40,  5.5812e-41,  2.8015e-40,  1.1811e-40,  2.1082e-40],
              [-3.6278e-41,  2.5986e-40,  2.7502e-40,  1.4821e-40,  2.4914e-40]],
    
             [[ 3.9166e-42,  6.7395e-41, -8.4744e-41, -4.4148e-41, -2.9493e-41],
              [ 1.0914e-40,  2.5633e-41, -5.1007e-43, -6.5903e-42, -6.0495e-41],
              [ 3.3967e-41, -1.6410e-40, -2.5057e-41, -3.3541e-41, -4.2608e-41],
              [ 8.3189e-41,  6.4992e-41,  6.6314e-41,  8.8091e-41,  2.1912e-41],
              [-6.3861e-41, -7.2604e-41,  6.4621e-41, -1.1785e-42, -1.1996e-40]],
    
             ...,
    
             [[ 4.1137e-40,  2.4206e-40,  2.3007e-40,  1.6895e-40,  2.2842e-40],
              [ 2.8129e-40,  8.5408e-41,  1.6451e-40,  1.5224e-40,  1.9287e-40],
              [ 2.6074e-40,  2.4235e-40,  2.5573e-40,  3.9370e-40,  4.5433e-40],
              [ 2.6148e-40,  2.9018e-40,  3.5624e-40,  4.5531e-40,  6.6990e-40],
              [ 2.5510e-40,  3.3147e-40,  4.2648e-40,  5.1864e-40,  5.9322e-40]],
    
             [[ 2.5369e-40,  4.7113e-41, -1.5077e-41,  3.3149e-41, -2.0829e-41],
              [ 2.4696e-40,  1.7236e-43,  4.1906e-41,  3.1777e-41,  3.8935e-41],
              [ 1.6955e-40,  2.0198e-41,  1.6242e-40,  1.1496e-40,  8.7264e-41],
              [ 1.0135e-40,  5.4534e-41,  1.3852e-40,  6.5338e-41,  8.4022e-42],
              [ 1.0016e-40, -7.4283e-42,  1.1114e-40,  1.4705e-40,  4.8484e-41]],
    
             [[ 2.0926e-40,  1.4028e-40, -1.0265e-41, -2.4481e-41,  3.6189e-41],
              [ 1.6027e-40,  6.8294e-41, -6.0819e-41, -2.0833e-41, -5.6856e-41],
              [ 1.0328e-40,  1.0364e-40,  7.4401e-41,  1.1567e-40,  1.1753e-40],
              [ 1.0959e-40,  2.5626e-41,  1.3686e-41,  5.3249e-43, -3.0349e-41],
              [ 1.1897e-40,  6.4100e-41,  3.4312e-41,  1.1119e-41, -5.0915e-41]]],
    
    
            [[[ 1.4662e-38,  1.5431e-38,  1.4983e-38,  1.2950e-38,  1.2152e-38],
              [ 1.2146e-38,  1.2342e-38,  1.0066e-38,  1.0105e-38,  1.1917e-38],
              [ 1.9027e-38,  1.6908e-38,  1.3193e-38,  1.4725e-38,  1.4102e-38],
              [ 2.0809e-38,  1.8831e-38,  1.8597e-38,  1.7129e-38,  1.5240e-38],
              [ 1.3053e-38,  1.3484e-38,  1.7967e-38,  1.6684e-38,  1.6895e-38]],
    
             [[ 5.3563e-38,  4.8828e-38,  4.6409e-38,  4.8274e-38,  5.0432e-38],
              [ 6.2486e-38,  5.8735e-38,  5.6750e-38,  5.8216e-38,  5.8660e-38],
              [ 5.9887e-38,  5.5443e-38,  5.6732e-38,  5.6578e-38,  5.6695e-38],
              [ 4.3201e-38,  4.3521e-38,  4.9232e-38,  5.4427e-38,  5.5830e-38],
              [ 4.5234e-38,  4.9747e-38,  4.9099e-38,  5.0786e-38,  5.2366e-38]],
    
             [[ 8.1463e-39,  1.0253e-38,  1.0630e-38,  7.8646e-39,  7.5326e-39],
              [ 9.7835e-39,  1.1051e-38,  1.0965e-38,  1.2432e-38,  1.3490e-38],
              [ 1.3959e-38,  1.6227e-38,  1.3839e-38,  1.2490e-38,  1.2103e-38],
              [ 1.0112e-38,  8.7160e-39,  9.0044e-39,  1.0260e-38,  1.0688e-38],
              [ 8.9246e-39,  1.0697e-38,  1.3313e-38,  1.2410e-38,  1.0787e-38]],
    
             ...,
    
             [[ 3.2893e-37,  3.2883e-37,  3.2253e-37,  3.2043e-37,  3.1716e-37],
              [ 3.3839e-37,  3.4049e-37,  3.3253e-37,  3.2935e-37,  3.2629e-37],
              [ 3.4392e-37,  3.4256e-37,  3.3859e-37,  3.4001e-37,  3.4012e-37],
              [ 3.3051e-37,  3.3164e-37,  3.2674e-37,  3.3269e-37,  3.3157e-37],
              [ 3.1571e-37,  3.1555e-37,  3.1232e-37,  3.1865e-37,  3.2236e-37]],
    
             [[ 6.2293e-38,  6.3567e-38,  6.1235e-38,  6.0557e-38,  6.3100e-38],
              [ 5.7881e-38,  5.6589e-38,  5.4495e-38,  5.4187e-38,  5.6164e-38],
              [ 7.1316e-38,  7.1149e-38,  6.5948e-38,  6.7596e-38,  6.4591e-38],
              [ 8.0009e-38,  8.1832e-38,  7.7737e-38,  7.8341e-38,  7.4009e-38],
              [ 6.7749e-38,  6.8170e-38,  6.6614e-38,  7.1198e-38,  7.1925e-38]],
    
             [[ 2.7812e-38,  3.0031e-38,  3.2314e-38,  3.2083e-38,  3.1903e-38],
              [ 2.4032e-38,  2.6284e-38,  3.1041e-38,  2.8011e-38,  2.6076e-38],
              [ 2.1931e-38,  2.3604e-38,  2.7069e-38,  2.5632e-38,  2.6954e-38],
              [ 2.8879e-38,  3.6137e-38,  3.2733e-38,  3.2093e-38,  2.9419e-38],
              [ 3.4123e-38,  3.5518e-38,  2.7894e-38,  2.9626e-38,  3.1088e-38]]]],
           device='cuda:0'), 'exp_avg_sq': tensor([[[[9.6963e-13, 9.4209e-13, 1.0998e-12, 1.5737e-12, 1.9341e-12],
              [1.2599e-13, 4.8425e-14, 9.1494e-13, 1.2636e-12, 7.8867e-14],
              [3.7967e-13, 1.0470e-12, 1.7849e-13, 1.5345e-13, 7.5012e-13],
              [3.4313e-13, 4.3369e-13, 1.2201e-12, 4.6413e-13, 1.5158e-12],
              [2.7779e-12, 4.1180e-12, 4.7641e-12, 4.5659e-12, 5.1273e-12]],
    
             [[3.8784e-12, 1.0162e-11, 3.2213e-11, 4.4678e-11, 3.1297e-11],
              [2.5041e-11, 4.7593e-11, 5.3143e-11, 4.5839e-11, 4.0147e-11],
              [3.4111e-11, 4.0213e-11, 4.9022e-11, 4.8103e-11, 5.5834e-11],
              [5.9103e-11, 5.1276e-11, 3.2208e-11, 4.6216e-11, 5.4732e-11],
              [5.2560e-11, 3.7577e-11, 4.2556e-11, 3.6758e-11, 6.4134e-11]],
    
             [[2.5089e-12, 4.1718e-12, 6.6172e-12, 8.7235e-12, 6.2381e-12],
              [1.2442e-11, 7.8617e-12, 7.0807e-12, 7.4668e-12, 4.0077e-12],
              [2.8494e-12, 4.4452e-12, 7.0254e-12, 5.5439e-12, 4.5104e-12],
              [3.8124e-12, 4.0555e-12, 5.3123e-12, 3.7686e-12, 1.4182e-12],
              [6.2496e-12, 4.4727e-12, 4.4535e-12, 1.7122e-12, 5.2205e-12]],
    
             ...,
    
             [[1.5588e-09, 1.4303e-09, 1.5054e-09, 1.6293e-09, 1.6769e-09],
              [1.5848e-09, 1.6813e-09, 1.7798e-09, 1.8858e-09, 2.0318e-09],
              [1.7894e-09, 1.8961e-09, 1.9514e-09, 2.0061e-09, 2.1152e-09],
              [1.9243e-09, 2.0553e-09, 2.0526e-09, 2.1495e-09, 2.3071e-09],
              [2.1886e-09, 2.1023e-09, 2.1796e-09, 2.2038e-09, 2.3072e-09]],
    
             [[5.5230e-11, 3.3931e-11, 5.1268e-11, 6.6555e-11, 8.6796e-11],
              [3.5271e-11, 2.3218e-11, 2.2042e-11, 5.1047e-11, 6.6054e-11],
              [5.7715e-11, 5.5008e-11, 3.7287e-11, 5.5371e-11, 7.2367e-11],
              [6.0549e-11, 6.0407e-11, 7.2313e-11, 7.6276e-11, 9.9925e-11],
              [6.3027e-11, 5.8417e-11, 7.2082e-11, 6.7595e-11, 9.6899e-11]],
    
             [[8.5389e-11, 7.1998e-11, 4.1968e-11, 4.2426e-11, 3.4840e-11],
              [4.4917e-11, 3.1792e-11, 2.9369e-11, 2.9479e-11, 3.2734e-11],
              [3.9314e-11, 4.3681e-11, 3.0966e-11, 2.0557e-11, 2.2221e-11],
              [3.1182e-11, 2.3796e-11, 1.1693e-11, 1.6520e-11, 2.3211e-11],
              [2.8829e-11, 2.2924e-11, 2.4536e-11, 2.4022e-11, 2.4558e-11]]],
    
    
            [[[1.1704e-13, 1.2616e-13, 1.9880e-13, 2.1754e-13, 8.6848e-15],
              [1.3849e-14, 3.3967e-15, 3.7100e-14, 1.1803e-14, 5.1718e-14],
              [1.5038e-13, 4.4336e-13, 3.1819e-15, 4.1695e-14, 4.8520e-15],
              [2.4180e-13, 2.7431e-14, 1.4288e-15, 1.0989e-14, 6.0754e-17],
              [1.7857e-13, 5.8865e-15, 4.7062e-16, 6.2685e-17, 3.6708e-15]],
    
             [[9.1175e-13, 1.2100e-12, 5.1843e-13, 1.0772e-12, 4.7072e-13],
              [2.1800e-12, 7.0397e-13, 5.7956e-14, 4.2306e-14, 3.1996e-13],
              [7.9828e-13, 9.1644e-16, 2.6853e-14, 1.0495e-14, 1.3626e-13],
              [2.3097e-13, 1.8390e-13, 5.1343e-14, 4.0028e-15, 1.3364e-13],
              [1.6226e-13, 4.0738e-15, 1.4898e-14, 1.0649e-15, 9.5522e-16]],
    
             [[3.1259e-15, 1.5482e-14, 3.5071e-13, 1.0742e-13, 2.9317e-14],
              [1.9309e-13, 9.1226e-14, 3.1710e-13, 2.5421e-13, 1.1834e-15],
              [1.1217e-15, 1.1284e-13, 1.2503e-14, 3.2802e-14, 3.9806e-14],
              [2.9323e-13, 1.8002e-13, 2.1756e-15, 2.2551e-14, 1.7593e-14],
              [1.8005e-13, 7.5385e-15, 1.0682e-13, 1.7527e-15, 7.0965e-14]],
    
             ...,
    
             [[2.1799e-11, 2.1705e-11, 1.8466e-11, 2.2494e-11, 2.4473e-11],
              [2.1614e-11, 1.7265e-11, 1.4912e-11, 1.2650e-11, 1.6290e-11],
              [2.0819e-11, 1.5556e-11, 1.3761e-11, 1.0613e-11, 1.5101e-11],
              [1.1528e-11, 9.1235e-12, 5.4068e-12, 4.6576e-12, 7.0111e-12],
              [1.4913e-11, 1.0310e-11, 6.5463e-12, 4.0493e-12, 6.3009e-12]],
    
             [[7.3351e-15, 9.6327e-13, 6.6488e-13, 2.2915e-12, 1.0366e-12],
              [1.7165e-13, 5.3710e-13, 5.0010e-13, 9.2852e-13, 1.8748e-13],
              [1.1191e-12, 1.8073e-12, 2.0140e-12, 1.3055e-12, 1.8325e-12],
              [1.1825e-13, 8.0672e-13, 4.0457e-13, 1.8666e-13, 5.7435e-13],
              [1.0427e-13, 9.3070e-13, 1.3069e-13, 1.5157e-14, 1.8210e-14]],
    
             [[4.3284e-13, 6.9419e-13, 2.5571e-13, 7.3392e-13, 1.3848e-12],
              [2.5328e-14, 7.7954e-14, 1.8130e-14, 3.7759e-13, 5.7355e-14],
              [5.1785e-13, 1.2760e-13, 7.7835e-13, 8.2961e-13, 1.4081e-13],
              [8.7029e-14, 3.6786e-13, 1.2240e-12, 4.8685e-13, 2.6152e-13],
              [4.4825e-14, 3.0247e-12, 8.7359e-13, 6.3567e-13, 3.1051e-14]]],
    
    
            [[[7.3257e-14, 1.0939e-13, 3.4876e-13, 3.6612e-13, 6.4915e-13],
              [1.6030e-13, 1.1793e-14, 3.1116e-13, 3.1776e-13, 3.7085e-14],
              [1.1540e-13, 5.3466e-14, 7.0535e-13, 1.5861e-14, 5.8601e-13],
              [1.1678e-14, 1.7389e-14, 1.3936e-13, 1.0816e-13, 1.1984e-13],
              [8.5858e-14, 1.4198e-13, 4.0809e-13, 4.6039e-13, 2.0209e-13]],
    
             [[1.2529e-12, 1.0282e-12, 4.7006e-13, 1.9367e-13, 1.2372e-12],
              [2.2072e-13, 3.3426e-13, 5.3577e-13, 1.6401e-12, 2.8387e-12],
              [3.0860e-13, 5.5585e-13, 8.6606e-13, 1.0012e-12, 5.5385e-13],
              [1.0791e-12, 1.2595e-12, 2.0289e-12, 5.5439e-13, 4.1854e-13],
              [2.0627e-12, 1.3212e-12, 3.8833e-13, 2.4313e-13, 9.1432e-13]],
    
             [[3.4536e-13, 1.2863e-13, 2.4726e-13, 2.5266e-13, 1.7222e-13],
              [1.1031e-12, 1.1514e-12, 1.8505e-13, 8.7402e-14, 2.7782e-13],
              [8.7251e-13, 4.2034e-13, 1.2522e-12, 9.1698e-13, 6.3254e-14],
              [1.5409e-12, 8.1201e-13, 5.6117e-13, 7.3541e-13, 1.1553e-13],
              [2.4027e-14, 1.3059e-13, 1.0978e-13, 1.0837e-13, 4.4286e-14]],
    
             ...,
    
             [[2.2293e-11, 2.2844e-11, 1.8598e-11, 1.0446e-11, 8.2854e-12],
              [2.5550e-11, 2.2626e-11, 2.2321e-11, 1.1626e-11, 7.8916e-12],
              [2.3929e-11, 1.8418e-11, 1.5245e-11, 1.0227e-11, 7.6322e-12],
              [1.4810e-11, 1.9661e-11, 1.6784e-11, 1.2522e-11, 8.6223e-12],
              [1.8205e-11, 2.2811e-11, 2.0479e-11, 1.5726e-11, 1.2637e-11]],
    
             [[6.7577e-13, 2.9830e-13, 3.9389e-13, 6.4219e-13, 1.2252e-12],
              [1.1293e-12, 9.9395e-13, 2.4264e-12, 7.2199e-13, 5.1415e-13],
              [6.6242e-12, 4.4713e-12, 6.5137e-12, 2.4586e-12, 9.8589e-13],
              [1.5118e-12, 1.9218e-12, 3.1341e-12, 3.6195e-13, 1.7869e-13],
              [4.8243e-13, 4.7094e-13, 1.6897e-12, 7.2529e-13, 1.5358e-12]],
    
             [[1.3851e-12, 2.9779e-12, 6.5261e-13, 4.6744e-13, 2.4505e-12],
              [1.4318e-12, 2.3556e-12, 1.0154e-12, 2.5221e-12, 1.4463e-12],
              [1.7681e-12, 2.9512e-12, 2.2367e-12, 2.5879e-12, 4.1402e-12],
              [1.4402e-12, 6.2709e-13, 2.4734e-12, 2.1768e-12, 1.8454e-12],
              [2.6212e-12, 7.6886e-13, 1.9427e-12, 1.4236e-12, 8.6891e-13]]],
    
    
            ...,
    
    
            [[[7.1824e-11, 5.2885e-11, 2.2155e-11, 1.2718e-11, 2.6851e-11],
              [4.0214e-11, 1.9509e-11, 3.5391e-12, 2.1330e-12, 1.4719e-11],
              [1.3013e-11, 6.8845e-12, 6.2695e-12, 8.1287e-12, 2.9594e-11],
              [8.6850e-12, 4.5686e-12, 8.7221e-12, 3.1287e-11, 4.3672e-11],
              [1.1234e-11, 1.4606e-11, 3.9802e-11, 7.2927e-11, 4.8329e-11]],
    
             [[2.8398e-10, 2.4235e-10, 3.6108e-10, 4.7963e-10, 4.7664e-10],
              [2.3456e-10, 2.2005e-10, 5.0373e-10, 8.0132e-10, 7.3831e-10],
              [2.4680e-10, 2.6643e-10, 7.5950e-10, 1.2530e-09, 9.5707e-10],
              [2.4979e-10, 3.7140e-10, 8.6112e-10, 1.0529e-09, 6.8260e-10],
              [2.8443e-10, 4.3564e-10, 6.9910e-10, 6.7658e-10, 4.4195e-10]],
    
             [[4.8688e-11, 5.2332e-11, 2.6393e-11, 1.4338e-11, 1.3931e-11],
              [3.0735e-11, 2.0696e-11, 6.1723e-12, 1.7129e-12, 8.5367e-12],
              [2.4597e-11, 2.3095e-11, 2.2386e-11, 1.8086e-11, 3.5081e-11],
              [2.0400e-11, 1.8231e-11, 1.4174e-11, 1.1678e-11, 1.6517e-11],
              [1.5914e-11, 1.7371e-11, 3.3711e-11, 2.7362e-11, 1.6050e-11]],
    
             ...,
    
             [[2.2775e-08, 2.1087e-08, 2.1154e-08, 2.1961e-08, 2.3081e-08],
              [2.1199e-08, 1.9841e-08, 2.0790e-08, 2.3316e-08, 2.5140e-08],
              [2.0744e-08, 1.9805e-08, 2.1936e-08, 2.5540e-08, 2.8336e-08],
              [2.0149e-08, 1.9781e-08, 2.3129e-08, 2.6624e-08, 2.9286e-08],
              [1.9653e-08, 2.0351e-08, 2.2609e-08, 2.4925e-08, 2.6205e-08]],
    
             [[1.0577e-09, 9.6753e-10, 8.7312e-10, 8.3176e-10, 8.4467e-10],
              [9.8227e-10, 8.1173e-10, 7.6761e-10, 7.0777e-10, 7.3328e-10],
              [8.9404e-10, 6.7233e-10, 6.6086e-10, 6.5911e-10, 8.8104e-10],
              [8.5167e-10, 6.6188e-10, 7.4690e-10, 1.0181e-09, 1.3080e-09],
              [7.7643e-10, 7.5824e-10, 8.5284e-10, 1.1714e-09, 1.3707e-09]],
    
             [[2.6694e-10, 2.9583e-10, 2.7364e-10, 2.2497e-10, 1.7188e-10],
              [3.0339e-10, 3.5935e-10, 2.8873e-10, 2.2948e-10, 1.8943e-10],
              [3.2496e-10, 3.0508e-10, 1.6448e-10, 7.2818e-11, 7.0403e-11],
              [2.9829e-10, 2.6260e-10, 1.3525e-10, 4.6758e-11, 8.5298e-11],
              [2.9328e-10, 2.6203e-10, 1.3717e-10, 7.4262e-11, 1.3378e-10]]],
    
    
            [[[2.1852e-15, 7.8895e-16, 3.1054e-16, 2.6841e-14, 9.3261e-15],
              [7.0665e-15, 1.5339e-15, 2.4998e-14, 1.1133e-14, 1.7569e-15],
              [8.4551e-16, 1.9275e-14, 2.6429e-15, 1.5414e-14, 6.8990e-15],
              [3.2876e-15, 9.1246e-15, 1.2425e-14, 8.7113e-15, 8.2352e-15],
              [1.0916e-13, 1.5328e-14, 2.1953e-14, 1.4872e-15, 1.0299e-15]],
    
             [[2.8843e-15, 1.8014e-14, 9.3301e-15, 1.4910e-15, 9.0125e-16],
              [3.9488e-14, 1.2418e-14, 1.2251e-14, 1.1580e-15, 1.2943e-14],
              [1.7398e-13, 2.5494e-14, 1.9118e-14, 1.2459e-14, 5.9379e-14],
              [2.7827e-13, 8.9944e-15, 1.3050e-13, 2.5481e-14, 7.7990e-14],
              [4.7372e-14, 9.8940e-14, 1.3713e-13, 3.9186e-14, 1.1292e-13]],
    
             [[6.3571e-16, 6.5563e-15, 1.9181e-14, 7.4210e-15, 4.5064e-15],
              [2.0624e-14, 3.0251e-15, 1.0939e-14, 8.2536e-15, 2.0152e-14],
              [1.1771e-15, 7.3340e-14, 7.0013e-15, 7.7290e-15, 7.4647e-15],
              [9.1660e-15, 4.2145e-15, 5.6020e-15, 8.9306e-15, 8.8563e-16],
              [2.8544e-14, 2.1233e-14, 4.0591e-15, 1.3550e-15, 3.6237e-14]],
    
             ...,
    
             [[1.7350e-13, 1.0531e-13, 1.0341e-13, 9.6035e-14, 7.7598e-14],
              [9.1595e-14, 9.1664e-14, 9.2542e-14, 5.8411e-14, 5.1972e-14],
              [9.5567e-14, 1.1449e-13, 1.0695e-13, 1.8068e-13, 2.4323e-13],
              [1.3769e-13, 1.6233e-13, 1.4892e-13, 2.4276e-13, 5.8565e-13],
              [1.3851e-13, 1.6275e-13, 2.0350e-13, 3.2069e-13, 4.3214e-13]],
    
             [[7.8655e-14, 1.5576e-14, 3.3458e-14, 2.2112e-14, 2.8981e-14],
              [9.6470e-14, 6.3407e-15, 5.1162e-15, 6.3151e-15, 2.2749e-15],
              [5.1296e-14, 7.4890e-15, 3.9465e-14, 2.0798e-14, 1.3308e-14],
              [1.2550e-14, 1.3285e-14, 2.5576e-14, 6.3994e-15, 5.1644e-16],
              [1.3819e-14, 1.2600e-14, 1.3775e-14, 3.4246e-14, 2.2973e-15]],
    
             [[4.9128e-14, 1.8790e-14, 2.2508e-14, 2.1553e-14, 1.2587e-14],
              [2.6243e-14, 5.9131e-15, 2.1413e-14, 2.4829e-14, 2.4749e-14],
              [1.0710e-14, 1.1438e-14, 7.1679e-15, 1.5462e-14, 1.8447e-14],
              [1.1034e-14, 5.7968e-15, 8.1964e-15, 9.6850e-15, 1.6875e-14],
              [2.1526e-14, 4.9269e-15, 3.3839e-15, 1.2572e-14, 1.9254e-14]]],
    
    
            [[[2.6939e-10, 3.2607e-10, 2.8017e-10, 2.1465e-10, 2.1575e-10],
              [2.5474e-10, 2.2677e-10, 1.1588e-10, 1.6174e-10, 1.9352e-10],
              [4.7102e-10, 3.4405e-10, 2.2089e-10, 2.4182e-10, 2.7555e-10],
              [5.6629e-10, 3.4923e-10, 3.4168e-10, 2.9691e-10, 2.7624e-10],
              [2.0203e-10, 2.2609e-10, 3.4716e-10, 3.5226e-10, 3.2383e-10]],
    
             [[4.7739e-09, 3.6435e-09, 3.1366e-09, 4.1729e-09, 4.3422e-09],
              [6.3307e-09, 5.2446e-09, 5.2671e-09, 5.9396e-09, 5.6790e-09],
              [5.1610e-09, 4.3900e-09, 5.3561e-09, 5.5749e-09, 5.1806e-09],
              [2.8613e-09, 2.9917e-09, 3.9029e-09, 4.9600e-09, 4.7280e-09],
              [2.7143e-09, 3.4561e-09, 3.8040e-09, 3.9483e-09, 3.8612e-09]],
    
             [[8.5138e-11, 1.8732e-10, 1.8681e-10, 1.2181e-10, 1.0369e-10],
              [1.3787e-10, 1.8948e-10, 1.4330e-10, 2.4941e-10, 2.6796e-10],
              [2.4325e-10, 3.1410e-10, 2.3400e-10, 1.9533e-10, 1.7773e-10],
              [1.7676e-10, 8.9222e-11, 8.8438e-11, 1.2437e-10, 1.3466e-10],
              [9.9250e-11, 1.4122e-10, 2.0661e-10, 1.7647e-10, 1.3095e-10]],
    
             ...,
    
             [[2.2295e-07, 2.1829e-07, 2.1147e-07, 2.0885e-07, 2.1054e-07],
              [2.3699e-07, 2.3657e-07, 2.2623e-07, 2.2280e-07, 2.2228e-07],
              [2.4398e-07, 2.4200e-07, 2.3657e-07, 2.4100e-07, 2.3936e-07],
              [2.2295e-07, 2.2362e-07, 2.2032e-07, 2.3054e-07, 2.3058e-07],
              [1.9998e-07, 2.0008e-07, 2.0178e-07, 2.1110e-07, 2.1696e-07]],
    
             [[6.7342e-09, 7.2030e-09, 6.7263e-09, 6.6267e-09, 7.5775e-09],
              [6.3493e-09, 6.5808e-09, 5.4570e-09, 5.5083e-09, 6.0044e-09],
              [1.0125e-08, 1.0345e-08, 8.5350e-09, 8.9506e-09, 8.3620e-09],
              [1.2194e-08, 1.2525e-08, 1.1181e-08, 1.1402e-08, 1.0101e-08],
              [8.9607e-09, 9.0172e-09, 8.3922e-09, 9.8974e-09, 9.9833e-09]],
    
             [[1.6344e-09, 1.7656e-09, 2.2381e-09, 1.9293e-09, 1.9315e-09],
              [1.1020e-09, 1.5212e-09, 2.1363e-09, 1.5273e-09, 1.2873e-09],
              [1.0606e-09, 1.4353e-09, 1.5058e-09, 1.3473e-09, 1.4050e-09],
              [1.7018e-09, 2.5116e-09, 1.9555e-09, 1.7406e-09, 1.5083e-09],
              [2.3813e-09, 2.5545e-09, 1.6195e-09, 1.5307e-09, 1.6532e-09]]]],
           device='cuda:0')}, 139973667685000: {'step': 740, 'exp_avg': tensor([ 6.6779e-38,  4.9241e-39,  8.6384e-39,  3.5585e-38,  3.3697e-37,
             5.5394e-39,  1.9965e-37,  2.0398e-37,  5.7881e-37,  8.0090e-38,
             1.0080e-37, -7.7623e-09,  3.8039e-38,  1.6730e-37,  5.1122e-38,
             3.9811e-38,  2.9041e-37, -5.1627e-05,  2.3256e-37,  2.8166e-37,
             6.2365e-37,  7.2592e-39,  2.2252e-05,  1.9309e-37,  2.8631e-37,
             6.2320e-34,  7.0470e-39,  2.5622e-33, -6.1890e-05,  2.0233e-39,
             1.6908e-39,  7.0521e-38,  1.3576e-37,  1.1119e-39,  2.5055e-37,
             2.0767e-19,  2.0145e-37,  6.1245e-37,  3.1481e-38,  1.8749e-38,
             2.9281e-37,  6.7801e-39,  1.5235e-37,  9.9503e-38,  2.4556e-39,
             4.2161e-37,  1.1276e-04,  2.0351e-37,  0.0000e+00,  4.7678e-26,
            -1.9265e-09,  7.0082e-39,  6.4629e-38,  7.2582e-39,  3.1866e-41,
            -5.0053e-05,  3.0879e-37,  3.3206e-38,  6.0382e-37,  6.7161e-38,
             1.9056e-37,  1.4860e-37,  2.8229e-40,  3.8269e-37], device='cuda:0'), 'exp_avg_sq': tensor([4.6731e-09, 4.3306e-11, 7.7940e-11, 1.6832e-09, 1.9131e-07, 5.6034e-11,
            6.8616e-08, 6.9939e-08, 4.4707e-07, 1.1854e-08, 1.9972e-08, 8.3584e-06,
            1.4743e-09, 1.1581e-08, 2.6607e-09, 2.4738e-09, 1.3026e-07, 1.3764e-03,
            9.1800e-08, 1.3008e-07, 6.1994e-07, 1.3456e-10, 5.7396e-04, 5.3560e-08,
            1.3725e-07, 6.7543e-07, 4.9025e-11, 2.2506e-06, 2.3100e-03, 3.5302e-12,
            3.8032e-12, 4.5848e-09, 2.2512e-08, 1.0933e-12, 1.1237e-07, 3.6139e-07,
            4.6314e-08, 5.8258e-07, 1.7107e-09, 5.7119e-10, 1.1679e-07, 1.0778e-10,
            3.0644e-08, 1.3918e-08, 5.3780e-12, 2.8609e-07, 1.8178e-03, 6.9104e-08,
            0.0000e+00, 8.2583e-07, 1.1283e-06, 5.5637e-11, 1.7736e-09, 1.2430e-10,
            2.2253e-15, 1.8167e-03, 1.7274e-07, 1.7296e-09, 6.1366e-07, 4.2971e-09,
            6.5642e-08, 4.2045e-08, 2.4269e-13, 3.8302e-07], device='cuda:0')}, 139973667685072: {'step': 740, 'exp_avg': tensor([[[[ 1.5488e-39,  1.5853e-39,  1.4922e-39,  1.1222e-39,  1.2268e-39],
              [ 2.3622e-39,  8.8358e-40,  9.3351e-40,  1.1929e-39,  9.2362e-40],
              [ 1.7900e-39,  9.5676e-40,  1.3480e-39,  1.6503e-39,  1.6630e-39],
              [ 1.4000e-39,  1.4321e-39,  1.0750e-39,  1.9393e-39,  1.9661e-39],
              [ 1.7555e-39,  2.2088e-39,  2.2126e-39,  1.4844e-39,  1.6217e-39]],
    
             [[ 7.3494e-41,  1.2215e-40, -7.3431e-41, -1.9948e-40, -4.0208e-40],
              [ 6.7622e-41, -3.0843e-42,  1.0727e-41, -1.2243e-40, -8.7315e-42],
              [-3.6049e-40, -4.1156e-42, -6.6010e-41,  4.6720e-40,  4.5912e-41],
              [-5.8171e-41, -1.8101e-40, -6.0353e-41,  3.0084e-40,  1.0323e-40],
              [-2.5981e-40, -1.4704e-41,  3.2850e-40,  2.0623e-40, -2.0487e-42]],
    
             [[-5.4243e-41, -1.4411e-40, -1.7763e-40, -6.1506e-40,  6.7066e-41],
              [ 1.1956e-40, -4.1511e-41,  2.5641e-40, -3.6172e-40,  9.2954e-41],
              [ 1.4554e-40, -4.8437e-40, -3.5214e-40, -7.8207e-40, -1.4792e-40],
              [ 4.3018e-40,  6.4748e-41, -5.1247e-41, -1.7554e-41,  1.1306e-40],
              [ 3.7730e-40,  5.6768e-41,  4.7571e-40, -4.1645e-40, -2.8302e-40]],
    
             ...,
    
             [[ 3.9458e-39,  4.0987e-39,  3.4654e-39,  3.7925e-39,  4.4793e-39],
              [ 4.0031e-39,  4.3228e-39,  4.3627e-39,  4.3682e-39,  4.6900e-39],
              [ 4.4848e-39,  5.1160e-39,  4.8241e-39,  5.3610e-39,  5.1816e-39],
              [ 4.2244e-39,  4.5538e-39,  4.2048e-39,  4.8149e-39,  4.5079e-39],
              [ 3.5021e-39,  3.7904e-39,  4.0418e-39,  4.7337e-39,  5.0242e-39]],
    
             [[-6.5805e-41, -7.7448e-41, -1.2416e-42, -4.5481e-41, -5.2375e-41],
              [-7.5950e-42, -1.4575e-41, -7.1845e-42, -9.3200e-42,  6.6898e-42],
              [-3.8588e-41, -4.5374e-42,  3.4619e-41, -5.5940e-41, -3.5719e-41],
              [ 1.9108e-41,  1.2924e-41,  5.6465e-41, -1.4181e-42, -4.8163e-41],
              [ 5.2205e-41,  4.5765e-41,  2.5078e-41,  9.5989e-43, -1.4966e-42]],
    
             [[ 1.1096e-38,  1.1260e-38,  1.1477e-38,  1.0810e-38,  1.0272e-38],
              [ 1.1798e-38,  1.1656e-38,  1.0393e-38,  1.0413e-38,  9.9146e-39],
              [ 1.0166e-38,  1.0898e-38,  1.1373e-38,  1.2131e-38,  1.1314e-38],
              [ 9.3270e-39,  9.4765e-39,  1.0388e-38,  1.1499e-38,  1.0456e-38],
              [ 1.0257e-38,  1.0811e-38,  9.0207e-39,  9.4215e-39,  7.7806e-39]]],
    
    
            [[[ 1.2759e-39,  1.2699e-39,  2.0132e-39,  2.1326e-39,  2.4046e-39],
              [ 1.7078e-39,  2.5331e-39,  3.0757e-39,  2.7940e-39,  2.3517e-39],
              [ 1.7041e-39,  1.8807e-39,  2.3600e-39,  2.2296e-39,  1.6534e-39],
              [ 1.4297e-39,  1.5428e-39,  1.8680e-39,  2.1882e-39,  1.6863e-39],
              [ 8.2974e-40,  1.9459e-39,  1.9705e-39,  1.7634e-39,  1.7962e-39]],
    
             [[ 3.7037e-40,  1.0445e-40,  9.6402e-41,  4.3565e-41, -1.5282e-40],
              [ 1.8098e-40,  9.9400e-41,  6.2785e-41, -4.6405e-41,  8.2719e-41],
              [ 3.0691e-40,  5.6865e-42, -3.0365e-41,  2.7096e-41,  2.6336e-41],
              [-1.0809e-40,  8.8395e-41,  1.1910e-40, -5.5371e-41, -1.5912e-40],
              [-5.1842e-41, -1.3622e-40,  8.2745e-41, -1.1128e-40,  2.2709e-41]],
    
             [[ 1.6473e-40,  6.7670e-40,  8.9289e-40,  4.6578e-40,  5.0700e-40],
              [ 4.4499e-40,  4.8919e-40,  1.9661e-40,  4.5130e-40,  4.5299e-40],
              [ 1.7236e-40,  2.8006e-40,  2.7175e-40,  8.1141e-41, -3.0342e-40],
              [-5.8025e-40,  5.4726e-41,  3.3773e-40,  6.7900e-41, -2.0406e-40],
              [-3.1008e-40, -1.3408e-41,  2.4943e-40, -4.1230e-41, -2.3287e-40]],
    
             ...,
    
             [[ 1.4932e-39,  2.0743e-39,  1.9141e-39,  1.9025e-39,  2.0155e-39],
              [ 1.6115e-39,  1.4825e-39,  1.7587e-39,  1.5023e-39,  1.1512e-39],
              [ 1.8894e-39,  1.2936e-39,  1.6612e-39,  1.2622e-39,  1.1024e-39],
              [ 1.5378e-39,  2.0367e-39,  1.5057e-39,  1.0658e-39,  6.2558e-40],
              [ 2.0306e-39,  2.1312e-39,  1.1745e-39,  1.4770e-39,  8.4062e-40]],
    
             [[-2.3990e-41,  2.6814e-41,  1.5158e-41,  3.9930e-41, -2.5179e-41],
              [ 2.9413e-42, -2.4677e-42, -9.2318e-42,  4.0097e-41,  3.5606e-41],
              [ 3.5173e-43, -2.7514e-41,  5.7593e-42, -2.0425e-41, -1.2139e-41],
              [-1.6055e-41, -1.9918e-41,  3.2657e-41,  5.1302e-42, -1.2201e-41],
              [ 1.7562e-41, -2.3850e-41,  6.0637e-41, -2.1647e-41, -1.2352e-41]],
    
             [[ 4.2638e-39,  4.6202e-39,  5.4218e-39,  6.1888e-39,  5.2983e-39],
              [ 6.1234e-39,  5.8085e-39,  6.1885e-39,  6.2798e-39,  5.8460e-39],
              [ 5.5190e-39,  5.9088e-39,  5.5130e-39,  5.0566e-39,  5.3190e-39],
              [ 5.6857e-39,  6.9636e-39,  6.3300e-39,  6.5912e-39,  5.4829e-39],
              [ 4.7928e-39,  4.5219e-39,  5.7797e-39,  5.3696e-39,  5.5042e-39]]],
    
    
            [[[ 1.6857e-39,  7.7346e-40,  3.1415e-40,  5.4145e-40,  3.4249e-40],
              [ 1.5432e-39,  1.8217e-39,  1.1767e-39,  1.2920e-39,  1.3853e-39],
              [ 2.5567e-39,  2.8540e-39,  1.5639e-39,  7.2663e-41, -5.5155e-42],
              [ 1.9561e-39,  1.7528e-39,  5.5733e-40,  3.2123e-40, -3.7921e-40],
              [ 9.7548e-40,  3.1353e-40, -5.8696e-40, -8.0532e-40, -8.0580e-40]],
    
             [[ 1.7382e-41,  1.1365e-42,  3.5743e-41,  2.2749e-40, -3.7600e-41],
              [-4.6887e-41,  2.4202e-40,  2.2753e-40,  1.6749e-40,  4.6667e-41],
              [ 3.6253e-41,  2.3239e-40,  3.1936e-40,  8.0764e-41,  5.6708e-41],
              [ 1.7121e-41,  3.2741e-40,  1.3862e-40, -1.9706e-41,  2.8867e-43],
              [ 2.2891e-40,  2.7733e-40, -7.6144e-41, -1.3595e-40,  3.4011e-41]],
    
             [[-2.2404e-41,  3.6434e-40,  3.3606e-40, -3.6326e-40, -3.1696e-40],
              [ 4.8337e-40,  8.0381e-40,  2.7906e-40, -5.8827e-40, -9.7096e-42],
              [ 5.1555e-40,  7.7724e-40, -8.1429e-41, -4.0765e-40,  1.7433e-40],
              [-1.6852e-40,  3.0969e-40, -4.4675e-40, -1.9784e-40, -1.2692e-40],
              [-4.3107e-41,  3.2046e-41, -3.2352e-40, -3.2971e-41, -1.6653e-40]],
    
             ...,
    
             [[ 5.4075e-40,  9.2538e-40,  3.4075e-40,  1.4436e-40, -4.4790e-41],
              [ 7.0408e-40,  2.8118e-40, -7.6676e-40, -1.2736e-40, -2.2847e-40],
              [-3.0457e-40, -5.5672e-40, -6.9778e-40,  5.0861e-40, -8.8635e-40],
              [-3.8480e-40, -6.0701e-40,  9.8342e-41, -5.7852e-40, -1.6126e-40],
              [-4.5229e-40, -6.1622e-40, -3.1680e-40,  1.1788e-40,  4.1040e-40]],
    
             [[-3.0129e-41,  1.3765e-41,  8.5462e-41,  2.7429e-41,  2.5672e-42],
              [ 2.7765e-41,  1.5054e-41,  2.4177e-41,  7.3134e-42,  2.8348e-42],
              [ 2.1083e-41,  2.8043e-41, -1.7811e-41,  1.6929e-41, -3.5666e-41],
              [ 2.7209e-41, -1.5386e-42, -1.5934e-41,  2.0051e-41, -4.7802e-41],
              [ 1.0007e-40,  3.0030e-42,  6.8849e-41, -2.8189e-41, -1.5545e-41]],
    
             [[ 1.9963e-39,  1.6579e-39,  2.0194e-39,  1.7006e-39,  1.5084e-39],
              [ 2.1190e-39,  6.3931e-40,  1.6439e-39,  2.4337e-39,  1.5587e-39],
              [ 4.8487e-39,  2.9859e-39,  2.0832e-39,  1.2739e-39, -3.5032e-40],
              [ 4.6174e-39,  3.7091e-39,  3.3531e-39,  1.1245e-39,  2.7417e-40],
              [ 1.3122e-39,  3.2225e-40,  2.0305e-40,  7.8409e-40,  1.0255e-39]]],
    
    
            ...,
    
    
            [[[ 4.1965e-39,  4.5218e-39,  4.9861e-39,  5.2196e-39,  6.3007e-39],
              [ 4.1827e-39,  4.8509e-39,  4.2227e-39,  3.8255e-39,  5.8258e-39],
              [ 3.5786e-39,  4.0929e-39,  4.6229e-39,  4.3695e-39,  5.3163e-39],
              [ 4.2278e-39,  4.4674e-39,  4.6458e-39,  5.0863e-39,  4.5141e-39],
              [ 5.4488e-39,  5.3328e-39,  4.4498e-39,  5.5080e-39,  5.7829e-39]],
    
             [[-2.7352e-40, -5.0763e-41, -2.7379e-40,  7.7586e-41, -7.2094e-41],
              [-2.2341e-40,  2.3060e-41,  7.7610e-41,  3.3970e-41, -2.8620e-41],
              [ 3.1001e-41, -1.2865e-41,  1.0423e-40, -1.1666e-40, -9.4910e-41],
              [ 6.1750e-41, -4.7406e-42,  3.4499e-41, -6.3954e-41, -2.1613e-40],
              [-2.4460e-41, -1.7157e-41,  6.3280e-41, -2.1249e-40, -1.1097e-40]],
    
             [[ 2.6454e-40,  4.6827e-41,  2.0538e-40,  2.0859e-40,  1.1320e-40],
              [ 2.1067e-40, -5.2316e-40,  5.6943e-41, -1.3995e-40, -1.2162e-40],
              [ 5.9509e-41, -2.2292e-40, -8.0896e-41, -2.0006e-40,  2.5707e-40],
              [ 2.5160e-40,  3.2694e-40,  3.3397e-40,  5.7275e-40,  5.2783e-40],
              [-1.7337e-40, -3.4768e-41,  3.6912e-40,  9.1405e-41,  4.8346e-40]],
    
             ...,
    
             [[ 2.0779e-38,  2.1672e-38,  2.1553e-38,  2.1588e-38,  2.1603e-38],
              [ 2.0439e-38,  2.0848e-38,  2.1947e-38,  2.2331e-38,  2.1709e-38],
              [ 2.3306e-38,  2.2731e-38,  2.2589e-38,  2.2134e-38,  2.1873e-38],
              [ 2.2927e-38,  2.1213e-38,  2.1592e-38,  2.3628e-38,  2.3057e-38],
              [ 2.0495e-38,  1.9773e-38,  2.2452e-38,  2.2575e-38,  2.1871e-38]],
    
             [[-1.3825e-41,  2.3417e-41,  1.5582e-42,  4.5158e-41, -4.2431e-42],
              [ 1.2650e-41, -4.7784e-43,  1.3228e-41,  5.2170e-41,  7.4769e-41],
              [ 3.2070e-41, -2.7536e-41, -4.9380e-41, -1.2575e-41, -3.1833e-41],
              [ 7.3036e-42,  6.0061e-41,  1.7554e-41,  5.8358e-41,  1.1778e-41],
              [-1.2945e-41, -2.9877e-41, -8.5059e-42,  1.1772e-41,  3.1421e-41]],
    
             [[ 5.6661e-38,  5.7322e-38,  5.8843e-38,  5.8481e-38,  5.7457e-38],
              [ 5.2185e-38,  5.3568e-38,  5.4551e-38,  5.4283e-38,  5.1983e-38],
              [ 4.9405e-38,  4.9340e-38,  5.0506e-38,  5.1718e-38,  5.2201e-38],
              [ 5.4532e-38,  5.4278e-38,  5.6155e-38,  5.7348e-38,  5.6858e-38],
              [ 5.4421e-38,  5.3000e-38,  5.4879e-38,  5.4785e-38,  5.4803e-38]]],
    
    
            [[[ 3.3062e-40,  2.9428e-40,  3.5635e-40,  3.5736e-40,  4.2052e-40],
              [ 3.6965e-40,  4.0869e-40,  4.4445e-40,  4.0166e-40,  3.7164e-40],
              [ 4.2464e-40,  4.0854e-40,  3.9272e-40,  4.0474e-40,  3.2621e-40],
              [ 4.0690e-40,  3.5287e-40,  3.6601e-40,  4.2465e-40,  3.8114e-40],
              [ 4.6498e-40,  4.2926e-40,  4.3155e-40,  4.8766e-40,  4.8612e-40]],
    
             [[ 1.5197e-41,  5.9596e-41,  4.1654e-41, -1.3070e-41,  1.2343e-41],
              [ 1.0301e-41,  4.3596e-41,  1.4237e-41, -9.3242e-42,  7.0808e-41],
              [ 2.7045e-42,  3.2287e-41,  6.5020e-41,  1.3043e-41,  4.0366e-41],
              [ 4.8485e-42,  1.2519e-41,  6.2610e-42,  9.4419e-42,  6.5833e-42],
              [ 2.5380e-41, -8.9767e-42, -7.8080e-42,  1.0515e-41,  1.2829e-41]],
    
             [[ 2.2784e-41,  9.7620e-41,  4.2103e-41,  1.8309e-41,  9.7068e-41],
              [ 5.8200e-41,  1.0896e-40,  6.1185e-41,  6.5914e-41,  8.5572e-41],
              [-1.7483e-41,  5.7393e-41,  5.7617e-41,  6.5191e-41,  6.9269e-41],
              [ 9.4502e-41,  1.2039e-41,  4.1801e-42,  1.2008e-40,  7.9144e-41],
              [ 1.4062e-41, -1.5964e-41,  1.0654e-41,  1.2575e-40,  2.7290e-41]],
    
             ...,
    
             [[ 1.8732e-40,  1.1700e-40,  2.2356e-40,  2.1031e-40,  1.5611e-40],
              [ 1.6440e-40,  1.1490e-40,  1.4734e-40,  1.7447e-40,  1.5481e-40],
              [ 9.5491e-41,  1.4172e-40,  2.4594e-40,  2.6888e-40,  1.1459e-40],
              [ 1.0397e-40,  1.3425e-40,  2.9419e-40,  2.1813e-40,  5.8172e-41],
              [ 6.0168e-41,  1.5918e-40,  2.1586e-40,  1.1767e-40,  6.8791e-41]],
    
             [[-2.5784e-43,  3.1459e-42, -4.3889e-42, -6.6141e-43,  1.7348e-42],
              [-1.4307e-42,  1.1589e-42,  8.2887e-42, -2.6695e-42, -1.1897e-42],
              [ 2.2000e-43,  3.5621e-42, -5.6052e-45,  1.5596e-42,  6.1545e-42],
              [ 1.5274e-42, -7.7352e-43, -2.3542e-42,  5.2647e-42,  1.0479e-41],
              [ 5.6052e-45, -2.9007e-43,  3.5453e-42,  5.6052e-45, -2.6064e-43]],
    
             [[ 6.1839e-40,  4.0958e-40,  6.0945e-40,  6.3875e-40,  6.6608e-40],
              [ 9.4263e-40,  7.5815e-40,  7.6213e-40,  7.1966e-40,  7.2410e-40],
              [ 9.1759e-40,  8.6824e-40,  8.8740e-40,  7.2440e-40,  7.9198e-40],
              [ 7.4330e-40,  7.7390e-40,  9.8186e-40,  9.0438e-40,  8.4712e-40],
              [ 9.7788e-40,  9.1493e-40,  9.4401e-40,  9.5806e-40,  1.0859e-39]]],
    
    
            [[[ 6.1462e-39,  6.6463e-39,  6.0462e-39,  5.6475e-39,  6.1805e-39],
              [ 5.2997e-39,  6.1752e-39,  5.3483e-39,  5.2420e-39,  4.8055e-39],
              [ 6.7032e-39,  6.4988e-39,  6.9809e-39,  5.6533e-39,  4.8564e-39],
              [ 5.8783e-39,  5.8129e-39,  7.8682e-39,  7.1953e-39,  5.9213e-39],
              [ 5.7453e-39,  6.1072e-39,  7.2593e-39,  5.7074e-39,  4.4547e-39]],
    
             [[ 1.6242e-40,  2.0583e-40,  6.7028e-41, -3.6594e-40, -6.2104e-41],
              [-1.2012e-40,  2.6683e-40, -1.4648e-41, -1.6253e-40, -2.2143e-40],
              [ 5.7726e-41,  4.7182e-40, -9.3696e-41, -7.7120e-41,  1.1459e-40],
              [ 8.7679e-42,  2.4519e-40,  4.5691e-41, -5.2240e-41,  3.2353e-40],
              [ 2.0207e-40, -2.2202e-40, -1.6088e-40, -1.3440e-41,  4.2185e-40]],
    
             [[ 2.7292e-40,  9.7013e-41,  8.9724e-41, -1.9254e-40, -2.0368e-40],
              [ 2.8184e-41,  5.8367e-40,  3.0232e-40, -4.6789e-42, -3.6123e-40],
              [ 1.0680e-40,  1.3325e-40, -1.4198e-40,  1.7249e-40, -9.6028e-41],
              [ 7.9290e-40,  4.8460e-40,  1.4198e-40,  7.5198e-41,  3.0546e-41],
              [ 5.6035e-40,  2.4830e-40,  5.9241e-40,  5.0694e-40, -1.4794e-40]],
    
             ...,
    
             [[ 1.9479e-38,  2.0427e-38,  1.9543e-38,  2.1499e-38,  2.2059e-38],
              [ 1.9060e-38,  1.8277e-38,  1.9814e-38,  2.1139e-38,  1.9262e-38],
              [ 1.9957e-38,  1.9971e-38,  2.1611e-38,  2.0766e-38,  2.0367e-38],
              [ 1.7955e-38,  2.0041e-38,  2.0952e-38,  1.9450e-38,  2.0607e-38],
              [ 1.9762e-38,  1.9943e-38,  2.0229e-38,  2.0396e-38,  2.1565e-38]],
    
             [[ 3.3300e-41,  4.8722e-41,  3.3027e-41,  3.8017e-42,  1.3730e-41],
              [ 2.4919e-41,  1.6810e-41,  4.8733e-41, -7.0156e-41, -7.2750e-41],
              [ 8.8240e-42,  2.6441e-41, -1.5215e-41,  1.8242e-41,  5.7497e-41],
              [-7.9342e-42,  6.5413e-42, -1.8133e-42,  2.1277e-41,  4.0907e-41],
              [ 2.4069e-41,  2.4321e-41,  4.6820e-41,  4.4757e-42,  5.1817e-41]],
    
             [[ 5.4718e-38,  5.6102e-38,  5.6300e-38,  5.7249e-38,  5.5963e-38],
              [ 4.9879e-38,  5.1438e-38,  4.9882e-38,  5.0759e-38,  4.8867e-38],
              [ 5.0715e-38,  5.0087e-38,  4.9158e-38,  4.8086e-38,  4.7533e-38],
              [ 5.4996e-38,  5.7501e-38,  5.3852e-38,  5.3298e-38,  5.2109e-38],
              [ 5.1651e-38,  5.1609e-38,  4.7523e-38,  4.7822e-38,  4.7504e-38]]]],
           device='cuda:0'), 'exp_avg_sq': tensor([[[[2.3940e-12, 2.3867e-12, 2.2462e-12, 2.7793e-12, 2.1485e-12],
              [5.1097e-12, 3.3311e-12, 3.8084e-12, 2.9601e-12, 4.8816e-12],
              [3.2275e-12, 2.7183e-12, 2.3910e-12, 2.5790e-12, 3.1912e-12],
              [3.7199e-12, 4.8276e-12, 6.3292e-12, 4.5853e-12, 4.7132e-12],
              [4.8378e-12, 5.4615e-12, 4.5626e-12, 2.2488e-12, 2.9491e-12]],
    
             [[1.0791e-14, 3.0380e-14, 1.0956e-14, 8.1426e-14, 3.2882e-13],
              [8.8534e-15, 2.3795e-17, 2.3421e-16, 3.0830e-14, 1.7742e-16],
              [2.6715e-13, 3.4466e-17, 8.9436e-15, 4.4411e-13, 4.2573e-15],
              [6.7692e-15, 6.6493e-14, 7.7141e-15, 1.8354e-13, 2.1792e-14],
              [1.3724e-13, 4.3441e-16, 2.1783e-13, 8.6534e-14, 8.5271e-18]],
    
             [[3.2073e-14, 8.1863e-14, 1.4481e-13, 1.0312e-12, 4.1065e-15],
              [2.1766e-14, 9.0302e-15, 1.0179e-13, 3.4497e-13, 7.9087e-15],
              [2.4116e-14, 5.3686e-13, 3.1930e-13, 1.3953e-12, 7.2128e-14],
              [2.8613e-13, 3.8824e-15, 3.4817e-14, 5.8675e-15, 1.4534e-14],
              [2.1471e-13, 3.8507e-15, 3.9676e-13, 4.3108e-13, 2.3841e-13]],
    
             ...,
    
             [[2.9323e-11, 3.1274e-11, 2.6874e-11, 2.0236e-11, 2.8558e-11],
              [2.6211e-11, 3.1765e-11, 2.7602e-11, 2.8859e-11, 3.7447e-11],
              [3.0265e-11, 3.6236e-11, 3.2191e-11, 3.5682e-11, 3.6027e-11],
              [2.9260e-11, 3.0671e-11, 2.6179e-11, 3.2972e-11, 3.7980e-11],
              [2.4684e-11, 2.6694e-11, 2.8720e-11, 3.3060e-11, 3.7624e-11]],
    
             [[9.6131e-15, 1.2264e-14, 2.8467e-18, 5.3216e-15, 5.6969e-15],
              [1.1728e-16, 5.4058e-16, 1.2903e-15, 1.6527e-16, 9.5537e-17],
              [2.9226e-15, 4.5169e-17, 1.7099e-15, 6.5321e-15, 2.4590e-15],
              [7.8681e-16, 2.5608e-16, 6.4264e-15, 9.5008e-18, 4.8084e-15],
              [5.4147e-15, 4.5237e-15, 7.1142e-16, 1.5778e-17, 2.8953e-18]],
    
             [[1.7911e-10, 1.8651e-10, 1.7771e-10, 1.6491e-10, 1.5452e-10],
              [2.1594e-10, 2.3693e-10, 2.3323e-10, 2.4246e-10, 2.0827e-10],
              [1.6151e-10, 1.7947e-10, 1.7831e-10, 2.0685e-10, 1.9670e-10],
              [1.3984e-10, 1.5481e-10, 1.6626e-10, 1.9803e-10, 1.7802e-10],
              [1.3522e-10, 1.2670e-10, 1.0728e-10, 1.1275e-10, 9.9597e-11]]],
    
    
            [[[2.5039e-12, 2.4660e-12, 6.8715e-12, 7.2604e-12, 8.8281e-12],
              [4.7899e-12, 1.0529e-11, 1.5627e-11, 1.1481e-11, 7.8621e-12],
              [4.8535e-12, 5.3682e-12, 8.6079e-12, 6.5047e-12, 3.4400e-12],
              [2.8155e-12, 3.4030e-12, 4.5419e-12, 5.8629e-12, 3.6302e-12],
              [6.8222e-13, 5.7002e-12, 5.7377e-12, 4.4306e-12, 5.2580e-12]],
    
             [[2.7686e-13, 2.2197e-14, 1.8934e-14, 3.8617e-15, 4.7519e-14],
              [6.6644e-14, 2.0104e-14, 7.9142e-15, 4.3820e-15, 1.2697e-14],
              [1.9165e-13, 6.5834e-17, 1.8760e-15, 1.4938e-15, 1.2152e-15],
              [2.3773e-14, 1.5899e-14, 2.8864e-14, 6.2259e-15, 5.1476e-14],
              [5.4687e-15, 3.7756e-14, 1.3931e-14, 2.5195e-14, 9.5816e-16]],
    
             [[4.9830e-14, 8.8862e-13, 1.5619e-12, 4.1822e-13, 4.7219e-13],
              [3.7704e-13, 4.1785e-13, 5.5753e-14, 3.7746e-13, 3.8069e-13],
              [5.5239e-14, 1.3995e-13, 1.2994e-13, 6.4198e-15, 2.2616e-13],
              [7.0848e-13, 3.8670e-15, 2.3132e-13, 9.0993e-15, 9.2331e-14],
              [2.0240e-13, 4.9081e-16, 1.2658e-13, 4.0445e-15, 1.1146e-13]],
    
             ...,
    
             [[2.0385e-12, 4.1908e-12, 3.3811e-12, 3.3473e-12, 3.7284e-12],
              [2.3737e-12, 2.0375e-12, 2.8193e-12, 2.2666e-12, 1.5697e-12],
              [3.2601e-12, 1.6633e-12, 2.5153e-12, 1.5962e-12, 1.1912e-12],
              [2.2207e-12, 4.0087e-12, 2.1228e-12, 1.1328e-12, 5.8467e-13],
              [4.1766e-12, 4.4573e-12, 1.3432e-12, 2.1018e-12, 8.6932e-13]],
    
             [[1.2455e-15, 1.2447e-15, 4.6753e-16, 3.2441e-15, 1.2899e-15],
              [8.0048e-18, 1.1909e-17, 1.7230e-16, 3.2711e-15, 2.5807e-15],
              [2.5190e-19, 1.5405e-15, 6.5422e-17, 8.5352e-16, 2.9971e-16],
              [5.2453e-16, 8.0731e-16, 2.1699e-15, 5.6572e-17, 3.0299e-16],
              [6.2762e-16, 1.1574e-15, 7.4808e-15, 9.5358e-16, 3.1048e-16]],
    
             [[1.9071e-11, 2.3715e-11, 3.3808e-11, 4.3001e-11, 2.8186e-11],
              [4.1971e-11, 3.9171e-11, 4.5114e-11, 4.5002e-11, 3.5704e-11],
              [3.1289e-11, 3.7337e-11, 3.1508e-11, 2.5810e-11, 2.9157e-11],
              [3.3440e-11, 5.2877e-11, 4.0126e-11, 4.6670e-11, 2.9297e-11],
              [2.1990e-11, 1.8898e-11, 3.2994e-11, 2.8208e-11, 2.8828e-11]]],
    
    
            [[[5.6803e-12, 1.2029e-12, 1.9496e-13, 5.8521e-13, 2.3596e-13],
              [4.8379e-12, 6.7417e-12, 2.7892e-12, 3.3929e-12, 3.9031e-12],
              [1.3258e-11, 1.6530e-11, 4.9661e-12, 1.0788e-14, 2.9017e-16],
              [7.7369e-12, 6.2062e-12, 6.3170e-13, 1.9486e-13, 3.2698e-13],
              [1.8944e-12, 1.9757e-13, 7.0513e-13, 1.3799e-12, 1.3431e-12]],
    
             [[6.1469e-16, 2.6262e-18, 2.5996e-15, 1.0530e-13, 2.8766e-15],
              [4.4733e-15, 1.1918e-13, 1.0533e-13, 5.7077e-14, 4.4313e-15],
              [2.6742e-15, 1.0988e-13, 2.0751e-13, 1.3271e-14, 6.5430e-15],
              [5.9648e-16, 2.1810e-13, 3.9094e-14, 7.9017e-16, 1.6980e-19],
              [1.0661e-13, 1.5649e-13, 1.1797e-14, 3.7605e-14, 2.3535e-15]],
    
             [[1.0213e-15, 2.7009e-13, 2.2978e-13, 2.6848e-13, 2.0441e-13],
              [4.7388e-13, 1.3146e-12, 1.5845e-13, 7.0410e-13, 1.8267e-16],
              [5.4079e-13, 1.2274e-12, 1.3492e-14, 3.3812e-13, 5.2898e-14],
              [5.7783e-14, 1.9513e-13, 4.0784e-13, 8.2941e-14, 3.6238e-14],
              [3.9761e-15, 1.7180e-15, 2.1340e-13, 2.2120e-15, 5.5916e-14]],
    
             ...,
    
             [[5.3625e-13, 1.6348e-12, 1.7584e-13, 2.5113e-14, 1.6908e-14],
              [8.7979e-13, 1.1506e-13, 1.3147e-12, 5.6128e-14, 1.5502e-13],
              [2.6610e-13, 7.5669e-13, 1.1307e-12, 3.6421e-13, 1.8175e-12],
              [3.7091e-13, 8.3526e-13, 1.1177e-14, 9.1433e-13, 8.8414e-14],
              [4.9650e-13, 8.8761e-13, 3.2852e-13, 1.3437e-14, 2.8522e-13]],
    
             [[1.8436e-15, 3.8558e-16, 1.4860e-14, 1.5308e-15, 1.3414e-17],
              [1.5762e-15, 4.6109e-16, 1.1893e-15, 1.0885e-16, 1.6368e-17],
              [9.0440e-16, 1.6001e-15, 6.4544e-16, 5.8308e-16, 2.8657e-15],
              [1.5062e-15, 4.8246e-18, 5.1669e-16, 8.1812e-16, 4.6492e-15],
              [2.0376e-14, 1.8355e-17, 9.6447e-15, 1.6169e-15, 4.9181e-16]],
    
             [[7.6682e-12, 5.2396e-12, 7.8678e-12, 5.5964e-12, 4.3464e-12],
              [8.7996e-12, 7.4627e-13, 5.2786e-12, 1.1672e-11, 4.7382e-12],
              [4.6381e-11, 1.7310e-11, 7.9463e-12, 2.8182e-12, 3.4364e-13],
              [4.2359e-11, 2.7210e-11, 2.1525e-11, 2.1218e-12, 7.2636e-14],
              [3.3150e-12, 1.4989e-13, 4.3507e-14, 9.5857e-13, 1.6291e-12]]],
    
    
            ...,
    
    
            [[[4.1278e-11, 5.6236e-11, 5.3459e-11, 4.9676e-11, 5.6529e-11],
              [4.7240e-11, 5.2125e-11, 4.3085e-11, 3.9933e-11, 5.6853e-11],
              [4.3491e-11, 5.2032e-11, 5.8248e-11, 5.2698e-11, 5.8959e-11],
              [5.1562e-11, 7.2080e-11, 8.3584e-11, 6.5039e-11, 5.8089e-11],
              [6.9371e-11, 5.8094e-11, 5.3145e-11, 4.7964e-11, 4.2775e-11]],
    
             [[1.6063e-13, 5.5184e-15, 1.5467e-13, 1.2039e-14, 1.1929e-14],
              [1.0233e-13, 8.1560e-16, 1.1223e-14, 2.1113e-15, 3.1387e-15],
              [1.9253e-15, 4.3251e-16, 2.0512e-14, 2.8486e-14, 2.2873e-14],
              [6.9054e-15, 7.1496e-17, 2.0689e-15, 8.3928e-15, 9.6825e-14],
              [1.5425e-15, 9.9548e-16, 7.8752e-15, 9.3788e-14, 2.6886e-14]],
    
             [[6.4343e-14, 4.2739e-14, 3.7869e-14, 3.7700e-14, 3.2726e-14],
              [3.8302e-14, 9.5568e-13, 1.0341e-13, 2.3763e-13, 1.6933e-13],
              [3.7482e-14, 4.1350e-13, 1.8574e-13, 2.8832e-13, 6.4036e-14],
              [5.8367e-14, 9.5943e-14, 1.2071e-13, 3.9100e-13, 3.0614e-13],
              [2.4250e-13, 1.2073e-13, 1.5293e-13, 2.7695e-14, 2.8834e-13]],
    
             ...,
    
             [[8.5793e-10, 8.9772e-10, 8.3888e-10, 8.7417e-10, 8.7117e-10],
              [7.6747e-10, 8.4780e-10, 9.2148e-10, 8.6383e-10, 8.0237e-10],
              [9.5203e-10, 9.5917e-10, 9.1040e-10, 8.3970e-10, 7.7876e-10],
              [9.1287e-10, 8.0349e-10, 8.0878e-10, 9.3598e-10, 8.5441e-10],
              [7.6618e-10, 7.2397e-10, 9.0387e-10, 9.3921e-10, 8.5114e-10]],
    
             [[2.2351e-15, 1.1193e-15, 1.2322e-16, 1.9460e-15, 1.6285e-16],
              [2.6864e-16, 1.5235e-18, 1.7716e-16, 5.5104e-15, 1.0065e-14],
              [1.3954e-15, 3.4176e-15, 7.4986e-15, 1.5953e-15, 2.3390e-15],
              [9.7461e-17, 7.3115e-15, 6.4653e-16, 6.6202e-15, 1.2744e-16],
              [6.6600e-16, 2.0956e-15, 2.4589e-15, 2.2562e-16, 1.0209e-15]],
    
             [[5.0037e-09, 5.0847e-09, 5.1438e-09, 5.3682e-09, 5.1922e-09],
              [4.6725e-09, 4.6050e-09, 4.8297e-09, 4.7564e-09, 4.5066e-09],
              [4.4154e-09, 4.3797e-09, 4.3547e-09, 4.4981e-09, 4.5296e-09],
              [4.5667e-09, 4.8237e-09, 5.1401e-09, 5.2742e-09, 4.9705e-09],
              [4.3294e-09, 4.3860e-09, 4.7741e-09, 4.6333e-09, 4.6560e-09]]],
    
    
            [[[2.2241e-13, 1.7620e-13, 2.5837e-13, 2.5983e-13, 3.5980e-13],
              [2.7802e-13, 3.3984e-13, 4.0191e-13, 3.2825e-13, 2.8102e-13],
              [3.6688e-13, 3.3960e-13, 3.1381e-13, 3.3331e-13, 2.1651e-13],
              [3.3688e-13, 2.5334e-13, 2.7257e-13, 3.6690e-13, 2.9557e-13],
              [4.3990e-13, 3.7490e-13, 3.7892e-13, 4.8386e-13, 4.8082e-13]],
    
             [[4.7000e-16, 7.2262e-15, 3.5303e-15, 3.4759e-16, 3.1009e-16],
              [2.1588e-16, 3.8674e-15, 4.1245e-16, 1.7691e-16, 1.0201e-14],
              [1.4887e-17, 2.1214e-15, 8.6018e-15, 3.4609e-16, 3.3152e-15],
              [4.7839e-17, 3.1897e-16, 7.9766e-17, 1.8140e-16, 8.8180e-17],
              [1.3105e-15, 1.6399e-16, 1.2403e-16, 2.2498e-16, 3.3487e-16]],
    
             [[1.0562e-15, 1.9389e-14, 3.6072e-15, 6.8210e-16, 1.9172e-14],
              [6.8922e-15, 2.4158e-14, 7.6174e-15, 8.8396e-15, 1.4899e-14],
              [6.2190e-16, 6.7022e-15, 6.7547e-15, 8.6471e-15, 9.7627e-15],
              [1.8171e-14, 2.9493e-16, 3.5564e-17, 2.9340e-14, 1.2744e-14],
              [4.0242e-16, 5.1861e-16, 2.3100e-16, 3.2176e-14, 1.5154e-15]],
    
             ...,
    
             [[7.1391e-14, 2.7852e-14, 1.0169e-13, 8.9993e-14, 4.9585e-14],
              [5.4991e-14, 2.6862e-14, 4.4168e-14, 6.1936e-14, 4.8766e-14],
              [1.8554e-14, 4.0865e-14, 1.2307e-13, 1.4710e-13, 2.6718e-14],
              [2.1994e-14, 3.6672e-14, 1.7609e-13, 9.6808e-14, 6.8856e-15],
              [6.7073e-15, 5.2501e-14, 9.4802e-14, 2.8171e-14, 1.2462e-14]],
    
             [[1.3524e-19, 2.0124e-17, 3.9197e-17, 8.8941e-19, 6.1391e-18],
              [4.1725e-18, 2.7373e-18, 1.3979e-16, 1.4513e-17, 2.8900e-18],
              [9.8537e-20, 2.5864e-17, 9.3354e-26, 4.9616e-18, 7.7065e-17],
              [4.7526e-18, 1.2167e-18, 1.1275e-17, 5.6432e-17, 2.2345e-16],
              [5.8821e-25, 1.7143e-19, 2.5576e-17, 1.4425e-24, 1.4008e-19]],
    
             [[7.7806e-13, 3.4133e-13, 7.5574e-13, 8.3016e-13, 9.1052e-13],
              [1.7739e-12, 1.1629e-12, 1.1818e-12, 1.0538e-12, 1.0485e-12],
              [1.7090e-12, 1.5338e-12, 1.5570e-12, 1.0677e-12, 1.2906e-12],
              [1.1241e-12, 1.2186e-12, 1.9615e-12, 1.6641e-12, 1.4601e-12],
              [1.8098e-12, 1.7032e-12, 1.8132e-12, 1.7836e-12, 2.4213e-12]]],
    
    
            [[[5.4276e-11, 6.9277e-11, 6.6145e-11, 6.8826e-11, 7.3286e-11],
              [4.6012e-11, 6.6761e-11, 5.9034e-11, 4.8019e-11, 5.1751e-11],
              [5.4065e-11, 5.3973e-11, 7.2010e-11, 6.3323e-11, 5.9848e-11],
              [4.6345e-11, 4.3871e-11, 7.7285e-11, 6.8272e-11, 7.2367e-11],
              [4.1573e-11, 4.5054e-11, 6.7421e-11, 4.6703e-11, 4.7023e-11]],
    
             [[5.1698e-14, 8.5820e-14, 9.1987e-15, 2.8112e-13, 8.0085e-15],
              [2.8847e-14, 1.4006e-13, 4.9119e-16, 5.8972e-14, 1.0590e-13],
              [7.7504e-15, 4.4708e-13, 1.8435e-14, 1.3360e-14, 2.5148e-14],
              [7.1980e-17, 1.2050e-13, 3.5412e-15, 6.2969e-15, 2.1126e-13],
              [7.9572e-14, 1.0135e-13, 5.6317e-14, 8.5660e-16, 3.4932e-13]],
    
             [[6.4311e-14, 2.5527e-14, 2.9504e-14, 2.9326e-13, 3.2846e-13],
              [8.2307e-14, 3.6008e-13, 7.2738e-14, 1.3013e-13, 5.5822e-13],
              [1.2130e-13, 3.8113e-14, 2.7221e-13, 8.8469e-14, 1.8564e-13],
              [9.4059e-13, 3.4078e-13, 3.8860e-14, 4.7439e-14, 7.7928e-14],
              [4.3039e-13, 5.3112e-14, 3.8034e-13, 2.6495e-13, 2.1263e-13]],
    
             ...,
    
             [[8.3947e-10, 9.2201e-10, 9.2152e-10, 9.9529e-10, 9.9590e-10],
              [7.3936e-10, 7.6634e-10, 9.0582e-10, 9.6185e-10, 9.4392e-10],
              [8.1918e-10, 8.9889e-10, 1.0991e-09, 9.7175e-10, 9.5708e-10],
              [7.4718e-10, 9.5340e-10, 9.9028e-10, 8.4246e-10, 9.4127e-10],
              [9.1030e-10, 9.7240e-10, 8.4294e-10, 8.7836e-10, 9.8215e-10]],
    
             [[1.0286e-15, 4.2495e-15, 1.7818e-15, 1.3223e-17, 1.8703e-16],
              [6.0194e-16, 2.9648e-16, 4.8259e-15, 1.0393e-14, 1.1828e-14],
              [7.1057e-17, 7.9030e-16, 4.7282e-16, 9.9993e-16, 4.8399e-15],
              [7.9695e-16, 4.3630e-17, 7.9518e-16, 5.5335e-16, 3.4403e-15],
              [6.0912e-16, 1.1756e-15, 2.4905e-15, 2.7025e-17, 4.8089e-15]],
    
             [[4.9660e-09, 5.1811e-09, 5.3684e-09, 6.0208e-09, 5.9292e-09],
              [4.8418e-09, 4.7045e-09, 4.5628e-09, 4.8082e-09, 4.8756e-09],
              [4.7721e-09, 5.0959e-09, 4.8835e-09, 4.5806e-09, 4.4987e-09],
              [4.9878e-09, 5.3622e-09, 5.1266e-09, 5.2884e-09, 5.4885e-09],
              [4.7050e-09, 4.6293e-09, 4.1582e-09, 4.3935e-09, 4.5656e-09]]]],
           device='cuda:0')}, 139973667685144: {'step': 740, 'exp_avg': tensor([-2.3578e-07,  1.7366e-11,  1.5549e-22,  5.2076e-19, -1.8281e-05,
            -7.7255e-06, -3.5721e-05, -2.5476e-05, -1.7329e-05, -2.4635e-06,
            -3.1397e-05, -6.3135e-05, -7.5900e-06, -9.5382e-06, -3.2276e-05,
            -2.1523e-05, -4.3575e-06, -8.8478e-06, -2.7037e-05, -3.7758e-05,
            -8.6851e-06, -5.4438e-06,  1.0101e-25,  9.3567e-09,  1.2188e-05,
             8.2908e-06, -2.5006e-05, -2.7254e-05, -2.2864e-05, -6.1570e-05,
            -5.6700e-16, -1.7000e-05, -2.1298e-06, -1.7412e-05, -2.0757e-09,
            -2.5051e-05,  7.7556e-07, -3.0588e-05, -1.6622e-11, -1.2612e-06,
            -6.8120e-06, -6.6586e-10, -1.4387e-05, -2.0960e-06, -2.8340e-05,
            -1.9506e-05, -1.3806e-05, -2.6239e-06,  3.0618e-07,  1.7345e-05,
             2.2577e-06, -3.5866e-05, -7.9663e-06, -2.6547e-06, -3.6667e-06,
            -1.0664e-05, -1.8409e-07, -6.4878e-06, -2.7534e-05, -1.5584e-05,
             1.1785e-06,  1.8901e-06,  6.3523e-29,  4.9625e-25, -9.5772e-06,
            -2.1365e-05, -1.7151e-05, -3.1553e-05, -6.1863e-06,  3.0573e-06,
            -5.4574e-05, -3.2391e-05,  1.2749e-06, -9.5886e-06, -1.4300e-06,
             1.3023e-06,  1.3670e-29, -1.7720e-05,  3.6608e-24, -3.4560e-06,
             2.6718e-14, -2.0614e-08, -1.2198e-05, -5.9687e-05, -1.5313e-05,
            -1.2989e-05, -1.3389e-05, -9.2824e-06, -5.7392e-05, -8.5485e-06,
             1.9321e-05, -3.0838e-05, -1.3434e-05, -1.4496e-05, -1.6302e-05,
            -1.0413e-04,  1.2052e-05, -2.6687e-05, -2.2160e-06, -2.6546e-05,
            -1.3059e-05,  3.0810e-34, -1.5668e-05, -1.3952e-06, -3.8475e-05,
            -8.8029e-06,  8.2060e-06, -1.1511e-05, -3.0121e-06, -9.0662e-10,
            -3.4194e-05, -5.4217e-16, -4.5742e-06,  4.4291e-06, -4.7962e-05,
            -4.2949e-05, -1.5130e-05, -2.1306e-05, -3.5656e-05, -8.4310e-06,
            -1.4576e-05, -4.0412e-05, -5.1420e-29,  1.2060e-29, -1.3509e-05,
            -2.4034e-05, -3.7227e-06, -5.8067e-06], device='cuda:0'), 'exp_avg_sq': tensor([1.3866e-05, 1.1242e-09, 1.5566e-09, 7.0718e-12, 3.2980e-05, 3.4578e-04,
            5.2116e-05, 9.4578e-05, 1.4290e-05, 1.0740e-05, 2.6562e-04, 9.3015e-05,
            1.3564e-05, 4.6311e-05, 3.1203e-05, 2.0981e-05, 4.8307e-05, 4.3598e-06,
            8.0037e-05, 3.9891e-05, 7.7653e-07, 4.8897e-04, 1.0031e-09, 4.6854e-09,
            5.9576e-05, 9.6634e-06, 4.6110e-05, 8.1561e-05, 3.0534e-04, 6.6793e-05,
            2.7897e-08, 5.2202e-05, 7.3921e-05, 1.5870e-04, 1.3495e-07, 9.4329e-05,
            2.8361e-05, 7.3709e-05, 8.4973e-10, 8.9003e-07, 1.1627e-04, 1.8006e-08,
            3.6863e-05, 6.8438e-05, 4.4290e-05, 5.8228e-05, 2.7083e-05, 2.3722e-06,
            1.1249e-05, 1.7240e-05, 1.3531e-04, 1.0815e-04, 4.4811e-05, 5.7509e-07,
            3.5909e-06, 2.9426e-05, 1.0804e-07, 1.7187e-05, 6.2409e-06, 5.3692e-05,
            5.9836e-05, 2.3810e-04, 6.9539e-09, 2.7687e-09, 3.2458e-05, 4.4086e-05,
            1.8845e-04, 1.8748e-04, 5.3662e-06, 8.0701e-06, 6.2403e-05, 1.6103e-04,
            8.9488e-05, 1.7797e-04, 2.5125e-06, 1.0689e-04, 2.2584e-09, 6.2075e-07,
            1.8678e-08, 3.7741e-05, 4.1751e-08, 3.9592e-08, 5.0769e-04, 7.6856e-05,
            2.2369e-05, 5.6442e-04, 1.7151e-05, 4.7165e-05, 1.3747e-04, 1.3953e-04,
            3.2468e-05, 2.2569e-04, 1.6059e-04, 2.4150e-05, 3.6875e-05, 2.8025e-04,
            1.2252e-04, 1.1426e-04, 9.6744e-05, 4.7579e-04, 6.0130e-05, 3.3897e-11,
            4.1518e-04, 5.3536e-05, 9.8099e-05, 4.2151e-05, 5.2717e-05, 6.8428e-05,
            1.7289e-04, 2.4628e-07, 1.8815e-04, 4.4741e-06, 9.4062e-08, 1.4293e-04,
            2.2801e-04, 5.7207e-05, 2.4369e-05, 3.1279e-05, 8.7716e-05, 2.2729e-04,
            7.1701e-05, 2.0141e-04, 2.4391e-07, 1.1172e-10, 1.5487e-04, 3.5605e-05,
            6.2320e-05, 5.5772e-05], device='cuda:0')}, 139973667685288: {'step': 740, 'exp_avg': tensor([[-1.2614e-40, -1.3596e-40, -1.7828e-40,  ...,  9.8055e-32,
              8.7192e-32, -2.4532e-40],
            [-6.9298e-10, -1.0759e-09, -4.1266e-08,  ...,  7.1499e-08,
             -4.3367e-09, -4.0954e-08],
            [ 1.8424e-40,  2.2654e-40,  3.1042e-40,  ...,  7.1240e-40,
              6.9902e-40,  8.9693e-40],
            ...,
            [ 3.3841e-40,  2.8444e-40,  4.5396e-40,  ...,  9.9471e-40,
              7.8638e-40,  1.0274e-39],
            [ 1.0853e-40,  1.4774e-40,  2.4504e-40,  ...,  8.7235e-40,
              7.9137e-40,  8.1443e-40],
            [ 6.5421e-41,  5.4733e-41,  1.4447e-40,  ...,  1.6458e-40,
              1.1660e-40,  2.7427e-40]], device='cuda:0'), 'exp_avg_sq': tensor([[3.3649e-14, 3.9272e-14, 6.5697e-14,  ..., 3.8333e-13, 3.6747e-13,
             4.8539e-13],
            [7.8873e-11, 1.2541e-10, 1.4939e-10,  ..., 7.3927e-09, 4.7030e-09,
             3.5334e-09],
            [6.9066e-14, 1.0442e-13, 1.9605e-13,  ..., 1.0326e-12, 9.9417e-13,
             1.6369e-12],
            ...,
            [2.3301e-13, 1.6461e-13, 4.1929e-13,  ..., 2.0132e-12, 1.2582e-12,
             2.1477e-12],
            [2.3964e-14, 4.4410e-14, 1.2217e-13,  ..., 1.5483e-12, 1.2742e-12,
             1.3496e-12],
            [8.7083e-15, 6.0953e-15, 4.2467e-14,  ..., 5.5112e-14, 2.7663e-14,
             1.5306e-13]], device='cuda:0')}, 139973667685576: {'step': 740, 'exp_avg': tensor([ 7.0964e-30, -2.2308e-06,  8.4228e-39,  6.0729e-38, -5.0292e-09,
             4.1776e-38,  5.5048e-40,  3.2280e-38,  3.5510e-36,  3.9122e-07,
             2.5716e-38,  3.6103e-29,  1.5041e-07,  1.4867e-36,  1.3321e-39,
             9.1540e-40, -1.9110e-09,  6.8519e-39,  1.2180e-38,  5.0591e-38,
             6.3986e-39,  7.8831e-22, -7.1080e-08,  3.2570e-39,  4.0388e-29,
            -9.9433e-10,  1.0207e-38,  4.0584e-23,  9.4005e-39,  2.0727e-38,
             5.4088e-38, -1.3699e-06,  1.5631e-37, -3.9072e-07,  3.6433e-30,
             2.1935e-38,  3.9599e-17, -1.7636e-09,  1.4819e-39, -3.1185e-10,
            -5.0586e-07,  4.1662e-20,  2.7550e-39,  1.6704e-19, -4.7524e-11,
             6.0336e-38, -1.8663e-07,  1.2247e-18,  4.2761e-39,  4.9226e-18,
             6.9207e-07,  8.4366e-39,  6.9918e-30,  3.8416e-38,  3.5073e-20,
             5.7660e-07,  1.7105e-38, -1.0731e-18, -4.8479e-06, -4.1765e-07,
             2.5945e-39,  1.4307e-38,  3.8044e-07,  4.1306e-39,  9.0032e-40,
             7.4784e-39, -5.9642e-07,  1.9896e-21,  2.1423e-26,  1.4755e-08,
            -3.1983e-07,  4.9670e-39, -2.4644e-09,  8.3028e-31,  4.2345e-38,
             1.6923e-20, -1.8626e-07, -1.2220e-06,  2.4535e-30,  9.3552e-38,
            -3.5983e-07,  1.7707e-34,  1.4847e-39, -1.0168e-06,  4.5709e-39,
             7.2945e-39,  1.1373e-37, -4.3633e-06, -3.5771e-06,  4.7881e-38,
             4.6576e-39,  6.8019e-39,  3.7654e-29,  1.3121e-37,  4.0534e-40,
             1.9472e-39,  5.3806e-38,  1.6883e-39,  1.0503e-18, -3.0477e-06,
            -3.7367e-09, -3.1225e-13, -1.2141e-08,  4.9240e-38, -6.1840e-07,
             5.9718e-39, -2.5003e-07,  2.5333e-39, -1.3697e-06, -2.6122e-06,
             7.0003e-23, -1.2918e-06,  5.5469e-27,  4.7716e-38,  5.6870e-40,
             1.0731e-20,  1.3801e-36,  2.7097e-20,  4.2417e-39,  5.4293e-39,
            -4.6862e-08, -5.3254e-07, -1.9873e-13,  1.3410e-37,  1.8046e-18,
            -2.4470e-06,  3.6216e-38,  2.5057e-38, -2.1758e-08,  4.5452e-30,
             3.1014e-38,  2.2910e-09, -8.9793e-08, -1.2846e-06, -4.2214e-06,
             3.8147e-37,  2.1093e-35, -5.2801e-06,  2.1446e-23,  2.3049e-36,
             2.9844e-21,  1.0782e-39,  3.9380e-30,  8.5452e-39, -4.9201e-06,
            -2.5818e-06, -2.8760e-08, -6.4455e-12,  1.4885e-38,  9.3452e-29,
             3.4351e-07,  4.8771e-38,  7.0203e-39, -1.0766e-07,  4.1878e-20,
             5.4508e-40,  4.8375e-38,  5.1286e-38,  1.0547e-38,  2.7759e-39,
             2.0815e-37,  3.9875e-39,  2.5336e-37, -4.6212e-07,  2.3362e-37,
            -5.3456e-09, -8.4814e-16,  7.0766e-30, -1.8451e-07,  3.5574e-38,
            -7.1613e-21,  6.4744e-25,  7.9959e-39,  2.6233e-38,  1.4521e-38,
            -2.3324e-08,  6.3991e-39, -9.2862e-08, -8.0518e-19,  2.0196e-18,
             4.1609e-38,  4.2001e-29,  1.3058e-38,  1.4281e-34,  4.7702e-37,
            -2.9887e-08,  4.6638e-39,  7.3121e-38, -3.2202e-06,  3.5916e-38,
             5.0241e-39,  3.5906e-07,  4.6301e-39,  3.2292e-38, -2.1062e-11,
             3.8179e-38,  9.9652e-37,  2.4010e-37,  1.9553e-38, -3.4618e-06,
             2.0281e-07,  4.1976e-39,  5.1975e-07,  1.4833e-38,  3.0221e-38,
             2.3399e-29,  1.4405e-38, -9.8022e-07, -3.2157e-07, -1.0858e-14,
             5.2295e-39,  6.2241e-40,  1.2339e-38,  5.7574e-07, -5.2579e-08,
            -2.8215e-06,  3.2318e-40, -2.2407e-06,  1.3349e-17,  3.0006e-33,
             1.2045e-38,  6.9622e-22, -1.2151e-06,  1.4890e-38, -3.2882e-06,
             8.5544e-40,  1.0373e-30,  1.0755e-39, -4.1238e-06,  6.0409e-28,
            -5.9565e-09,  2.1284e-08,  1.7275e-20,  4.1939e-38,  2.5316e-18,
             8.1513e-37,  9.9681e-39, -5.4767e-08,  4.1527e-36, -1.6897e-09,
            -4.1049e-08,  1.2895e-38,  3.8851e-19,  1.1109e-35, -8.7836e-07,
             4.6827e-25,  4.7698e-38,  3.7681e-38,  1.2022e-38, -2.2372e-08,
             2.5885e-38,  3.0097e-38,  4.7928e-38,  8.6353e-29,  1.5456e-17,
            -2.9933e-06, -3.1121e-06,  8.9508e-39,  4.1391e-39,  8.7141e-39,
             1.1619e-38,  4.0496e-39, -2.1389e-07, -3.4810e-08, -7.9283e-08,
             7.6374e-37, -9.0208e-09, -3.0963e-10,  4.5228e-38,  2.5259e-38,
            -4.5357e-08,  3.4121e-38,  2.0261e-37,  9.2239e-37,  8.5158e-28,
             1.6567e-27,  1.1093e-20,  3.6227e-39, -1.8451e-06, -2.4706e-06,
             4.2903e-08,  1.2067e-19, -2.0723e-06,  4.3839e-38,  6.3129e-39,
            -1.0063e-07,  9.2354e-39, -3.6195e-07,  4.4622e-27,  3.9577e-22,
             2.0495e-38, -3.1896e-08,  9.5112e-40,  4.8040e-22,  6.9059e-38,
             2.2367e-07,  6.6670e-39, -1.1867e-06,  4.5085e-39,  7.8998e-27,
             4.5546e-39,  2.0681e-37,  9.8615e-39,  9.6565e-39,  2.6019e-39,
             5.6120e-40,  1.1310e-35,  2.0500e-38,  2.3172e-39,  4.2281e-17,
             1.8658e-38,  1.3912e-29,  3.2049e-25,  5.3234e-36,  1.4995e-39,
             1.3759e-20,  6.7087e-40,  1.0798e-27, -2.2996e-08,  1.9189e-38,
             1.0241e-38, -5.4231e-08, -2.4715e-06,  2.8037e-38,  5.5246e-39,
             1.7275e-30,  2.9973e-38,  2.2433e-26,  7.2484e-38,  4.6093e-39,
            -1.0892e-15,  5.6828e-07, -2.8656e-09, -5.5907e-10,  8.0675e-39,
             2.8194e-39, -1.3315e-07, -1.4572e-08, -7.4477e-08, -3.9993e-06,
             2.2874e-30,  8.7833e-39, -2.1039e-08, -2.5876e-06,  1.2498e-39,
             6.7945e-36,  1.8361e-36, -2.9356e-06,  2.6065e-29,  9.3404e-39,
            -9.3819e-07, -4.2988e-08, -3.9538e-11,  4.9662e-37, -9.3441e-07,
            -1.5169e-06,  2.6993e-38,  8.6813e-29, -1.7722e-09,  1.7030e-38,
             3.4656e-38,  2.6353e-09,  5.9705e-38, -3.9788e-06, -1.9264e-09,
            -3.1317e-07, -1.4351e-18,  3.1955e-34,  5.8742e-41,  2.0423e-26,
             6.4100e-30,  3.1646e-07,  9.9825e-38,  1.3289e-38,  6.0719e-38,
            -3.7499e-06,  5.0201e-38,  2.5072e-38, -1.6480e-06, -1.0906e-07,
            -3.0579e-06,  2.1062e-38,  4.1077e-38,  7.7350e-07,  4.7125e-39,
             6.9095e-38, -3.3104e-08,  3.2927e-38, -2.6729e-06,  9.1583e-39,
            -2.7983e-06,  8.5494e-39,  4.4592e-38,  4.4645e-38, -1.3042e-06,
             2.8368e-38, -2.6650e-18, -1.1297e-06,  7.0904e-38, -2.1617e-06,
             2.3624e-38,  1.9240e-37,  3.6523e-39,  2.8915e-37,  1.1501e-38,
            -6.8881e-07,  2.3096e-06, -1.1478e-08,  2.9065e-33,  5.9329e-38,
             1.0605e-38, -1.0951e-06,  3.1245e-18,  1.6947e-39,  4.9329e-17,
             2.0875e-38,  8.6978e-20,  1.2764e-38,  2.8952e-39,  1.3526e-39,
             3.1020e-39,  1.0497e-39, -3.1149e-08, -4.5325e-08, -9.3377e-08,
             2.7716e-38,  4.1102e-38,  3.3085e-25,  2.8823e-30,  1.1684e-37,
             4.4197e-08,  3.0396e-38, -2.6545e-08,  1.3720e-37,  7.1060e-25,
             1.1511e-09,  2.1311e-40,  7.5219e-36,  9.8443e-39,  3.4517e-38,
            -1.3348e-07,  4.2060e-07,  2.7998e-22,  4.9937e-36,  2.3814e-38,
             9.7268e-23,  1.4910e-38,  3.3850e-38,  2.9722e-37,  3.3783e-38,
             2.3529e-38, -7.3977e-08,  1.4807e-38,  5.7783e-07,  2.1016e-38,
             3.7943e-24,  1.5689e-38,  5.5980e-38, -1.5684e-06, -2.4527e-06,
            -1.4383e-09,  1.7305e-38, -9.3614e-12,  4.2592e-38,  5.1048e-35,
            -1.6909e-06,  3.6117e-38,  5.4916e-30,  7.5690e-28,  2.3690e-38,
             2.3366e-24, -4.8068e-06,  1.8849e-38, -5.1981e-06,  7.7476e-39,
             1.0757e-38,  8.0675e-38,  5.4849e-38,  2.4683e-39,  2.1582e-38,
             4.0030e-39,  1.2643e-38, -1.7578e-08, -4.5309e-06,  2.5034e-37,
             3.0775e-39, -1.0492e-08,  1.2544e-39, -1.6025e-07, -1.1906e-06,
             2.7750e-07, -3.7673e-06, -2.8914e-07,  5.0325e-40,  3.8570e-31,
             3.4935e-38, -6.0124e-07, -2.5429e-06,  6.4777e-38, -7.9797e-10,
             2.1205e-37,  3.1606e-38,  1.8264e-38,  1.2413e-38,  7.9836e-24,
             5.9301e-38,  6.3344e-27,  7.1406e-23,  1.8558e-37,  1.2415e-38,
             9.1654e-39,  4.4317e-39], device='cuda:0'), 'exp_avg_sq': tensor([8.1645e-11, 1.1857e-06, 1.4435e-10, 7.5038e-09, 2.8463e-11, 3.5509e-09,
            6.1657e-13, 2.1201e-09, 1.1276e-09, 3.2030e-06, 1.0909e-09, 2.7384e-09,
            7.6479e-07, 1.1864e-11, 3.6106e-12, 1.3824e-12, 1.6841e-09, 9.5522e-11,
            3.0184e-10, 5.2076e-09, 8.3304e-11, 3.0924e-09, 2.4932e-08, 2.1584e-11,
            1.8247e-09, 1.8934e-09, 2.1197e-10, 3.7707e-10, 1.7980e-10, 8.7407e-10,
            5.9523e-09, 4.4744e-08, 4.6254e-10, 2.2293e-10, 1.5925e-10, 9.7899e-10,
            2.6222e-09, 4.5963e-09, 4.4684e-12, 1.3010e-08, 3.3593e-09, 6.3666e-10,
            1.5443e-11, 3.0137e-09, 5.5578e-09, 7.4069e-09, 6.7655e-09, 4.5881e-10,
            3.7203e-11, 1.0362e-09, 3.8212e-06, 1.1742e-10, 9.6789e-10, 3.0027e-09,
            1.8168e-09, 4.8817e-07, 5.9531e-10, 2.2112e-09, 5.1311e-07, 9.5798e-08,
            1.1019e-12, 4.1648e-10, 2.8008e-06, 1.8505e-11, 1.3372e-12, 1.1379e-10,
            3.2207e-07, 3.9577e-12, 5.1040e-11, 3.2729e-08, 5.8014e-07, 5.0196e-11,
            3.0378e-11, 1.1905e-10, 8.3110e-12, 4.3879e-10, 7.1662e-09, 2.2985e-07,
            1.1751e-09, 4.1021e-09, 2.8471e-07, 2.5035e-09, 4.4852e-12, 6.0000e-07,
            4.2510e-11, 1.0826e-10, 2.7085e-09, 7.4437e-07, 1.7080e-06, 4.6645e-09,
            2.6002e-11, 9.4134e-11, 1.4134e-09, 2.6157e-09, 2.7105e-13, 7.7143e-12,
            5.8905e-09, 4.7023e-12, 2.2208e-09, 1.1373e-06, 3.5184e-09, 9.7257e-11,
            4.1360e-09, 4.9332e-09, 1.9028e-08, 3.8677e-11, 6.5474e-07, 1.0587e-11,
            4.6359e-07, 6.9700e-07, 9.8023e-10, 1.6486e-07, 1.6660e-12, 4.6325e-09,
            4.3261e-13, 1.2438e-09, 3.5016e-10, 2.0907e-10, 3.6608e-11, 3.1969e-11,
            1.8186e-08, 1.3215e-08, 8.3623e-10, 6.0947e-12, 2.8264e-09, 4.8074e-07,
            2.6686e-09, 1.2775e-09, 1.5093e-08, 1.9737e-11, 1.9571e-09, 2.0842e-09,
            1.0973e-08, 3.6264e-07, 1.3435e-06, 2.1944e-08, 2.6899e-10, 1.3679e-06,
            8.7197e-10, 2.4554e-10, 1.5241e-09, 2.3652e-12, 9.6692e-10, 1.4857e-10,
            1.8199e-06, 1.5446e-06, 1.2355e-10, 2.2767e-09, 3.6550e-10, 4.2934e-10,
            9.1598e-08, 4.8396e-09, 1.0028e-10, 1.5032e-08, 4.2115e-10, 6.0452e-13,
            4.7613e-09, 5.3517e-09, 1.8353e-10, 1.5678e-11, 7.7851e-09, 6.3299e-13,
            8.2184e-09, 1.5110e-07, 8.0114e-09, 1.0888e-09, 3.0356e-09, 1.0024e-10,
            3.0876e-07, 2.5749e-09, 6.5653e-09, 1.3786e-09, 1.3008e-10, 1.0194e-11,
            4.2899e-10, 5.7233e-09, 8.3314e-11, 5.7392e-09, 3.5032e-09, 1.5788e-08,
            3.5225e-09, 2.6378e-08, 3.4695e-10, 1.5591e-09, 7.5551e-10, 4.9233e-10,
            3.5882e-11, 1.0878e-08, 2.1630e-06, 2.6246e-09, 4.1642e-11, 2.6870e-06,
            3.5366e-11, 2.1216e-09, 8.6337e-10, 1.2777e-09, 5.1391e-11, 1.6112e-08,
            1.4741e-09, 1.1235e-06, 3.7389e-09, 3.5850e-11, 4.4944e-07, 4.4765e-10,
            1.8583e-09, 5.9446e-13, 3.4231e-10, 9.6892e-07, 5.3715e-09, 5.6963e-09,
            5.5642e-11, 7.8820e-13, 1.6513e-10, 1.8413e-07, 4.8859e-10, 6.7326e-07,
            1.3971e-13, 1.0169e-06, 1.6604e-12, 2.1823e-09, 7.7164e-11, 1.9009e-10,
            6.5040e-07, 4.5110e-10, 4.4122e-07, 1.4889e-12, 5.9386e-10, 1.9084e-12,
            2.1945e-06, 3.3908e-10, 2.3652e-09, 6.5756e-11, 6.0641e-09, 2.9394e-09,
            5.7183e-12, 2.1524e-09, 2.0217e-10, 3.3621e-08, 4.6633e-10, 3.4345e-09,
            3.6680e-08, 3.3831e-10, 7.4316e-11, 3.9996e-09, 1.1131e-06, 4.2451e-09,
            4.6291e-09, 2.8890e-09, 1.9333e-10, 1.3134e-08, 1.1054e-09, 1.8430e-09,
            4.6738e-09, 1.4442e-09, 6.4337e-10, 1.4713e-06, 1.2057e-06, 8.6890e-11,
            3.4857e-11, 1.2527e-10, 2.7470e-10, 3.3367e-11, 7.7417e-09, 4.0259e-09,
            2.9734e-09, 4.6711e-12, 8.4973e-11, 2.8905e-08, 4.1620e-09, 1.0526e-09,
            3.4834e-08, 1.9206e-09, 4.7834e-09, 1.1090e-12, 6.8813e-09, 1.7780e-11,
            2.8548e-10, 2.6703e-11, 5.6268e-07, 8.8039e-07, 1.1441e-07, 5.5644e-11,
            1.0600e-06, 3.9103e-09, 8.1087e-11, 1.3057e-08, 1.7354e-10, 6.8598e-10,
            4.0711e-09, 4.3318e-09, 8.5463e-10, 6.8113e-09, 2.7877e-13, 1.2799e-09,
            9.7035e-09, 9.8147e-09, 3.9087e-11, 4.5280e-07, 4.1357e-11, 1.9140e-10,
            4.2208e-11, 1.1569e-10, 1.9787e-10, 1.8973e-10, 7.3420e-12, 6.4079e-13,
            3.2994e-10, 5.2305e-12, 1.0925e-11, 1.1261e-08, 7.0827e-10, 4.0161e-11,
            5.2200e-09, 1.1948e-10, 3.8569e-12, 4.8982e-09, 9.1573e-13, 9.1594e-11,
            2.4437e-09, 7.4920e-10, 1.1375e-10, 8.6790e-10, 7.3889e-07, 1.5994e-09,
            1.9497e-11, 1.1015e-10, 1.8279e-09, 7.8850e-11, 1.0690e-08, 4.3228e-11,
            2.7730e-10, 3.7572e-06, 5.8387e-09, 3.3373e-09, 1.3242e-10, 1.6173e-11,
            8.8698e-08, 4.9609e-09, 1.5189e-07, 7.8425e-07, 4.6001e-11, 1.5697e-10,
            3.8526e-10, 1.1836e-06, 3.1780e-12, 1.2314e-09, 6.1913e-12, 9.2314e-07,
            4.5153e-09, 1.7751e-10, 6.3527e-07, 5.7414e-10, 7.3589e-09, 1.3545e-10,
            1.5785e-08, 1.5340e-07, 1.4825e-09, 1.5957e-09, 2.1405e-09, 5.9008e-10,
            8.5635e-10, 1.2908e-08, 2.0941e-11, 5.3304e-07, 1.9214e-08, 2.5957e-06,
            2.4909e-10, 8.5071e-10, 5.6924e-15, 2.4803e-11, 4.8295e-10, 1.3362e-08,
            1.7084e-09, 2.9132e-10, 6.8714e-09, 1.4425e-06, 5.1275e-09, 1.2789e-09,
            1.1710e-07, 4.3216e-07, 9.6926e-07, 1.3670e-10, 3.4330e-09, 1.4800e-06,
            1.8212e-11, 4.5274e-10, 1.7092e-09, 1.8138e-09, 4.2608e-07, 1.7065e-10,
            6.2764e-07, 1.4872e-10, 4.0458e-09, 4.0553e-09, 1.5609e-08, 1.6374e-09,
            1.4008e-08, 4.9529e-08, 1.0229e-08, 3.5898e-07, 1.1355e-09, 1.2787e-10,
            1.7842e-11, 9.2286e-09, 2.6915e-10, 7.2050e-09, 7.4385e-08, 5.5513e-09,
            5.9823e-14, 7.1618e-09, 1.2197e-10, 1.6939e-06, 5.0155e-11, 4.7379e-12,
            6.4195e-10, 9.8854e-10, 1.9591e-09, 3.3150e-10, 1.7054e-11, 3.0182e-12,
            1.2871e-11, 2.2421e-12, 1.0942e-08, 2.7032e-08, 2.0580e-09, 1.5629e-09,
            3.4372e-09, 1.4460e-09, 1.4235e-10, 1.2969e-09, 7.1685e-10, 1.8799e-09,
            4.8371e-10, 4.1833e-09, 4.4305e-10, 1.7241e-08, 4.9257e-14, 5.7165e-09,
            1.9718e-10, 2.4241e-09, 4.5307e-09, 3.5828e-07, 2.8991e-09, 2.4158e-09,
            1.1539e-09, 1.3465e-09, 4.5235e-10, 2.3314e-09, 1.0260e-12, 7.9747e-10,
            1.1264e-09, 8.8554e-09, 4.4610e-10, 4.6899e-06, 8.9863e-10, 1.0453e-10,
            5.0082e-10, 6.3761e-09, 2.2957e-08, 5.4530e-07, 1.2043e-08, 6.0932e-10,
            1.8331e-09, 3.6910e-09, 1.6777e-12, 1.8448e-06, 2.6541e-09, 7.0400e-11,
            4.4941e-10, 1.1418e-09, 1.6291e-09, 2.2244e-06, 7.2291e-10, 6.7115e-07,
            9.9024e-11, 2.3544e-10, 6.9723e-09, 6.1210e-09, 1.0051e-11, 6.1601e-11,
            3.2603e-11, 1.7775e-10, 2.2020e-10, 2.4109e-06, 1.0671e-08, 1.5624e-11,
            7.0888e-09, 2.5959e-12, 4.2153e-08, 6.9876e-07, 1.4839e-06, 6.9220e-07,
            2.8072e-07, 4.1781e-13, 1.2952e-10, 2.0731e-09, 2.2406e-07, 1.2401e-06,
            8.5376e-09, 1.5815e-09, 9.4007e-09, 2.0325e-09, 6.7869e-10, 3.1349e-10,
            2.4426e-10, 3.0496e-09, 2.3711e-09, 1.9323e-11, 9.1396e-09, 3.1359e-10,
            1.7092e-10, 3.9961e-11], device='cuda:0')}, 139973667685648: {'step': 740, 'exp_avg': tensor([[-1.2143e-29,  3.5562e-05, -1.1128e-39,  ..., -2.3823e-38,
             -1.0957e-39, -7.7332e-40],
            [ 1.2143e-29, -3.5562e-05,  1.1128e-39,  ...,  2.3823e-38,
              1.0957e-39,  7.7332e-40]], device='cuda:0'), 'exp_avg_sq': tensor([[3.2265e-07, 3.8158e-05, 2.5194e-12,  ..., 1.1547e-09, 2.4429e-12,
             1.2168e-12],
            [3.2265e-07, 3.8158e-05, 2.5194e-12,  ..., 1.1547e-09, 2.4429e-12,
             1.2168e-12]], device='cuda:0')}, 139973667685720: {'step': 740, 'exp_avg': tensor([ 8.0949e-06, -8.0949e-06], device='cuda:0'), 'exp_avg_sq': tensor([0.0003, 0.0003], device='cuda:0')}}
    param_groups 	 [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'params': [139973667684784, 139973667684856, 139973667684928, 139973667685000, 139973667685072, 139973667685144, 139973667685288, 139973667685576, 139973667685648, 139973667685720]}]


# 5. Classify your own images/ Loading just model for inferencing 
1. dependencies mentioned here.
2. Loads best model trained.
3. Can run on the GPU.
4. <b>Load images by changing path below or replacing test.jpg in validate folder.</b>
5. Classification Ouputs Label in the form of Tensor index.<br>
Tensor [Swift, Innova]


```python
# Loading Saved model for inference 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Selecting whether to run on GPU (if available) or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
class Net(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        x = torch.randn(120,120).view(-1,1,120,120)
        self._to_linear = None
        self.convs(x)   # Partial forward pass
        self.fc1 = nn.Linear(self._to_linear, 512) 
        self.fc2 = nn.Linear(512, 2)  
    def convs(self, x):           
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,(2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = torch.load('checkpoint_acc_0.867_img_size120_batch_50_epochs_20.pt',map_location={'cuda:0': 'cpu'})
net.eval()

img_test = cv2.imread('validate/test_2.jpg', cv2.IMREAD_GRAYSCALE)
img_test = cv2.resize(img_test, (120,120))
test_data = []
test_data.append(np.array(img_test))       
test = torch.Tensor([test_data]).view(-1, 120, 120)
plt.imshow(img_test, cmap = "gray")
plt.show()

net_out = net(test.view(-1,1,120,120).to(device))
prediction = torch.argmax(net_out)

pred_num = prediction.numpy()

if pred_num == 1 :
    print("Prediction : " , prediction)
    print("The image is of a Toyota Innova")
    
else :
    print("Prediction : " , prediction)
    print("The image is of a Maruti Swift")  
```

    Running on the CPU



![png](output_22_1.png)


    Prediction :  tensor(0)
    The image is of a Maruti Swift

