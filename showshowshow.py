import os  
import numpy as np  
  
import torch  
import torch.nn  
import Inception_v3

import torchvision.utils as utils
from torch.autograd import Variable   
import torch.cuda  
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt
  
from PIL import Image  
  
img_to_tensor = transforms.ToTensor()  
  
def make_model():  
    v3model=Inception_v3.inception_v3(pretrained=True) 
    return v3model
  
# inference
def inference(v3model,imgpath):  
    v3model.eval()#
      
    img=Image.open(imgpath)  # open images, resize and to tensor
    img=img.resize((299,299))  
    tensor=img_to_tensor(img)  
      
    tensor=tensor.resize_(1,3,299,299)  
              
    result=v3model(Variable(tensor))  # compute...
    result_npy=result.data.cpu().numpy()
    max_index=np.argmax(result_npy[0])  
    
    return max_index  
      
#feature
def extract_feature(v3model,imgpath):  
    v3model.fc=torch.nn.LeakyReLU(0.1)  
    v3model.eval()  
    
    img=Image.open(imgpath)  
    img=img.resize((299,299))  
    tensor=img_to_tensor(img)  
    
    tensor=tensor.resize_(1,3,299,299)  
    
    result=v3model(Variable(tensor))
    # test tensor grid
    result_npy=result.data.cpu().numpy()
    
    #return result_npy[0]
    return result.data

# imshow, matplotlib inline
#def imshow(img):
#    npimg = img.numpy()
#    plt.imshow(npimg)
      
if __name__=="__main__":  
    model=make_model()
    
   #print model
    imgpath='./IMG_5190.jpg'  
    print inference(model,imgpath)
    #imshow(extract_feature(model, imgpath))
    #print extract_feature(model, imgpath)
    #data=extract_feature(model, imgpath)
    #utils.make_grid(data, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)