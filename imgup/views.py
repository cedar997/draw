
from django.shortcuts import render
from .models import TagImage,Img
from .forms import AddForm
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  
import numpy as np
import re
import base64
from PIL import Image
from io import BytesIO
from io import StringIO
from django.template import engines
from django.http import HttpResponse
# Create your views here.
from django.core.files.base import ContentFile

from django.views.decorators.csrf import  csrf_exempt
@csrf_exempt
def index(request):
    
    if request.method=="POST":
        image_data = request.POST['image_data']
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        image_data = dataUrlPattern.match(image_data).group(2)
        image_data = image_data.encode()
        image_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_data))
        
        pred=predict(img)
        return render(request, 'index.html', context={"pred":pred}) 
    else:
        pred="未输入"
        return render(request,'index.html',context={"pred":pred})

def upLoad(request):
    tag="none"
    if request.method=="POST":
        tag= request.POST['tag']
        etc=request.POST['etc']
        ti = TagImage(tag=tag, etc=etc)
        print(etc)
        print(request.POST)
        if etc=="draw":
            image_data = request.POST['image_data']
            dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
            image_data = dataUrlPattern.match(image_data).group(2)
            image_data = image_data.encode()
            image_data = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_data))
            img.thumbnail((28, 28), Image.ANTIALIAS)
            # tensor = torchvision.transforms.ToTensor()(img)
            img_io=BytesIO()
            img.save(img_io,img.format,quality=60)
            # return response

            ti.image.save("1.png",ContentFile(img_io.getvalue()))
            ti.save()
        elif etc=="file":
            img_file=request.FILES.get('img')
            img=Image.open(img_file)
            img.thumbnail((28, 28), Image.ANTIALIAS)
            img_io = BytesIO()
            img.save(img_io, img.format, quality=60)
            ti.image.save("1.png",ContentFile(img_io.getvalue()))
            ti.save()
    pred=TagImage.objects.last()
    return render(request, 'upLoad.html', context={"pred": pred})
def showImg(request):
    imgs = TagImage.objects.all() # 从数据库中取出所有的图片路径
    context = {
        'imgs' : imgs
    }
    return render(request, 'showImg.html', context)


def predict(img):
    cnn=CNN2()
    cnn.cuda()
    cnn.load_state_dict(torch.load("./net.pkl"))
    print(cnn)
    img=torchvision.transforms.Scale(28)(img)
    img=torchvision.transforms.Grayscale()(img)
    # img.show()
    tensor=torchvision.transforms.ToTensor()(img)
    tensor.reshape(1,28,28)
    tensor=torch.unsqueeze(tensor,dim=1).cuda()
    predictions = cnn(tensor).cpu()
    return predictions.argmax(dim=1).item()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),)
        #self.conv3=nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten(),)
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x=self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
