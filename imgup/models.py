from django.db import models


# Create your models here.
class User(models.Model):
    name = models.CharField(max_length=50)
    # upload_to 指定上传文件位置
    # 这里指定存放在 img/ 目录下
    headimg = models.FileField(upload_to="img/")
    
    # 返回名称
    def __str__(self):
        return self.name
def my_save(instance,filename):
    root="tag_images/"
    return "{0}/{1}/{2}".format(root,instance.tag,filename)
class TagImage(models.Model):
    tag=models.CharField(max_length=20)
    image=models.ImageField(upload_to=my_save)
    etc=models.CharField(max_length=10)

class Img(models.Model):
    img_url = models.ImageField(upload_to='img')