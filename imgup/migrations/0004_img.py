# Generated by Django 3.0.3 on 2020-03-30 15:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imgup', '0003_auto_20200330_1404'),
    ]

    operations = [
        migrations.CreateModel(
            name='Img',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img_url', models.ImageField(upload_to='img')),
            ],
        ),
    ]
