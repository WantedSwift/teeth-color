# Generated by Django 4.0 on 2022-11-15 16:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('colors', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='img',
            name='image',
            field=models.ImageField(upload_to='img/'),
        ),
    ]