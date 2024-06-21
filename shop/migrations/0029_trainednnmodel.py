# Generated by Django 4.2.13 on 2024-06-21 12:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0028_nnmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainedNnModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('serialized_model', models.BinaryField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('nn_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shop.nnmodel')),
            ],
        ),
    ]
