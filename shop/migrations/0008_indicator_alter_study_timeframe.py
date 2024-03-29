# Generated by Django 4.2.10 on 2024-03-10 15:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0007_stockdata_study'),
    ]

    operations = [
        migrations.CreateModel(
            name='Indicator',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=40)),
                ('description', models.CharField(max_length=255)),
                ('functionName', models.CharField(max_length=50)),
                ('parameters', models.CharField(max_length=255)),
            ],
        ),
        migrations.AlterField(
            model_name='study',
            name='timeFrame',
            field=models.CharField(default='1day', max_length=12),
        ),
    ]
