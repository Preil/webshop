# Generated by Django 4.2.10 on 2024-03-02 19:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0005_study_enddate_study_startdate'),
    ]

    operations = [
        migrations.AddField(
            model_name='study',
            name='timeFrame',
            field=models.CharField(default='1Day', max_length=12),
        ),
    ]