# Generated by Django 4.2.11 on 2024-05-04 15:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0019_remove_studystockdataindicatornormalvalue_stockdataitem_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='study',
            name='priceNormalizer',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='priceNormalizer', to='shop.studyindicator'),
        ),
        migrations.AddField(
            model_name='study',
            name='volumeNormalizer',
            field=models.ForeignKey(default=2, on_delete=django.db.models.deletion.CASCADE, related_name='volumeNormalizer', to='shop.studyindicator'),
        ),
    ]
