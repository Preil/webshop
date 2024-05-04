# Generated by Django 4.2.11 on 2024-05-04 12:29

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0017_alter_studyindicator_mask'),
    ]

    operations = [
        migrations.AlterField(
            model_name='studyorder',
            name='limitPrice',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.AlterField(
            model_name='studyorder',
            name='lpoffsetTP',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.AlterField(
            model_name='studyorder',
            name='slTP',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.AlterField(
            model_name='studyorder',
            name='stopLoss',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.AlterField(
            model_name='studyorder',
            name='takeProfit',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.AlterField(
            model_name='studyorder',
            name='tpTP',
            field=models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True),
        ),
        migrations.CreateModel(
            name='StudyStockDataIndicatorNormalValue',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('normalValue', models.CharField(max_length=255)),
                ('stockDataItem', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shop.stockdata')),
                ('studyIndicator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shop.studyindicator')),
            ],
        ),
        migrations.CreateModel(
            name='StudyOrderNormalValues',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('normLimitPrice', models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True)),
                ('normTakeProfit', models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True)),
                ('normStopLoss', models.DecimalField(blank=True, decimal_places=8, max_digits=16, null=True)),
                ('studyOrder', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='shop.studyorder')),
            ],
        ),
    ]
