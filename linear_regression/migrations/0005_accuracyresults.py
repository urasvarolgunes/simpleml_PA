# Generated by Django 2.2.5 on 2022-07-24 23:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('linear_regression', '0004_testresult'),
    ]

    operations = [
        migrations.CreateModel(
            name='AccuracyResults',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('train', models.FloatField()),
                ('val', models.FloatField()),
                ('test', models.FloatField()),
                ('train_cnt', models.IntegerField()),
                ('val_cnt', models.IntegerField()),
                ('test_cnt', models.IntegerField()),
            ],
        ),
    ]
