# Generated by Django 2.2.5 on 2022-06-22 03:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('linear_regression', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='EdgeData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('graph_id', models.IntegerField()),
                ('node1_id', models.IntegerField()),
                ('node2_id', models.IntegerField()),
            ],
        ),
    ]