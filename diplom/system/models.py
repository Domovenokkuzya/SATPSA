from django.db import models
from django.http import HttpResponseRedirect
from django.urls import reverse


# Create your models here.

class BaseModel(models.Model):
    objects = models.Manager()

    class Meta:
        abstract = True


class Articles(BaseModel):
    article_id = models.AutoField(primary_key=True)
    title = models.CharField(unique=True, max_length=500)
    journal = models.ForeignKey('Journals', models.DO_NOTHING, db_column='journal')
    keywords = models.TextField()
    annotations = models.TextField()
    fio = models.CharField(unique=False, max_length=300)
    pub_date = models.DateTimeField(blank=True, null=True, auto_now=True)
    pdf_path = models.CharField(max_length=500, blank=True, null=True)
    txt_path = models.CharField(max_length=500, blank=True, null=True)
    topic = models.ForeignKey('Topics', models.DO_NOTHING, db_column='topic')
    user_name = models.CharField(max_length=100)
    temporary_bool = models.IntegerField(blank=True, null=True)
    pdf_file = models.FileField(upload_to='pdf', blank=True, null=True)
    statistics_bool = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('profile')

    class Meta:
        managed = False
        db_table = 'articles'


class Topics(BaseModel):
    topic_id = models.AutoField(primary_key=True)
    topic_name = models.CharField(unique=True, max_length=255)

    def __str__(self):
        return self.topic_name

    class Meta:
        managed = False
        db_table = 'topics'


class Words(BaseModel):
    word_id = models.AutoField(primary_key=True)
    word = models.CharField(unique=True, max_length=45)

    def __str__(self):
        return self.word

    class Meta:
        managed = False
        db_table = 'words'


class Artword(BaseModel):
    artword_id = models.AutoField(primary_key=True)
    quantity = models.IntegerField()
    article = models.ForeignKey(Articles, models.DO_NOTHING, db_column='article')
    word = models.ForeignKey('Words', models.DO_NOTHING, db_column='word')

    class Meta:
        managed = False
        db_table = 'artword'


class Journals(BaseModel):
    journal_id = models.AutoField(primary_key=True)
    description = models.TextField(blank=True, null=True)
    journal_name = models.CharField(max_length=100, unique=True)

    class Meta:
        managed = False
        db_table = 'journals'

    def __str__(self):
        return self.journal_name
