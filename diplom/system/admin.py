from django.contrib import admin

from .models import Articles, Topics, Artword, Words, Journals

# Register your models here.

admin.site.register(Articles)
admin.site.register(Topics)
admin.site.register(Artword)
admin.site.register(Words)
admin.site.register(Journals)
