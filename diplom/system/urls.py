from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import PostCreateView, PostDeleteView, PostDeleteView1

urlpatterns = [
    path("", views.articleList, name="index"),
    path('articles/<int:id>/', views.getArticle, name='article_url'),
    path('articles/<int:id>/pdf', views.pdf, name='pdf'),
    path('about', views.about, name="about"),
    path('register', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('login/', auth_views.LoginView.as_view(template_name='system/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='system/logout.html'), name='logout'),
    path('new_article/',  PostCreateView.as_view(template_name='system/new_article.html'), name='new_article'),
    path("profile/temporary_articles", views.temporaryArticleList, name="temporary_articles_list"),
    path('profile/temporary_articles/<int:id>/', views.getTemporaryArticle, name='temporary_article_url'),
    path('profile/temporary_articles/<int:id>/temporary_pdf', views.temporaryPdf, name='temporary_pdf'),
    path('profile/temporary_articles/<int:pk>/delete/', PostDeleteView.as_view(template_name='system/admin_delete.html'), name='article_delete'),
    path('profile/temporary_articles/<int:id>/confirm', views.confirm, name='confirm'),
    path('articles/<int:id>/similar_W2V', views.similar_W2V, name='similar_W2V'),
    path('profile/temporary_articles/<int:id>/txt', views.txt, name='txt'),
    path('articles/<int:id>/similar_D2V', views.similar_D2V, name='similar_D2V'),
    path('articles/<int:id>/similar_LSA', views.similar_LSA, name='similar_LSA'),
    path('articles/<int:id>/similar_FT', views.similar_FT, name='similar_FT'),
    path('profile/temporary_articles/<int:id>/classifier', views.classifier, name='classifier'),
    path('articles/<int:id>/similar_BERT', views.similar_BERT, name='similar_BERT'),
    path('articles/<int:id>/stats', views.stats, name='stats'),
    path('journal', views.journalList, name='journal_list'),
    path('journal/<int:id>', views.journalGet, name='journal'),
    path('articles/<int:pk>/article_delete/', PostDeleteView1.as_view(template_name='system/delete.html'), name='article_delete')
]
