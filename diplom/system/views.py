import os
import pickle
import re
import textwrap

import PyPDF2
import gensim as gensim
import numpy as np
import torch as torch
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404, FileResponse
from django.template import loader
from torch import cosine_similarity

from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk import word_tokenize, RegexpTokenizer, WordNetLemmatizer, pos_tag, FreqDist
from nltk.corpus import stopwords
from scipy import spatial

from .models import Articles, Words, Artword, Topics, Journals
from django.contrib import messages
from .forms import UserRegisterForm
from django.views.generic import (
    CreateView,
    DeleteView
)
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from transformers import BertTokenizerFast, BertModel

# Список русских стоп слов
rus_stop_words = stopwords.words("russian")

# Добавление стоп слов
stop_words_list = 'рис', 'это', 'также', 'которые', 'удк', 'гг', 'однако', 'тыс', 'которых', 'др', 'твой', 'которой', \
    'которого', 'свой', 'твоя', 'этими', 'слишком', 'нами', 'всему', 'будь', 'саму', 'чаще', 'ваше', 'сами', 'наш', \
    'затем', 'еще', 'самих', 'наши', 'ту', 'каждое', 'весь', 'этим', 'наша', 'своих', 'который', 'зато', 'те', 'этих', \
    'вся', 'ваш', 'такая', 'теми', 'ею', 'которая', 'нередко', 'каждая', 'также', 'чему', 'собой', 'самими', 'нем', \
    'вами', 'ими', 'откуда', 'такие', 'тому', 'та', 'очень', 'сама', 'нему', 'алло', 'оно', 'этому', 'кому', 'тобой', \
    'таки', 'твоё', 'каждые', 'твои', 'мой', 'нею', 'самим', 'ваши', 'ваша', 'кем', 'мои', 'однако', 'сразу', 'свое', \
    'ними', 'всё', 'неё', 'тех', 'хотя', 'всем', 'тобою', 'тебе', 'одной', 'другие', 'этого', 'само', 'эта', 'буду', \
    'самой', 'моё', 'своей', 'такое', 'всею', 'будут', 'своего', 'кого', 'свои', 'мог', 'нам', 'особенно', 'её', \
    'самому', 'наше', 'кроме', 'вообще', 'вон', 'мною', 'никто', 'это'

rus_stop_words.extend(stop_words_list)

# Буквы русского алфавита
alphabet_rus = ["а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
                "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]

# Буквы английского алфавита
alphabet_eng = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"]

# Буквы греческого алфавита
alphabet_gre = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ',
                'χ', 'ψ', 'ω']

# Странные буквы латинского алфавита
alphabet_latin = ['à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó',
                  'ô', 'õ', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ']


def frequent_words(text, lst):
    for token in text:
        lst.append(token)


# Первая часть предобработки текста, до токенизации
def pre1(text):
    # lowercasing
    text = text.lower()
    # removing extra whitespaces
    text = " ".join(text.split())
    # join words with -
    ind = 0
    for symb in text:
        if symb == '-' and ind + 3 < len(text) and text[ind + 1] == ' ':
            text = text[:ind] + text[ind + 1:]
            text = text[:ind] + text[ind + 1:]
            ind -= 2
        ind += 1
    # removing urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    return text


# Вторая часть преобразования текста, с токенизацией и после
def pre2(text):
    # tokenization
    text = word_tokenize(text, "russian")
    # removing punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    text = tokenizer.tokenize(' '.join(text))
    # removing russian stopwords and russian litters and numbers
    ind = 0
    for word in text:
        while ind < len(text) and (text[ind] in rus_stop_words or text[ind].isnumeric() or text[ind] in alphabet_rus):
            text.pop(ind)
        ind += 1
    # lemmatization
    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
        result.append(wordnet.lemmatize(token, pos))
    text = result
    return text


# Create your views here.

def articleList(request):
    article_list = Articles.objects.filter(temporary_bool=0)
    template = loader.get_template("system/index.html")
    context = {
        "article_list": article_list,
    }
    return HttpResponse(template.render(context, request))


def getArticle(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    author = article.user_name
    template = loader.get_template("system/article_page.html")
    context = {
        'article': article,
        'author': author,
    }
    return HttpResponse(template.render(context, request))


def pdf(request, id):
    pdf = Articles.objects.filter(article_id=id)[0]
    try:
        return FileResponse(open(pdf.pdf_path, 'rb'),
                            content_type='application/pdf')
    except FileNotFoundError:
        raise Http404()


def about(request):
    template = loader.get_template("system/about.html")
    context = {}
    return HttpResponse(template.render(context, request))


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Создан аккаунт {username}!')
            return redirect(request.META.get('HTTP_REFERER'))
    else:
        form = UserRegisterForm()
    return render(request, 'system/register.html', {'form': form})


@login_required
def profile(request):
    article_list = Articles.objects.filter(user_name=request.user.username)
    template = loader.get_template("system/profile.html")
    context = {
        "article_list": article_list,
    }
    return HttpResponse(template.render(context, request))


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Articles
    fields = ['title', 'journal', 'keywords', 'annotations', 'fio', 'topic', 'pdf_file']

    def form_valid(self, form):
        file = self.request.FILES['pdf_file']
        form.instance.temporary_bool = 1
        form.instance.statistics_bool = 1
        form.instance.user_name = self.request.user.username
        form.instance.pdf_path = r"C:/Users/user/PycharmProjects/diplom_new/diplom/database/pdf/" + file.name

        txt_path = r"C:/Users/user/PycharmProjects/diplom_new/diplom/database/txt/" + file.name
        txt_path = txt_path[:-4] + ".txt"

        form.instance.txt_path = txt_path

        form.save()

        pdffileobj = open(form.instance.pdf_path, 'rb')
        pdfreader = PyPDF2.PdfReader(pdffileobj)
        x = len(pdfreader.pages)
        pageobj = pdfreader.pages[0:x - 1]
        text = ''
        for i in range(0, x - 1):
            text += pageobj[i].extract_text()

        # Начало предобработки текста: шрифт, удаление лишних пробелов, удаление переноса на новую строку, url-ы
        text = pre1(text)

        # Удаление всех английских слов
        ew_ind = 0
        for eng_word in text:
            while ew_ind < len(text) and text[ew_ind] in alphabet_eng:
                _index = text.find(eng_word)
                text = text[:ew_ind] + text[ew_ind + len(eng_word):]
            ew_ind += 1

        # Удаление всех греческих букв
        gre_ind = 0
        for gre_word in text:
            while gre_ind < len(text) and text[gre_ind] in alphabet_gre:
                _index = text.find(gre_word)
                text = text[:gre_ind] + text[gre_ind + len(gre_word):]
            gre_ind += 1

        # Удаление всех латинских символов
        lat_ind = 0
        for lat_word in text:
            while lat_ind < len(text) and text[lat_ind] in alphabet_latin:
                _index = text.find(lat_word)
                text = text[:lat_ind] + text[lat_ind + len(lat_word):]
                lat_ind -= 1
            lat_ind += 1

        text = pre2(text)

        file1 = open(txt_path, "w", encoding="utf-8")
        for word in text:
            if len(word) > 1:
                file1.writelines(word)
                file1.writelines(' ')

        file1.close()
        pdffileobj.close()

        return super().form_valid(form)


def temporaryArticleList(request):
    temporaryarticle_list = Articles.objects.filter(temporary_bool=1)
    template = loader.get_template("system/admin_article_list.html")
    context = {
        "temporaryarticle_list": temporaryarticle_list,
    }
    return HttpResponse(template.render(context, request))


def getTemporaryArticle(request, id):
    temporary_article = Articles.objects.filter(article_id=id)[0]
    template = loader.get_template("system/temporary_article_page.html")
    context = {
        'temporary_article': temporary_article,
    }
    return HttpResponse(template.render(context, request))


def temporaryPdf(request, id):
    pdf = Articles.objects.filter(article_id=id)[0]
    try:
        return FileResponse(open(pdf.pdf_path, 'rb'),
                            content_type='application/pdf')
    except FileNotFoundError:
        raise Http404()


class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Articles
    success_url = '/system/profile/temporary_articles'

    def test_func(self):
        post = self.get_object()
        pdf_path = post.pdf_path
        txt_path = post.txt_path
        try:
            os.remove(pdf_path)
        except OSError as e:  # exception e
            print("Failed with:", e.strerror)
        try:
            os.remove(txt_path)
        except OSError as e:  # exception e
            print("Failed with:", e.strerror)
        return True


class PostDeleteView1(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Articles
    success_url = '/system'

    def test_func(self):
        post = self.get_object()
        pdf_path = post.pdf_path
        txt_path = post.txt_path
        try:
            os.remove(pdf_path)
        except OSError as e:  # exception e
            print("Failed with:", e.strerror)
        try:
            os.remove(txt_path)
        except OSError as e:  # exception e
            print("Failed with:", e.strerror)
        return True


def confirm(request, id):
    Articles.objects.filter(article_id=id).update(temporary_bool=0)

    temporaryarticle_list = Articles.objects.filter(temporary_bool=1)
    template = loader.get_template("system/admin_article_list.html")
    context = {
        "temporaryarticle_list": temporaryarticle_list,
    }
    return HttpResponse(template.render(context, request))


def avg_feature_vector(sentence, model, num_features, index2word_set):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in sentence:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def avg_feature_vectorFT(sentence, model, num_features):
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in sentence:
        n_words += 1
        feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


similar = 0.5

model_path = r"C:\Users\user\PycharmProjects\diplom_new\diplom\database\models"  # Заменить на свой путь


def similar_W2V(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    articles = Articles.objects.all()
    # Load pre-trained Word2Vec model.
    w2v_model = gensim.models.Word2Vec.load(
        model_path + "\w2v_model")
    index2word_set = set(w2v_model.wv.index_to_key)

    file = open(article.txt_path, "r", encoding="utf-8")
    txt = file.read()

    copomap = []

    s1_afv = avg_feature_vector(txt.split(), model=w2v_model, num_features=62, index2word_set=index2word_set)
    for i, doc in enumerate(articles):
        file = open(doc.txt_path, "r", encoding="utf-8")
        txt = file.read()
        s2_afv = avg_feature_vector(txt.split(), model=w2v_model, num_features=62, index2word_set=index2word_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        if sim >= similar:
            copomap.append({'similarity': sim, 'article': doc})

    template = loader.get_template("system/similar_W2V.html")
    context = {
        "copomap": copomap,
    }
    return HttpResponse(template.render(context, request))


def similar_D2V(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    articles = Articles.objects.all()

    d2v_model = gensim.models.Doc2Vec.load(
        model_path + "\d2v_model")
    index2Dword_set = set(d2v_model.wv.index_to_key)

    file = open(article.txt_path, "r", encoding="utf-8")
    txt = file.read()

    copomap = []

    s1_afv = avg_feature_vector(txt.split(), model=d2v_model, num_features=62, index2word_set=index2Dword_set)
    for i, doc in enumerate(articles):
        file = open(doc.txt_path, "r", encoding="utf-8")
        txt = file.read()
        s2_afv = avg_feature_vector(txt.split(), model=d2v_model, num_features=62, index2word_set=index2Dword_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        if sim >= similar:
            copomap.append({'similarity': sim, 'article': doc})

    template = loader.get_template("system/similar_D2V.html")
    context = {
        "copomap": copomap,
    }
    return HttpResponse(template.render(context, request))


def similar_LSA(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    articles = Articles.objects.all()

    docs = []

    for i, doc in enumerate(articles):
        file = open(doc.txt_path, "r", encoding="utf-8")
        text = file.read()
        text = text.split()
        docs.append(text)

    bigram = Phrases(docs, min_count=10, threshold=2, delimiter=',')

    bigram_phraser = Phraser(bigram)

    bigram_token = []
    for sent in docs:
        bigram_token.append(bigram_phraser[sent])

    # Создание словаря из текста документов
    dictionary = Dictionary(bigram_token)

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    lsa_model = gensim.models.LsiModel.load(
        model_path + "\lsa_model_bigram_minten")

    file = open(article.txt_path, "r", encoding="utf-8")
    txt1 = file.read()

    courseoutcome = txt1.split()

    covec = dictionary.doc2bow(courseoutcome)

    index = similarities.MatrixSimilarity(lsa_model[corpus])
    lsivec = lsa_model[covec]

    sims = index[lsivec]

    copomap = []
    for i, sim in enumerate(sims):
        if sim >= similar:
            copomap.append({'similarity': sim, 'article': articles[i]})

    template = loader.get_template("system/similar_LSA.html")
    context = {
        "copomap": copomap,
    }
    return HttpResponse(template.render(context, request))


def similar_FT(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    articles = Articles.objects.all()

    ft_model = gensim.models.FastText.load(
        model_path + "/ft_model")
    index2FTword_set = set(ft_model.wv.index_to_key)

    file = open(article.txt_path, "r", encoding="utf-8")
    txt = file.read()

    copomap = []

    s1_afv = avg_feature_vector(txt.split(), model=ft_model, num_features=62, index2word_set=index2FTword_set)
    for i, doc in enumerate(articles):
        file = open(doc.txt_path, "r", encoding="utf-8")
        txt = file.read()
        s2_afv = avg_feature_vector(txt.split(), model=ft_model, num_features=62, index2word_set=index2FTword_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        if sim >= similar:
            copomap.append({'similarity': sim, 'article': doc})

    template = loader.get_template("system/similar_FT.html")
    context = {
        "copomap": copomap,
    }
    return HttpResponse(template.render(context, request))


def docfunc(tokenizer, model, s):
    sum = 0
    tens = 0
    s_l = s[0]
    s_l = tokenizer.encode(s_l)
    s_l = torch.tensor(s_l)
    # print("2",s1) # prints tensor([ 101, 7592, ...
    s_l = s_l.unsqueeze(
        0)  # add an extra dimension, why ? the model needs to be fed in batches, we give a dummy batch 1
    # print("3",s1) # prints tensor([[ 101, 7592,
    with torch.no_grad():
        output_1 = model(s_l)
    logits_s = output_1[0]
    logits_s = torch.squeeze(logits_s)
    s_l = logits_s.reshape(1, logits_s.numel())
    tens = s_l

    for chunk in s:
        chunk = tokenizer.encode(chunk)
        chunk = torch.tensor(chunk)
        # print("2",s1) # prints tensor([ 101, 7592, ...
        chunk = chunk.unsqueeze(
            0)  # add an extra dimension, why ? the model needs to be fed in batches, we give a dummy batch 1
        # print("3",s1) # prints tensor([[ 101, 7592,
        with torch.no_grad():
            output_1 = model(chunk)
        logits_s = output_1[0]
        logits_s = torch.squeeze(logits_s)
        a = logits_s.reshape(1, logits_s.numel())
        tens = torch.cat((tens, a), 1)
    return tens


def similar_BERT(request, id):
    bert = BertModel.from_pretrained(model_path + "/BERT1/")
    tokenizer = BertTokenizerFast.from_pretrained("D:/data/")
    article = Articles.objects.filter(article_id=id)[0]
    articles = Articles.objects.all()

    copomap = []

    file = open(article.txt_path, "r", encoding="utf-8")
    text1 = file.read()

    text1 = textwrap.wrap(text1, 512)

    a = docfunc(tokenizer, bert, text1)
    for i, doc in enumerate(articles):
        file = open(doc.txt_path, "r", encoding="utf-8")
        text2 = file.read()
        text2 = textwrap.wrap(text2, 512)
        b = docfunc(tokenizer, bert, text2)

        if a.shape[1] < b.shape[1]:
            pad_size = (0, b.shape[1] - a.shape[1])
            a = torch.nn.functional.pad(a, pad_size, mode='constant', value=0)
        else:
            pad_size = (0, a.shape[1] - b.shape[1])
            b = torch.nn.functional.pad(b, pad_size, mode='constant', value=0)

        cos_sim = cosine_similarity(a, b)
        if cos_sim >= similar:
            copomap.append({'similarity': cos_sim, 'article': doc})

    template = loader.get_template("system/similar_BERT.html")
    context = {
        "copomap": copomap,
    }
    return HttpResponse(template.render(context, request))


def txt(request, id):
    txt = Articles.objects.filter(article_id=id)[0]
    try:
        return FileResponse(open(txt.txt_path, 'rb'),
                            content_type='application/txt')
    except FileNotFoundError:
        raise Http404()


def classifier(request, id):
    temporary_article = Articles.objects.filter(article_id=id)[0]

    w2v_model = gensim.models.Word2Vec.load(
        model_path + "\w2v_model")
    index2word_set = set(w2v_model.wv.index_to_key)

    LR_model = pickle.load(open(model_path + '\model.pkl', 'rb'))

    file = open(temporary_article.txt_path, "r", encoding="utf-8")
    txt = file.read()

    X_test_vect_avg = []

    s_afv = avg_feature_vector(txt.split(), model=w2v_model, num_features=62, index2word_set=index2word_set)
    X_test_vect_avg.append(s_afv)

    answer = LR_model.predict(X_test_vect_avg)

    template = loader.get_template("system/temporary_article_page.html")
    context = {
        "answer": answer,
        'temporary_article': temporary_article,
    }

    return HttpResponse(template.render(context, request))


def replacer(a):
    for i in range(0, len(a)):
        if a[i] == 'ё':
            a[i] = 'е'
    return a


def rusword(a):
    for i in range(0, len(a)):
        if not 'а' <= a[i] <= 'я':
            return False
    return True


def stats(request, id):
    article = Articles.objects.filter(article_id=id)[0]
    template = loader.get_template("system/stats.html")

    file = open(article.txt_path, "r", encoding="utf-8")
    text = file.read()
    text = text.split()

    lst = []
    frequent_words(text, lst)
    fdist = FreqDist(lst)

    if article.statistics_bool == 1:

        word = Words.objects.all()
        word_list = []
        for w in word:
            word_list.append(w.word)

        unique = []
        for token in text:
            if token not in unique and rusword(token):
                token = replacer(token)
                unique.append(token)
                if token not in word_list:
                    word_list.append(token)
                    wordObj = Words(word=token)
                    wordObj.save()

        for token in unique:
            artwordObj = Artword(quantity=fdist.get(token), article=article, word=Words.objects.filter(word=token)[0])
            Words.objects.filter()
            artwordObj.save()

        Articles.objects.filter(article_id=id).update(statistics_bool=0)

    unique = []
    for token in text:
        if token not in unique and rusword(token):
            token = replacer(token)
            unique.append(token)

    statisticS = [0] * 62

    for token in unique:
        sum = 0
        sumT = [0] * 62
        TF = [0] * 62
        ArtwordObj = Artword.objects.filter(word=Words.objects.filter(word=token)[0])
        for obj in ArtwordObj:
            sum += obj.quantity
            sumT[obj.article.topic.topic_id - 1] += obj.quantity
        ArtwordObj = Artword.objects.filter(word=Words.objects.filter(word=token)[0],
                                            article=Articles.objects.filter(article_id=id)[0])[0]
        for t in range(0, 62):
            TF[t] = sumT[t] / sum
            IDF = ArtwordObj.quantity / sum
            statisticS[t] += TF[t] * IDF

    topics = Topics.objects.order_by('topic_id')
    context = {
        "article": article,
        "most_common": fdist.most_common(25),
        "stat": statisticS,
        "topics": topics,
    }

    return HttpResponse(template.render(context, request))


def journalList(request):
    journal_list = Journals.objects.all()

    template = loader.get_template("system/journal_list.html")
    context = {
        'journal_list': journal_list,
    }
    return HttpResponse(template.render(context, request))


def journalGet(request, id):
    journal = Journals.objects.filter(journal_id=id)[0]

    article_list = Articles.objects.filter(journal=journal, temporary_bool=0)

    template = loader.get_template("system/journal_page.html")
    context = {
        'journal': journal,
        'article_list': article_list,
    }
    return HttpResponse(template.render(context, request))
