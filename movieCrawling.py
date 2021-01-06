from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import csv
import urllib
from selenium.webdriver.common.action_chains import ActionChains

options = webdriver.ChromeOptions()
options.add_experimental_option('prefs', {'intl.accept_languages': 'vi'})
browser = webdriver.Chrome(options=options)
browser.maximize_window()

records = []


def crawlAllComment(url):
    records = []
    browser.get(url)
    # tat quang cao
    browser.find_element_by_css_selector('html').send_keys(Keys.ESCAPE)

    # scroll
    time.sleep(1)
    total_height = int(browser.execute_script(
        "return document.body.scrollHeight"))
    for i in range(1, total_height, 5):
        browser.execute_script("window.scrollTo(0, {});".format(i))
    time.sleep(3)

    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')

    # tat dialog
    button = browser.find_elements_by_css_selector(
        'button.align-right.secondary.slidedown-button')

    if(len(button) > 0):
        button[0].click()

    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')

    commentDivs = soup.findAll(
        "div", {"class": "style__StyledComment-sc-103p4dk-5 dDtAUu review-comment"})
    for commentDiv in commentDivs:
        cmt = commentDiv.find(
            "div", {"class": "review-comment__content"}).text
        if(len(cmt) > 0):
            records.append(cmt)

    navigationButton = browser.find_elements_by_css_selector(
        'a.btn.next')

    while len(navigationButton) > 0:
        navigationButton[0].click()
        time.sleep(1)
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html.parser')
        commentDivs = soup.findAll(
            "div", {"class": "style__StyledComment-sc-103p4dk-5 dDtAUu review-comment"})
        for commentDiv in commentDivs:
            cmt = commentDiv.find(
                "div", {"class": "review-comment__content"}).text
            if(len(cmt) > 0):
                records.append(cmt)
        navigationButton = browser.find_elements_by_css_selector(
            'a.btn.next')

    return records
# class="keywords right_column"


def crawlMovieInfomation(url):
    browser.get(url)
    # tat quang cao
    browser.find_element_by_css_selector('html').send_keys(Keys.ESCAPE)

    # scroll
    time.sleep(1)
    total_height = int(browser.execute_script(
        "return document.body.scrollHeight"))
    for i in range(1, total_height, 5):
        browser.execute_script("window.scrollTo(0, {});".format(i))
    time.sleep(1)

    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')

    img = soup.find('img', {'class': 'poster'})
    imgFileName = ''
    if(img != None):
        imgFileName = 'poster/'+url.split("/")[len(url.split("/"))-1]+'.jpg'
        urllib.request.urlretrieve(
            'https://www.themoviedb.org'+img.get('src'), imgFileName)

    title = ''
    titleDiv = soup.find('div', {'class': 'title ott_false'})
    if(titleDiv!=None):
        titleDiv=titleDiv.find('h2')
        if(titleDiv != None):
            title = titleDiv.text.strip()

    user_score_chart_div = soup.find('div', {'class': 'user_score_chart'})
    user_score_chart = ''
    if(user_score_chart_div != None):
        user_score_chart = user_score_chart_div.get('data-percent')

    certificationDiv = soup.find('span', {'class': 'certification'})
    certification = ''
    if(certificationDiv != None):
        certification = certificationDiv.text.strip()

    overviewDiv = soup.find('div', {'class': 'overview'})
    overview = ''
    if(overviewDiv != None):
        overview = overviewDiv.text.strip()

    taglineDiv = soup.find('h3', {'class': 'tagline'})
    tagline = ''
    if(taglineDiv != None):
        tagline = taglineDiv.text.strip()

    genresDiv = soup.find('span', {'class': 'genres'})
    genres = ''
    if(genresDiv != None):
        genres = genresDiv.text.strip()

    trailerDiv = soup.findAll('a', {'class': 'no_click play_trailer'})
    trailerUrl = ''
    if(trailerDiv != None and len(trailerDiv) > 0):
        trailerUrl = 'https://www.themoviedb.org' + \
            trailerDiv[1].get('href')

    return {'url': url, 'title': title, 'score': user_score_chart, 'posterImagePath': imgFileName,
            'certification': certification, 'overview': overview, 'tagline': tagline, 'genres': genres, 'trailerUrl': trailerUrl}

    # button = browser.find_elements_by_css_selector(
    #     'div.filter-review__item')

    # if(len(button) > 0):
    #     button[3].click()
    #     time.sleep(1)

    #     html_source = browser.page_source
    #     soup = BeautifulSoup(html_source, 'html.parser')

    #     commentDivs = soup.findAll(
    #         "div", {"class": "style__StyledComment-sc-103p4dk-5 dDtAUu review-comment"})
    #     for commentDiv in commentDivs:
    #         cmt = commentDiv.find(
    #             "div", {"class": "review-comment__content"}).text
    #         title = commentDiv.find(
    #             "a", {"class": "review-comment__title"}).text
    #         record = {'comment': title+', '+cmt, 'is_trust': 1}
    #         print(record)
    #         records.append(record)

    #     button[3].click()
    #     button[6].click()
    #     button[7].click()
    #     time.sleep(1)

    #     html_source = browser.page_source
    #     soup = BeautifulSoup(html_source, 'html.parser')

    #     commentDivs = soup.findAll(
    #         "div", {"class": "style__StyledComment-sc-103p4dk-5 dDtAUu review-comment"})

    #     for commentDiv in commentDivs:
    #         cmt = commentDiv.find(
    #             "div", {"class": "review-comment__content"}).text
    #         title = commentDiv.find(
    #             "a", {"class": "review-comment__title"}).text
    #         record = {'comment': title+', '+cmt, 'is_trust': 0}
    #         print(record)
    #         records.append(record)

    # return records


def getProductDetailUrl(url):
    browser.get(url)
    # tat quang cao
    browser.find_element_by_css_selector('html').send_keys(Keys.ESCAPE)

    time.sleep(1)
    total_height = int(browser.execute_script(
        "return document.body.scrollHeight"))
    for i in range(1, total_height, 5):
        browser.execute_script("window.scrollTo(0, {});".format(i))
    time.sleep(1)

    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')

    # movieUrl = 'https://www.themoviedb.org' + \
    #     soup.find('div', {'id': 'trending_scroller'}).find(
    #         'div').find('div').find('h2').find('a').get('href')
    # return movieUrl

    time.sleep(1)
    t = 'document.body.scrollHeight'
    browser.execute_script(
        "window.scrollTo(0, {});".format(t))
    time.sleep(1)
    button = browser.find_elements_by_css_selector(
        'a.no_click.load_more')

    if(len(button) > 0):
        button[1].click()
        time.sleep(1)
    for i in range(60):  # scroll 3000 times
        browser.find_element_by_css_selector('html').send_keys(Keys.END)
        time.sleep(1)

    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')

    wrappers = soup.findAll('div', {'class': 'wrapper'})
    movieUrls = []
    for wrapper in wrappers:
        href = 'https://www.themoviedb.org'+wrapper.find('a').get('href')
        print(href)
        movieUrls.append(href)
    return movieUrls

# productUrls = getProductDetailUrl('https://tiki.vn/deal-hot?tab=now&page=1')

# fieldList = ['comment', 'is_trust']
# try:
#     with open('khdl_final_final6.csv', 'w', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldList)
#         writer.writeheader()
#         for productUrl in productUrls:
#             records = crawlComment(productUrl)
#             for record in records:
#                 print(record)
#                 writer.writerow(
#                     {'comment': record['comment'], 'is_trust': record['is_trust']})
# except IOError:
#     print("I/O error")
# print(crawlComment('https://www.themoviedb.org/movie/464052-wonder-woman-1984'))


def scroll(driver, timeout):
    scroll_pause_time = timeout

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # If heights are the same it will exit the function
            break
        last_height = new_height


categoriesUrls = ['https://www.themoviedb.org/tv', 'https://www.themoviedb.org/tv/on-the-air', 'https://www.themoviedb.org/tv/top-rated',
                  'https://www.themoviedb.org/movie', 'https://www.themoviedb.org/movie/now-playing', 'https://www.themoviedb.org/movie/top-rated']



# urls = []
# for url in categoriesUrls:
#     urls += getProductDetailUrl(url)
# print(len(urls))

# fieldList = ['id', 'movie_url']
# try:
#     with open('movie_url.csv', 'w', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldList)
#         writer.writeheader()
#         i = 0
#         for url in urls:
#             writer.writerow({'id': i, 'movie_url': url})
#             print(i)
#             i += 1
# except IOError:
#     print("I/O error")
# print(crawlComment('https://www.themoviedb.org/movie/464052-wonder-woman-1984'))


def movieUrls(fileName):
    file = open(fileName, 'r')
    urls = file.readlines()
    for i in range(len(urls)):
        urls[i] = urls[i].replace("\n", "").split(",")[1]
    urls.pop(0)
    return urls


urls = movieUrls('movie_url.csv')

fieldList = ['url', 'title', 'score', 'posterImagePath',
             'certification', 'overview', 'tagline', 'genres', 'trailerUrl']
try:
    with open('movie_part2.csv', 'w', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldList)
        writer.writeheader()
        i = 4342
        for url in urls[4342:]:
            record = crawlMovieInfomation(url)
            print(i, record)
            writer.writerow(record)
            i += 1
except IOError:
    print("I/O error")

browser.quit()
