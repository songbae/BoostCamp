from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
import time
from itertools import product
import pandas as pd


def preprocess_stat(result, elem):
    stat = elem.find_elements_by_tag_name('span')
    if stat:
        result['stat'] = stat[0].text


def preprocess_info(result, elem):
    mbti = elem.find_elements_by_tag_name('h4')
    enne = elem.find_elements_by_tag_name('h5')
    if mbti:
        result['mbti'] = mbti[0].text

    if enne:
        result['enneagram'] = enne[0].text

    result['role'] = elem.find_element_by_class_name(
        'card-container-profile-name').text

    result['movie'] = elem.find_element_by_class_name(
        'card-container-subcategory').text

    while True:
        img_url = elem.find_element_by_class_name('card-container-image').find_elements_by_tag_name(
            'img')

        if img_url:
            result['img_url'] = img_url[0].get_attribute('src')
            break

        print(result['role'])
        time.sleep(1)


def page_crawling(browser):
    result = []
    info_blocks = browser.find_elements_by_class_name('card-container')
    stat_blocks = browser.find_elements_by_class_name('card-stats-container')

    for stat, info in zip(stat_blocks, info_blocks):
        tot = dict()
        preprocess_stat(tot, stat)
        preprocess_info(tot, info)
        result.append(tot)
    return result


def crawling_by_type(browser, mbti, category):
    result = []
    # Browser Open
    url = f'https://www.personality-database.com/personality_type/{mbti}'
    browser.get(url)
    time.sleep(3)

    # Category
    select = Select(browser.find_element_by_id('category-filter'))
    select.select_by_visible_text(category)

    # List Size
    time.sleep(3)
    select = Select(browser.find_element_by_id('list-size'))
    select.select_by_visible_text('20')
    time.sleep(3)
    select = Select(browser.find_element_by_id('list-size'))
    select.select_by_visible_text('10')

    page_count = 1
    while True:
        time.sleep(7)
        print(f'{mbti} page : {page_count}\r', end='')
        result += page_crawling(browser)
        next_button = browser.find_element_by_class_name('next')
        if next_button.find_element_by_tag_name('a').get_attribute('aria-disabled') == 'true':
            break
        page_count += 1
        next_button.click()

    print(f'{mbti} : {len(result)}')
    return result


if __name__ == '__main__':
    chromedriver = './chromedriver/chromedriver'
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--no-sandbox")
    browser = webdriver.Chrome(chromedriver, chrome_options=chrome_options)

    mbti_elems = [
        ('E', 'I'),
        ('S', 'N'),
        ('F', 'T'),
        ('P', 'J'),
    ]

    category = 'Movies'

    result = []
    for mbti in product(*mbti_elems):
        mbti = ''.join(mbti)
        print(f'Start Crawling {mbti}')
        sub_result = crawling_by_type(browser, mbti, category)
        pd.DataFrame(sub_result).to_csv(f'{mbti}.csv')  # for safety
        result += sub_result

    browser.close()
    result = pd.DataFrame(result)
    print(result.head())
    result.to_csv('mbti.csv', index=False)
