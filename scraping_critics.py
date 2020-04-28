import json
import os
import time
import traceback

import numpy as np
from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import TooManyRedirects

"""
    Author: Sergey Volokhin (github: sergey-volokhin)

    This code scrapes Critics reviews from RottenTomatoes website
    for movies, which links are in "data/films_links.json" file.
"""


# mapping of most common critics scores to [1; 5] scale
score_map = {'*****': 5,
             '****': 4,
             '***': 3,
             '**': 2,
             '*': 1,
             'A-PLUS': 5,
             'A PLUS': 5,
             'A+': 5,
             'A': 5,
             'A-': 5,
             'A -': 5,
             'A MINUS': 5,
             'A-MINUS': 5,
             'B PLUS': 4,
             'B-PLUS': 4,
             'B +': 4,
             'B+': 4,
             'B': 4,
             'B-': 4,
             'B MINUS': 4,
             'B-MINUS': 4,
             'C PLUS': 3,
             'C-PLUS': 3,
             'C+': 3,
             'C': 3,
             'C-': 3,
             'C-MINUS': 3,
             'C MINUS': 3,
             'D+': 2,
             'D PLUS': 2,
             'D': 2,
             'D-': 2,
             'E+': 1,
             'E': 1,
             'E-': 1,
             'F+': 1,
             'F': 1,
             'F-': 1,
             }


# converting the critics score to [1; 5] scale
def calculate_score(score):
    if score is None or score.strip() == '':
        return np.nan
    score = score.strip().replace('  ', ' ')
    try:
        res = float(eval(score.replace("'", '').replace('"', '').replace(' stars out of ', '/').replace(' stars', '/5').replace(' out of ', '/').replace(' of ', '/')))
        if 0 <= res <= 1:
            return max(1, round(res * 5))
        return np.nan
    except Exception:
        pass
    try:
        return round(score_map[score.upper()])
    except Exception:
        return np.nan


# return soup of the page, after waiting for $crawl_rate$ seconds (to not get banned)
def make_soup(url, crawl_rate=1):
    time.sleep(crawl_rate)
    try:
        r = requests.get(url)
        return BeautifulSoup(r.content, 'html.parser')
    except TooManyRedirects:
        return ''


def get_critics_from_movie(movie):
    """ Scrapes all ids from critics, who left reviews about movie "movie" """

    soup = make_soup(f'https://www.rottentomatoes.com/m/{movie}/reviews')

    try:
        page_nums = int(soup.find('span', class_='pageInfo').text[9:])
    except AttributeError:
        page_nums = 1

    critics = []
    for page_num in range(1, page_nums + 1):
        page_soup = make_soup(f'https://www.rottentomatoes.com/m/{movie}/reviews?page={page_num}&sort=')
        reviews_soups = page_soup.find_all('div', class_='row review_table_row')
        for review_soup in reviews_soups:
            try:
                critics += [review_soup.find('a', class_='unstyled bold articleLink')['href'][8:]]
            except Exception:
                pass
    return critics


def get_reviews_from_critic(critic):
    """
        Scrapes all reviews left by critic "critic".
        Includes critic_id, movie_id, text of the review, score, and "freshness" of the movie.
        Writes excepted critics ids into "datapath/failed_critics.txt"
    """
    try:
        page = requests.get(f'https://www.rottentomatoes.com/napi/critic/{critic}/review/movie?offset=0').json()

        all_reviews_f_critic = []
        offset = 0
        total = page['totalCount']
        while offset < total:
            time.sleep(1)  # to not get banned
            page = requests.get(f'https://www.rottentomatoes.com/napi/critic/{critic}/review/movie?offset={offset}').json()
            for review in page['results']:
                current_review = {'critic_id': critic}
                try:
                    current_review['movie_id'] = review['media']['url'][33:].replace('-', '_')
                except Exception:
                    continue
                current_review['fresh'] = review['score']
                current_review['score'] = review['scoreOri']
                current_review['review'] = review['quote']
                all_reviews_f_critic.append(current_review)
            offset += len(page['results'])
        return all_reviews_f_critic

    except Exception as err:
        print(f"couldn't get reviews for {critic}. {err}")
        traceback.print_exc()
        open(f'{datapath}/failed_critics.txt', 'a+').write(critic + '\n')
        return []


def get_reviews_from_movie(page):
    """
        Function not currently used.
        Scrapes all reviews for movie "movie".
    """

    soup = make_soup(f'https://www.rottentomatoes.com/m/{page}/reviews')

    # getting the amount of pages of reviews
    try:
        page_nums = int(soup.find('span', class_='pageInfo').text[9:])
    except AttributeError:
        page_nums = 1

    reviews = []
    for page_num in range(1, page_nums + 1):
        page_soup = make_soup(f'https://www.rottentomatoes.com/m/{page}/reviews?page={page_num}&sort=')
        reviews_soups = page_soup.find_all('div', class_='row review_table_row')
        for review_soup in reviews_soups:
            cur_review = {}
            cur_review['movie_id'] = page.replace('-', '_')
            try:
                cur_review['critic_id'] = review_soup.find('a', class_='unstyled bold articleLink')['href'][8:]
            except Exception:
                continue

            # getting text
            cur_review['review'] = review_soup.find('div', class_='the_review').text.strip()

            # getting freshness
            if review_soup.find('div', class_='review_icon icon small fresh') is not None:
                cur_review['fresh'] = 'fresh'
            else:
                cur_review['fresh'] = 'rotten'

            # getting score
            try:
                cur_review['score'] = review_soup.find('div', class_='small subtle review-link').text.split('Original Score: ')[1].split('\n')[0]
            except Exception:
                cur_review['score'] = ''

            reviews.append(cur_review)
    return reviews


if __name__ == '__main__':

    movies = json.load(open('films_links.json', 'r'))

    datapath = 'RT_reviews'

    if not os.path.exists(datapath):
        os.system('mkdir ' + datapath)

    critics = []
    for film in tqdm(movies):
        new_critics = get_critics_from_movie(film)
        critics += new_critics
        break

    critics = sorted(set(critics))

    print('Critics collection finished')
    print('\n============================================\n')

    reviews = []
    print(f'GETTING REVIEWS FROM {len(critics)} CRITICS')
    for critic in tqdm(critics):
        if critic != '':
            reviews += get_reviews_from_critic(critic)

    df_all_reviews = pd.DataFrame.from_records(reviews).dropna().drop_duplicates(subset=['movie_id', 'critic_id'])
    df_all_reviews['score'] = df_all_reviews['score'].apply(calculate_score)
    df_all_reviews = df_all_reviews.replace('', np.nan).dropna()

    # cleanup of common templates
    df_all_reviews = df_all_reviews[df_all_reviews['movie_id'] != ':vanity']
    df_all_reviews['review_lower'] = df_all_reviews['review'].apply(lambda x: str(x).lower())
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].isin(['see website for more details.', '.'])]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('click to ')]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('click for ')]
    df_all_reviews = df_all_reviews[~df_all_reviews['review_lower'].str.startswith('full review ')]
    df_all_reviews.drop('review_lower', axis=1, inplace=True)

    prev_shape = df_all_reviews.shape
    new_shape = 0
    popularity_thres = 50

    # removing critics and movies with less than 50 reviews (until convergence)
    while prev_shape != new_shape:
        prev_shape = df_all_reviews.shape
        df_movies_cnt = pd.DataFrame(df_all_reviews.groupby('movie_id').size(), columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
        df_ratings_drop_movies = df_all_reviews[df_all_reviews.movie_id.isin(popular_movies)]
        df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('critic_id').size(), columns=['count'])
        prolific_users = list(set(df_users_cnt.query('count >= @popularity_thres').index))
        df_all_reviews = df_ratings_drop_movies[df_ratings_drop_movies.critic_id.isin(prolific_users)]
        new_shape = df_all_reviews.shape

    df_all_reviews.to_csv(f'{datapath}/reviews_clean.tsv', index=False, sep='\t')

    print('Reviews collection finished')
    print('\n============================================\n')
    print('TOTAL CRITICS:', df_all_reviews['critic_id'].nunique())
    print('TOTAL REVIEWS:', df_all_reviews.shape[0])
    print('TOTAL MOVIES:', df_all_reviews['movie_id'].nunique())
    print('Median of reviews per critic:', df_all_reviews.groupby('critic_id').size().median())
    print('Mean of reviews per critic:', df_all_reviews.groupby('critic_id').size().mean())
