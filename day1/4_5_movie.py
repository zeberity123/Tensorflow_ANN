# 4_5_movie.py
import numpy as np
import pandas as pd

users = pd.read_csv('ml-1m/users.dat',
                    header=None,
                    delimiter='::',
                    engine='python',
                    names='UserID::Gender::Age::Occupation::Zip-code'.split('::'))
# print(users)

movies = pd.read_csv('ml-1m/movies.dat',
                     header=None,
                     delimiter='::',
                     engine='python',
                     names='MovieID::Title::Genres'.split('::'),
                     encoding='ISO-8859-1')
ratings = pd.read_csv('ml-1m/ratings.dat',
                      header=None,
                      delimiter='::',
                      engine='python',
                      names='UserID::MovieID::Rating::Timestamp'.split('::'))

movies = pd.merge(movies, pd.merge(users, ratings))
print(movies, end='\n\n')

p1 = movies.pivot_table(values='Rating', index='Gender')
print(p1, end='\n\n')

# 퀴즈
# 남녀 평점 평균을 구하세요
print(movies.values.shape)

# m_sum, m_cnt = 0, 0
# for i in range(movies.shape[0]):
#     s = movies.iloc[i]
#     # print(s['Gender'], s['Rating'])
#
#     if s['Gender'] == 'M':
#         m_sum += s['Rating']
#         m_cnt += 1
#
# print('남자 :', m_sum / m_cnt)

# m_sum, m_cnt = 0, 0
# males = []
# for g, r in zip(movies['Gender'], movies['Rating']):
#     if g == 'M':
#         m_sum += r
#         m_cnt += 1
#         males.append(r)
#
# print('남자 :', m_sum / m_cnt, sum(males) / len(males))

b_males = (movies['Gender'] == 'M')
print(b_males)

males = movies['Rating'][b_males]
print(type(males))
print(males)
print('남자 :', np.mean(males))

p2 = movies.pivot_table(values='Rating', index='Gender', columns='Age')
print(p2, end='\n\n')

p3 = movies.pivot_table(values='Rating', index='Age', columns='Gender')
print(p3, end='\n\n')

p4 = movies.pivot_table(values='Rating', index='Age', columns='Occupation')
print(p4, end='\n\n')

p5 = movies.pivot_table(values='Rating', index=['Age', 'Gender'], columns='Occupation')
print(p5, end='\n\n')
