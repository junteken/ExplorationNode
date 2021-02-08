import os
import pandas as pd

# rating_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/ratings.dat'
# rating_file_path='/content/drive/MyDrive/DL_Study/AIFFEL/recommend/ml-1m/ratings.dat'
rating_file_path=os.getcwd()+'/data/ml-1m/ratings.dat'
# movie_file_path='/content/drive/MyDrive/DL_Study/AIFFEL/recommend/ml-1m/movies.dat'
movie_file_path=os.getcwd()+'/data/ml-1m/movies.dat'
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
orginal_data_size = len(ratings)
ratings.head()
# 3점 이상만 남깁니다.
ratings = ratings[ratings['rating']>=3]
filtered_data_size = len(ratings)

print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')
# rating 컬럼의 이름을 count로 바꿉니다.
ratings.rename(columns={'rating':'count'}, inplace=True)
# 영화 제목을 보기 위해 메타 데이터를 읽어옵니다.

cols = ['movie_id', 'title', 'genre']
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python',  encoding = "ISO-8859-1")
movies.head()

my_favorite = ['deep impact' , 'amagedon' ,'cast away' ,'matrix' ,'three rings']
movie_id_end = movies.tail(1)['movie_id'].item() # movie id의 마지막 id를 가져온다
my_favorite_id=[]
for i in range(len(my_favorite)):
    my_favorite_id.append(movie_id_end+i+1)


my_movielist = pd.DataFrame({'movie_id': my_favorite_id,
                             'title': my_favorite, 'genre': ['Sci-Fi']*5})

movies.append(my_movielist, ignore_index= True)
movies.tail()