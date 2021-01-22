# f= open('count.txt', mode='w',encoding='utf8')
# for i in range(10):
#     data=f'{i}번째 줄입니다.\n'
#     f.write(data)
# f.close()

# # 파이썬의 데이터 다루기 
# import os
# os.mkdir('songbae')
# try:
#     os.mkdir('songbae')
# except FileExistsError as e:
#     print('이미 존재합니다')

# import shutil 
# source = 'i_have_a_dream.txt'
# dest = os.path.join('abc','songbae.txt')

# # 최근에는 pathlib 모듈을 사용하여 path객체를 다룸 
# import pathlib
# cwd =pathlib.Path.cwd()

# #log 파일 생성하기 
# # 디렉토리가 있는지, 파일이 있는지 확인후

# if not os.path.exists('log'):
#     os.mkdir('log')
# TARGET_FILE_PATH= os.path.join('log', 'count_log.txt')
# if not os.path.exists(TARGET_FILE_PATH):
#     f=open('log/count_log.txt','w',encoding='utf8')
#     f.write('기록이 시작됩니다\n')
#     f.close()

# with open(TARGET_FILE_PATH,'a',encoding='utf8') as f:
#     import random,datetime
#     for i in range(1,11):
#         stamp =str(datetime.datetime.now())
#         value =random.random()*1000000
#         log_line = stamp+'\t'+str(value)+'값이 생성되었습니다\n'
#         f.write(log_line)

# #Pickle
# # 파이썬의 객체를 영속화하는 built in 객체
# # 데이터 오브젝트 등 실행중 정보를 저장-> 불러와서 사용
# # 저장해야하는 정보, 계산 결과(모델) 등 활용이 많음
# import pickle
# f=open('list.pickle','wb')
# test=[1,2,3,4,5]
# pickle.dump(test,f)
# f.close()

# # 로그 남기기 -logging
# # 프로그램이 실행되는 동안 일어나는 정보를 기록을 남기기
# # 유저의 접근,프로그램의 Exception,특정 함수의 사용
# # Console 화면에 출력, 파일에 남기기, DB등에 남기기 등등
# # 기록된 로그를 분석하여 의미있는 결과를 도출 할 수있음 
# # 실행시점에서 남겨야 하는 기록, 개발시점에서 남겨야 하는 기록

# # 기록을 print로 남기는 것도 가능함
# # 그러나 console 창에만 남기는 기록은 분석시 사용불가
# # 때로는 개별별(개발,운영) 로 기록을 남길 필요도 있음
# # 이러한 기능을 체계적으로 지원하는 모듈이 필요함 

# import logging

# logging.debug('')# 개발시 처리 기록을 남겨야 하는 로그 정보를 남김
# logging.info(' ')# 처리가 진행되는 동안의 정보를 알림
# logging.waring('')#
# logging.error('')
# logging.critical('')

# if __name__=='__main__':
#     logger=logging.getLogger('main')
#     logging.basicConfig(level=logging.DEBUG)
#     logger.setLevel(logging.INFO)
# ## 처음 logging시 level default 값이 warning 으로 되어있기 때문에 setlevel을 통해서 변경이 필요하다

import re
import urllib.request
url = 'http://goo.gl/U7mSQl'
html=urllib.request.urlopen(url)
html_contents=str(html.read())
id_result=re.findall(r'([A-Za-z0-9]+[*]+)',html_contents)

for res in id_result:
    print(res)
    