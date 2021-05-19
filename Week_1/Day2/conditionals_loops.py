# 파이썬은 -5~256까지는 메모리주소를 정적으로 미리 정해놨기 떄문에 
# 그 안에서는 is or == 으로 비교해도 True 가 반환되지만 아니라면 Flase가 반환된다.

#상항연산자
values =12
is_even =True if values%2==0 else False
print(is_even)
# 코드를 따로 디버깅 하기 위해서
 if __name__=='__main__':
     main()
#이런식으로 main문을 만들어서 따로 실행시켜줄수있다.