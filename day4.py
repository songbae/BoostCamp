# 띄워쓰기 부분에 '-'를 부분에 추가 뱀처럼 늘여쓰기, 파이썬 함수 변수명에 사용
# CamelCase : 띄어쓰기 부분에 대문자 낙타의등 모양, 파이썬 class며에 사용

#Attribute 추가하기 
#__init__ 추가는 , self 와 함께 , init은 객체 초기화 예약함수 

class SoccerPlayer(object):
    def __init__(self,name,position, back_number):
        self.name=name
        self.position=position
        self.back_number=back_number
    def __add__(self,other):
        return self.name + other.name
abc= SoccerPlayer('son',7,10)
kein =SoccerPlayer('park',10,13)
# print(abc+kein)
# print(kein)
# kkk=SoccerPlayer()
class Note(object):
    def __init__(self,content=None):
        self.content=content
    def write_content(self,content):
        self.content=content
    def remove_all(self):
        self.content=""
    def __add__(self,other):
        return self.content +other.content
    def __str__(self):
        return self.content

class Notebook(object):
    def __init__(self,title):
        self.title=title
        self.page_number=1
        self.notes={}
    def add_note(self,note,page=0):
        if self.page_number <300:
            if page==0:
                self.notes[self.page_number]=note
                self.page_number+=1
            else:
                self.notes={page:note}
                self.page_number+=1
        else:
            print('Page가 모두 채워졋습니다')
    def remove_note(self,page_number):
        if page_number in self.notes.keys():
            return self.notes.pop(page_number)
        else:
            print("해당 페이지는 존재하지 않습니다")
    def get_number_of_pages(self):
        return len(self.notes.keys())



# 객체지향 모델링에 필요한 것들
# inheritance, Polymorphsim, Visibility

# 상속 - 부모클래스로 부터 속성과 method 를 물려받은 자식 클래스를 생성 하는것 
class person():
    def __init__(self, name,age):
        self.name=name
        self.age=age
    def __str__(self):
        return f'저의 이름은 {self}입니다'

# class employee(Person):
#     def __init__(self,name,age,gender,salary, hire_date):
#         super().__init__(name,age,gender) #부모객체 사용
#         self.salary=salary
#         self.hire_date=hire_date
#     def do_work(self):
#         print('열심히 일을 합시다')
#     def about_me(self):
#         super().about_me()# 부모 클래스 함수 사용 
#         print('제 급여는',self.salary,'원 이구요 제 입사일은',self.hire_date,'입니다')

# 다형성:
# 같은 이름 메소드의 내부 로직을 다르게 작성
# Dynamic Typing 특성으로 인해 파이썬에서는 같은 부모클래스의 상속에서 주로 발생함
# 중ㅇ한 OOP의개념 그러나 너무 깊이 알 필요 x

#가시성:
# 객체의 정보를 볼 수 있는 레벨을 조절 하는것 
# 누구나 객체 안헤 모든 변수를 볼 필요가 없다.
## 객체를 사용하는 사용자가 임읭로 정보 수정
## 필요 없는 정보에는 접근 할 필요가 없음
## 만약 제품으로 판매 한다면 ? 소스의 보호

##상식 Encapsulation
## 캡슗롸 또는 정보 은닉 
## class 를 설계할때 클래스 간 간섭/정보공유의 최소화
## 캡슐을 던지듯, 인터페이스만 

# class Inventory(object):
#     def __init__(self):
#         self __items=[]
#     @property # property decorator 숨겨진 변수를반환하게 해줌 
#     def items(self):
#         return self.__items

# my_inventory = Inventory()
# my_inventory.add_new_item(Product())


# decorater
#first class pbject:
    # 일등함수 또는 일급객체
    # 변수나 데이터 구조에 할당이 가능한 객체
    # 파라메터로 전달이 가능 + 리턴값으로 사용 
#inner function
# 함수내에 또다른 함수가 존재 
#-closure 이너펑션을 리턴값으로 반환 

def star(func):
    def inner(*args, **kwargs):
        print('*'*30)
        func(*args,**kwargs)
        print('*'*30)
    return inner

@star
def printer(msg):
    print(msg)
# printer('hello')



# Module and project

# 프로그램에서는 작은 프로그램 조각들, 모듈들을 모아서 하나의 크 프로그램
# 패키지 모듈을 모아놓은 단위, 하나의 프로그램 
# import 할시 pycache가 생기는데 미리 실행하기 쉽도록 컴파일된 상태 (메모리에 올려놓은 상태이다)

# namespace 
## 모듈을 호출할때 범위 정하는 방법
## 모듈 안에는 함수와 클래스 등이 존개 가능
## 필요한 내용만 골라서 호출 할 수있음
## from 과 import 키워드를 사용함 

##package 만들기
## 폴더별로 __init__.py 구성하기
## 현재폴더가 패키지임을 알리는 초기화 스크립트
## 없을 경우 패키지로 간주하지않음 (3.3 부터는 x)
## 하위 폴더와 py 모듈을 모두 포함함
## import와 __all__keyword 사용한다

# python virtual Environment Overview
## 프로젝트 진행시 필요한 패키지만 설치하는 환경
## 기본 인터프리터 + 프로젝트 종류별 패키지 설정
## 웹 프로젝트 데이터 분석 프로젝트
 ### 각각 패키지 관리 할 수 있는 기능 
 ### 다양한 패키지 관리 도구를 사용함

 # 대표적인 도구 vitualenv 와 conda가 있음
## 