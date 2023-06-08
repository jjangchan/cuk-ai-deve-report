import pandas as pd

# sklearn라는 머신러닝 라이브러리에서 TfidfVectorizer와 cosine_similarity를 불러옴
# TfidfVectorizer는 문서의 텍스트 데이터를 벡터 형태로 변환하는데 사용하며, cosine_similarity는 두 벡터 간의 코사인 유사도를 계산
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleChatBot:
    # 챗북 클래스 함수, 클래스 전역 변수 설정
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.questions_len = len(self.questions) # 학습데이터 질문 길이 저장

    # ChatbotData.csv 파일 읽어와서 질문, 답변 열 뽑아서 파이썬 리스트로 저장하는 함수
    def load_data(self, filepath):
        data = pd.read_csv(filepath)    # ChatbotData.csv 파일 읽어오기
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()    # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    # chat 질문과 학습데이터 모든 질문에 레벤슈타인 거리 유사도 분석하고,
    # 제일 유사한 거리에 해당하는 학습데이터 질문에 맵핑된 답변 리턴하는 함수
    def find_best_answer(self, input_question):
        if input_question == "": return "올바른 질문을 입력해주세요."    # 질문이 공백이므로 레벤슈타인 거리 수행하지 않고 리턴
        lds = [] # 학습데이터 질문들과 챗봇 질문에 모든 레벤슈타인 거리를 담는 함수


        # for 문을 돌면서, 모든 학습데이터 질문과 챗봇 질문에 레벤슈타인 거리를 구한다.
        for i in range(0, self.questions_len):
            ds = self.calc_distance(self.questions[i], input_question)
            if ds == -1:
                # 챗봇 질문과 학습데이터에 동일한 질문이 있으므로,
                # 레벤슈타인 거리를 더이상 수행하지 않고 해당 질문에 맵핑된 답변 리턴
                return self.answers[i]
            lds.append(ds)

        # 레벤슈타인 거리에 제일 작은값에 인덱스 추출(유사도가 제일 높아서) 후,
        # 해당 질문에 맵핑된 답변 리턴

        best_match_index = lds.index(min(lds))
        #self.check_same_ld(lds, input_question)
        return self.answers[best_match_index]

    # 유사도가 제일 높은 레벤슈타인 거리가 같은 학습데이터 Q/A 출력 하는 함수
    def check_same_ld(self, lds, input_question):
        # [한계점]
        # 챗봇에서 입력한 "블록체인이" 에 유사도가 같은 학습데이터 질문들을 조회해본 결과,
        # 직관적으로 보면 "블록체인이 뭐야?"가 맞는거 같지만,
        # list.index() 함수는 같은 값중에 제일 가까운 index를 뽑아서 한계점이 뚜렷하게 드러난다.
        # 따라서, 같은 거리를 또다른 방법론으로 판별할 필요가 있어 보인다.
        best_score = min(lds)
        print("[in] %s" %(input_question))

        same_lds_question = {}
        for i in range(0, self.questions_len):
            if lds[i] == best_score:
                same_lds_question[self.questions[i]] = self.answers[i]
                print("(%d), [Q] %s  [A] %s" %(i,self.questions[i], self.answers[i]))

        # 같은 레벤슈타인 거리끼리 TF-IDF 벡터화 방식 해본 결과
        lds_keys = (list(same_lds_question.keys()))
        lds_vector = self.vectorizer.fit_transform(lds_keys)
        input_vector = self.vectorizer.transform([input_question])
        similarities = cosine_similarity(input_vector, lds_vector)
        best_match_index = similarities.argmax()
        print()
        print("[TF-IDF] ")
        print("[Q] %s [A] %s " %(lds_keys[best_match_index], same_lds_question[lds_keys[best_match_index]]))



    # 레벤슈타인 거리 구하기
    def calc_distance(self, data_question, input_question):
        ''' 레벤슈타인 거리 계산하기 '''
        # print(a," : ",b)
        if data_question == input_question: return -1  # 같으면 -1을 반환
        data_len = len(data_question)  # a 길이
        input_len = len(input_question)  # b 길이
        # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
        # matrix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0]]
        # [0, 1, 2, 3]
        # [1, 0, 0, 0]
        # [2, 0, 0, 0]
        # [3, 0, 0, 0]
        matrix = [[] for i in range(data_len+ 1)]  # 리스트 컴프리헨션을 사용하여 학습데이터 길이 만큼 행 개수 초기화
        for i in range(data_len + 1):  # 0으로 초기화
            matrix[i] = [0 for j in range(input_len+ 1)]  # 리스트 컴프리헨션을 사용하여 입력받은 챗봇 데이터 길이 만큼 열 개수 초기화
        # 0일 때 초깃값을 설정
        for i in range(data_len + 1):
            matrix[i][0] = i
        for j in range(input_len+ 1):
            matrix[0][j] = j
        # 표 채우기 --- (※2)
        for i in range(1, data_len+ 1):
            ac = data_question[i - 1]
            #print(ac, '=============')
            for j in range(1, input_len+ 1):
                bc = input_question[j - 1]
                #print(bc)
                cost = 0 if (ac == bc) else 1  # 파이썬 조건 표현식 예:) result = value1 if condition else value2
                #print("Cost : ", cost)
                matrix[i][j] = min([
                    matrix[i - 1][j] + 1,  # 문자 제거: 위쪽에서 +1
                    matrix[i][j - 1] + 1,  # 문자 삽입: 왼쪽 수에서 +1
                    matrix[i - 1][j - 1] + cost  # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
                ])
                # print(matrix)
        #if data_question == "관절염인가":
            #for i in range(1, data_len+1) :
                #print(matrix[i][1:])
        return matrix[data_len][input_len]

# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)
    
