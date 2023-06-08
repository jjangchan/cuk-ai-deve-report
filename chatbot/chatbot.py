import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        input_vector = self.vectorizer.transform([input_sentence])
        similarities = cosine_similarity(input_vector, self.question_vectors) # 코사인 유사도 값들을 저장
        
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index]

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
    
