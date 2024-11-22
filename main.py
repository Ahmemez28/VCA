from transformers import pipeline

emotionAnalyzer=pipeline("text-classification",model="j-hartmann/emotion-english-distilroberta-base")

sentence="I hate everyone because everyone is stupid"

words =sentence.split(" ")

wordResults={}
for word in words:
	result=emotionAnalyzer(word)
	print(f"Word: {word}, Emotion: {result[0]['label']}, Score: {result[0]['score']:.2f}")
	wordResults[word]=result[0]['label']
result=emotionAnalyzer(sentence)
print(f"Whole Sentence \n Emotion: {result[0]['label']}, Score: {result[0]['score']:.2f}")
colouredText=[]
for i in wordResults:
    if wordResults[i]==result[0]['label']:
        colouredText.append(i)
print(colouredText)
