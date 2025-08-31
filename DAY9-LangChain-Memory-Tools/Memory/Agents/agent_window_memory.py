from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.memory import ConversationBufferWindowMemory  # Yalnızca son k etkileşimi tutan kaydırmalı pencere bellek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from utils import print_memory_state, get_model_name  # Yardımcı fonksiyonlar


def build_chain():  # Son k mesajı referans alan zincir kurar
	load_dotenv()  # API anahtarlarını yükle
	llm = ChatGoogleGenerativeAI(  # LLM nesnesi
		model=get_model_name(),
		temperature=0.2
	)
	memory = ConversationBufferWindowMemory(  # Sliding window bellek
		k=3,  # Kaç son mesaj çiftini (insan + asistan) tutacağını belirler
		return_messages=True  # history'yi Message listesi olarak almak için
	)
	prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"Sen yardımcı bir Türkçe sohbet asistansın. Sadece son k mesajı görüyorsun.",  # Modeli sınırlı bağlam konusunda uyarır
		),
		MessagesPlaceholder(variable_name="history"),  # memory'den gelen son k etkileşim
		("human", "{input}"),
	])

	def enrich(inputs: dict):  # Pipe başlangıcı: history ekler
		# Window memory load_memory_variables -> {'history': [...]} döner
		vars_ = memory.load_memory_variables({})
		return {**inputs, **vars_}

	chain = RunnableLambda(enrich) | prompt | llm  # Pipeline tanımı
	return chain, memory


def main():
	chain, memory = build_chain()
	print("ConversationBufferWindowMemory ajanına hoş geldiniz. Çıkmak için 'exit' yazın.")
	while True:
		user_input = input("Sen: ").strip()
		if user_input.lower() in {"exit", "quit", "q"}:
			print("Çıkılıyor...")
			break
		if not user_input:
			continue
		response = chain.invoke({"input": user_input})
		answer = response.content if hasattr(response, "content") else str(response)
		print(f"Ajan: {answer}\n")
		memory.save_context({"input": user_input}, {"response": answer})
		print_memory_state(memory)


if __name__ == "__main__":
	main()

