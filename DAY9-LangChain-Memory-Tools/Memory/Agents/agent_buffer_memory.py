from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory  # Tüm geçmişi tutan bellek sınıfı
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda  # Zincire fonksiyon (lambda) eklemeye yarayan runnable adaptörü

from utils import print_memory_state, get_model_name  # Yardımcı fonksiyonlar


def build_chain():  # Ajanı (LLM + bellek + prompt akışı) kuran fonksiyon
	load_dotenv()
	llm = ChatGoogleGenerativeAI(  # LLM örneği oluşturulur
		model=get_model_name(),  # utils içindeki MODEL_NAME sabitinden okunur
		temperature=0.2  # Daha deterministik cevaplar için düşük sıcaklık
	)
	memory = ConversationBufferMemory(  # Tüm konuşmayı kesmeden tutan bellek
		return_messages=True  # Ham mesaj objeleri (Message) döndürsün; prompt içinde formatlamada esneklik sağlar
	)
	prompt = ChatPromptTemplate.from_messages([  # Mesaj tabanlı şablon oluşturma
		(
			"system",
			"Sen yardımcı bir Türkçe sohbet asistansın.",  # Modelin rol / davranış tanımı
		),
		MessagesPlaceholder(  # Bellekten gelen geçmiş mesajların yerleştirileceği placeholder
			variable_name="history"  # memory.load_memory_variables() çıktısındaki anahtar ile eşleşmeli
		),
		("human", "{input}"),  # Kullanıcının anlık girdisi şablona inject edilecek
	])

	def enrich(inputs: dict):  # Zincir başında çalışacak: bellek değişkenlerini input'a enjekte eder
		# ConversationBufferMemory, load_memory_variables({}) çağrısında {'history': [...mesajlar...]} döndürür
		vars_ = memory.load_memory_variables({})
		# Kullanıcıdan gelen mevcut input sözlüğünü (örn {'input': 'merhaba'}) bellek değişkenleriyle birleştirir
		return {**inputs, **vars_}

	# RunnableLambda(enrich) -> prompt -> llm sıralı pipe: 
	# 1) enrich: input + history
	# 2) prompt: mesaj listesi formatına çevirir
	# 3) llm: modeli çağırır
	chain = RunnableLambda(enrich) | prompt | llm
	return chain, memory  # Dışarıya hem zinciri hem belleği döndürür; ana döngü belleğe kaydetmek için kullanır


def main():
	chain, memory = build_chain()
	print("ConversationBufferMemory ajanına hoş geldiniz. Çıkmak için 'exit' yazın.")
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

