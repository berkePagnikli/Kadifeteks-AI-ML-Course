from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory  # Son mesajları tutup daha eskiyi özetleyen hibrit bellek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda  # Fonksiyonu runnable pipeline elemanına dönüştürür

from utils import print_memory_state, get_model_name  # Yardımcı fonksiyonlar


def build_chain():  # Zincir kurulum fonksiyonu (özet + buffer yaklaşımı)
	load_dotenv()  # .env yükle
	llm = ChatGoogleGenerativeAI(  # LLM örneği
		model=get_model_name(),
		temperature=0.2
	)
	memory = ConversationSummaryBufferMemory(  # Hem özet hem pencere mantığını kombine eder
		llm=llm,  # Özet üretimi aynı modelle yapılır
		max_token_limit=200,  # Bellekte tutulacak toplam token sınırı (aşınca özetleme tetiklenir)
		return_messages=True  # history'yi Message objeleri listesi olarak isteriz
	)
	prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"Sen yardımcı bir Türkçe sohbet asistansın. Kullanıcıyla olan geçmiş konuşmalarınızın özetine ulaşabilirsin.",
		),
		MessagesPlaceholder(variable_name="history"),  # memory tarafından mühendis edilmiş (özet + son mesajlar) birleşik geçmiş
		("human", "{input}"),
	])

	def enrich(inputs: dict):  # Bellekten history & gerekiyorsa summary bilgilerini çeker
		vars_ = memory.load_memory_variables({"input": inputs.get("input", "")})
		return {**inputs, **vars_}

	chain = RunnableLambda(enrich) | prompt | llm  # Pipeline
	return chain, memory

def main():
	chain, memory = build_chain()
	print("ConversationSummaryBufferMemory ajanına hoş geldiniz. Çıkmak için 'exit' yazın.")
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

