from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationEntityMemory  # Metinden kişi/yer/nesne varlıklarını çıkarıp izleyen bellek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables import RunnableLambda  # Fonksiyonu zincire dönüştüren adaptör

from utils import print_memory_state, get_model_name  # Yardımcı fonksiyonlar


def build_chain():  # Ajan kurulum fonksiyonu
	load_dotenv()  # Ortam değişkenlerini içeri alır
	llm = ChatGoogleGenerativeAI(  # LLM örneği oluşturulur
		model=get_model_name(),
		temperature=0.2
	)
	memory = ConversationEntityMemory(  # Konuşmadan varlıkları (entities) çıkarıp güncelleyen bellek
		llm=llm,  # Özetleme / çıkarım için aynı model kullanılabilir
		return_messages=True  # Geçmişi Message objeleri listesi olarak döndür
	)
	prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"Sen bir sohbet asistansın. Algılanan varlıklar adı altında kullanıcının sana verdiği bazı geçmiş bilgilere erişebiliyorsun.",
		),  # Rol & yetenek bildirimi
		MessagesPlaceholder(variable_name="history"),  # Geçmiş konuşma (mesaj listesi) buraya enjekte edilir
		(
			"system",
			"Algılanan varlıklar: {entities}",  # memory.load_memory_variables ile dönen 'entities' string'i yerleştirilir
		),
		("human", "{input}"),  # Kullanıcının güncel mesajı
	])

	def enrich(inputs: dict):  # Chain'in ilk adımı: bellekten dinamik alanları çeker
		# EntityMemory 'input' parametresine bakarak yeni varlık çıkarmak isteyebilir; bu yüzden input forward edilir
		mem_vars = memory.load_memory_variables({"input": inputs.get("input", "")})
		# Dönen dict tipik olarak {'history': [...], 'entities': '...'} içerir
		return {**inputs, **mem_vars}

	# enrich -> prompt -> llm sıralı pipe
	chain = RunnableLambda(enrich) | prompt | llm
	return chain, memory

def main():
	chain, memory = build_chain()
	print("ConversationEntityMemory ajanına hoş geldiniz. Çıkmak için 'exit' yazın.")
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

