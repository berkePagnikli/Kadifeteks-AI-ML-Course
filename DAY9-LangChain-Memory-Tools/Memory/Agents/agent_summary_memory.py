from dotenv import load_dotenv  
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.memory import ConversationSummaryMemory  # Konuşmayı özetleyerek hafızayı kısaltan bellek türü
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables import RunnableLambda

from utils import print_memory_state, get_model_name  # Yardımcı fonksiyonlar


def build_chain():  # Özet temelli bellek kullanan zincir kurulumu
	load_dotenv()  # Ortam değişkenlerini yükle
	llm = ChatGoogleGenerativeAI(  # Model örneği
		model=get_model_name(),
		temperature=0.2
	)
	memory = ConversationSummaryMemory(  # Konuşma büyüdükçe önceki mesajları özetleyen bellek
		llm=llm,  # Özet üretimi için kullanılacak LLM
		return_messages=True  # history'yi Message objeleri şeklinde döndür
	)
	prompt = ChatPromptTemplate.from_messages([
		(
			"system",
			"Sen yardımcı bir Türkçe sohbet asistansın. Geçmiş önce özetlenmiş olabilir.",  # Kullanıcıya bağlam: geçmiş tam olmayabilir
		),
		MessagesPlaceholder(variable_name="history"),  # memory'den dönen (özet + son mesajlar) birleşik geçmiş
		("human", "{input}"),  # Anlık kullanıcı girdisi
	])

	def enrich(inputs: dict):  # Prompt'a girmeden önce bellek değişkenlerini ekler
		# ConversationSummaryMemory 'input' parametresi verdiğimizde yeni özet gereksinimini değerlendirebilir
		vars_ = memory.load_memory_variables({"input": inputs.get("input", "")})
		return {**inputs, **vars_}

	chain = RunnableLambda(enrich) | prompt | llm  # Pipeline tanımı
	return chain, memory


def main():
	chain, memory = build_chain()
	print("ConversationSummaryMemory ajanına hoş geldiniz. Çıkmak için 'exit' yazın.")
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

