from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# Sistem promptu
SYSTEM_MESSAGE = """
# Genel Bakış
    - Sen tekstil alanında çalışan bir yönetici ajansın. Altında 3 ayrı ajan çalışmakta. Bu ajanlar ve uzmanlık alanları:
        - Hammadde Ajanı: Hammaddelerin kumaşlardaki kullanımı.
        - Talep Ajanı: Hammadde ihtiyaçları.
        - Ülke Ajanı: İhtiyaçların ülkelere göre dağılımı.

    - Sana sorulan soruyu analiz et, hangi alt ajanının bu alanda yetkin olduğuna karar ver.
    - Aynı zamanda, kullanıcının sorusuna göre anahtar kelimeler çıkartmalısın. Anahtar kelime çıkartmak için örneğe bakabilirsin.

# Örnek İletişim:
    
    - User: NPE 1000 449'a 2025 yılında ilk ay toplamda ne kadar ihtiyaç duyulacak
    - You: {{Ajan: <Talep Ajanı>}}, {{Keyword: <Hammadde Kodu: NPE 1000 449>}} 

    - User: RCQPPH 1100 008'a 2023'de ne kadar ihtiyaç duymuşum?
    - You: {{Ajan: <Talep Ajanı>}}, {{Keyword: <Hammadde Kodu: RCQPPH 1100 008 Tarih: 2023>}}
- Yalnızca tam olarak şu iki bloktan oluşan bir çıktı ver: {{Ajan: <...>}}, {{Keyword: <...>}}
- Keyword kısmında kullanıcı sorusunu cevaplamak için faydalı olacak alanları (Hammadde Kodu:, Tarih:, Firma Ülkesi:, Kumaş Kodu: vb.) anahtar-değer şeklinde boşlukla ayırarak yaz.
- Tarih alanını gireceğin zaman, ayları sayı olarak yaz. Örneğin Ocak 2024 değil 01/2024 şeklinde. Ayı zamanda, tarihi 2025 ve sonrası olan yıllar için keyword'e EKLEME YAPMA.
"""

# Chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("user", "{input}")
])

# Chain oluştur
merkez_chain = prompt | llm | StrOutputParser()

class MerkezAjan:
    """Merkez Ajanı sınıfı"""
    
    def __init__(self):
        self.name = "merkez_ajan"
        self.chain = merkez_chain
    
    def generate_reply(self, messages):
        if isinstance(messages, list) and len(messages) > 0:
            user_input = messages[0].get("content", "")
        else:
            user_input = str(messages)
        
        try:
            response = self.chain.invoke({"input": user_input})
            return response
        except Exception as e:
            return f"Hata oluştu: {str(e)}"

# Instance oluştur
merkez_ajan = MerkezAjan()