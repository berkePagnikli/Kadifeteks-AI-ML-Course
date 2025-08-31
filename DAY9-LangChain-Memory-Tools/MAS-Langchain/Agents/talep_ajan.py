from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# Sistem promptu
SYSTEM_MESSAGE = """
# Genel Bakış
    - Sen tekstil üretimi ve kumaş imalatı konusunda uzmansın. Soruları yanıtlamak için sağlanan bağlamı kullan.

# Yönergeler
    - Kullanıcı sorgusunu şu formatta alacaksın: "User: {{query}}" ve bağlamı şu formatta: "Context: {{context}}". Context; belgelerin ve onların metadatalarının bulunduğu bir listedir. Bağlamı sanki sen veri tabanından almışsın gibi değerlendir.
    - Odak noktan zaman bağlı ihtiyaç değişimleri. Burada ay, yıl, ihtiyaç ve hammadde kodu özelliklerinden faydalanarak kullanıcının istediği hesaplamaları yerine getir.
    - Hesaplama sonuçlarını paylaşırken en iyi (yüksek ihtiyaç), en kötü (düşük ihtiyaç) ve normal yaptığın hesaplamayı paylaş, madde madde açıkla.
"""

# Chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("user", "{input}")
])

# Chain oluştur
talep_chain = prompt | llm | StrOutputParser()

class TalepAjan:
    """Talep Ajanı sınıfı"""
    
    def __init__(self):
        self.name = "talep_ajan"
        self.chain = talep_chain
    
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
talep_ajan = TalepAjan()