from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

# Sistem promptu
SYSTEM_MESSAGE = """
# Genel Bakış
    - Sen tekstil alanında çalışan hammadde bilgisi yüksek bir ajansın. Sağlanan bağlamı kullanarak kullanıcıya istediği sonucu ver.

# Yönergeler
    - Kullanıcı sorgusunu şu formatta alacaksın: "User: {{query}}" ve bağlamı şu formatta: "Context: {{context}}". Context; belgelerin ve onların metadatalarının bulunduğu bir listedir. Bağlamı sanki sen veri tabanından almışsın gibi değerlendir.
    - Bağlam ve kullanıcı sorusunu değerlendirirken zamana bağlı değişimleri dikkate al.
    - Vardığın sonuca neden vardığını madde madde açıkla.
"""

# Chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("user", "{input}")
])

# Chain oluştur
hammadde_chain = prompt | llm | StrOutputParser()

class HammaddeAjan:
    """Hammadde Ajanı sınıfı"""
    
    def __init__(self):
        self.name = "hammadde_ajan"
        self.chain = hammadde_chain
    
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
hammadde_ajan = HammaddeAjan()