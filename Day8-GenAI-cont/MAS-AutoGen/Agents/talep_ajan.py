from qdrant_client import QdrantClient  
from autogen import AssistantAgent, config_list_from_json

config_list = config_list_from_json("OAI_CONFIG_LIST")

talep_ajan = AssistantAgent(
    name="talep_ajan",
    system_message="""
    # Genel Bakış
        - Sen tekstil üretimi ve kumaş imalatı konusunda uzmansın. Soruları yanıtlamak için sağlanan bağlamı kullan.

    # Yönergeler
        - Kullanıcı sorgusunu şu formatta alacaksın: "User: {query}" ve bağlamı şu formatta: "Context: {context}". Context; belgelerin ve onların metadatalarının bulunduğu bir listedir. Bağlamı sanki sen veri tabanından almışsın gibi değerlendir.
        - Odak noktan zaman bağlı ihtiyaç değişimleri. Burada ay, yıl, ihtiyaç ve hammadde kodu özelliklerinden faydalanarak kullanıcının istediği hesaplamaları yerine getir.
        - Hesaplama sonuçlarını paylaşırken en iyi (yüksek ihtiyaç), en kötü (düşük ihtiyaç) ve normal yaptığın hesaplamayı paylaş, madde madde açıkla.
    """,
    llm_config={
        "config_list": config_list, 
        "timeout": 600, 
    }
)