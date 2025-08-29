from qdrant_client import QdrantClient  
from autogen import AssistantAgent, config_list_from_json

config_list = config_list_from_json("OAI_CONFIG_LIST")

ülke_ajan = AssistantAgent(
    name="ülke_ajan",
    system_message="""
    # Genel Bakış
        - Sen tekstil alanında çalışan ve tekstil üretiminde ülke dağılımlarına hakim bir ajansın. Sağlanan bağlamı da kullanarak istenilen sonucu ver.

    # Yönergeler
        - Kullanıcı sorgusunu şu formatta alacaksın: "User: {query}" ve bağlamı şu formatta: "Context: {context}". Context; belgelerin ve onların metadatalarının bulunduğu bir listedir. Bağlamı sanki sen veri tabanından almışsın gibi değerlendir.
        - Bağlam ve kullanıcı sorusunu değerlendirirken zamana bağlı değişimleri dikkate al.
        - Vardığın sonuca neden vardığını madde madde açıkla.
    """,
    llm_config={
        "config_list": config_list, 
        "timeout": 600, 
    }
)