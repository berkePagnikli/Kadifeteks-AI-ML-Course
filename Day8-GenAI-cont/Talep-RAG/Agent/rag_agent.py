
from qdrant_client import QdrantClient  
from autogen import AssistantAgent, config_list_from_json 
from sentence_transformers import SentenceTransformer 
from typing import List, Dict

# ---------------------------------------------------------------
# Embedding modeli (MiniLM) hafif ve hızlı olduğu için seçildi.
# 'embedding_model' tek örnek: her sorguda tekrar yüklenmez -> performans.
# ---------------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------------
# Qdrant istemci: Lokal host'ta 6333 portu varsayılandır (Docker çalışıyor varsayımı).
# ---------------------------------------------------------------
client = QdrantClient(host="localhost", port=6333)

# ---------------------------------------------------------------
# config_list_from_json: OpenAI / LLM erişim yapılandırmalarını JSON dosyasından çeker.
# "OAI_CONFIG_LIST" dosya adı root klasörde mevcut olmalı.
# ---------------------------------------------------------------
config_list = config_list_from_json("OAI_CONFIG_LIST")

# ---------------------------------------------------------------
# retrieve_textile_data: Bir kullanıcı sorgusunu embedding'e çevirip birden
# fazla koleksiyonda (perspektifte) yakın komşu araması yapar. Sonuçları skor'a göre sıralar.
# Parametreler:
#   query: Kullanıcının doğal dil sorusu.
#   collection_names: Arama yapılacak koleksiyon listesi.
#   top_k: Her koleksiyondan kaç sonuç çekilecek (limit).
# Dönen: Skorlarına göre sıralanmış belge listesi (içerik + metadata).
# ---------------------------------------------------------------
def retrieve_textile_data(query: str, collection_names: List[str], top_k: int = 24) -> List[Dict]:
    query_embedding = embedding_model.encode(query).tolist()  # Sorgu için embedding vektörü
    
    results = []  # Tüm koleksiyonlardan toplanan sonuçların birleşik listesi
    for collection in collection_names:  # Her koleksiyonu sırayla ara
        hits = client.query_points(
            collection_name=collection,  # Hangi koleksiyonda arama yapılacak
            query=query_embedding,       # Arama vektörü
            limit=top_k,                 # O koleksiyondan üst top_k sonuç
            with_payload=True            # Orijinal metadata/payload da gelsin
        ).points  # Qdrant yanıtındaki point listesi
        for hit in hits:  # Her hit bir candidate belge
            results.append({
                "content": hit.payload["text_description"],  # Embedding'e esas olan açıklayıcı metin
                "metadata": {
                    "type": hit.payload["data_type"],        # Hangi perspektif (koleksiyon türü)
                    "id": hit.payload["id"],                # Orijinal ID (indexleme sırasında atanmış)
                    "score": hit.score,                      # Benzerlik skoru (daha yüksek = daha ilgili)
                }
            })
    
    # Tüm sonuçlar tek listede => skora göre azalan sırala.
    return sorted(results, key=lambda x: x["metadata"]["score"], reverse=True)


# ---------------------------------------------------------------
# AssistantAgent: AutoGen kütüphanesinin LLM ile etkileşim arayüzü.
# system_message: LLM'e davranış ve bağlam kullanımı kurallarını anlatır.
# llm_config: Hangi LLM sağlayıcı/ayarlarının kullanılacağı.
# name: Aracının kimliği (diyalog yönlendirme için).
# ---------------------------------------------------------------
assistant = AssistantAgent(
    name="assistant",  # Agent adı
    system_message="""
    # Genel Bakış
        - Sen tekstil üretimi ve kumaş imalatı konusunda uzmansın. Soruları yanıtlamak için sağlanan bağlamı kullan.

    # Yönergeler
        - Kullanıcı sorgusunu şu formatta alacaksın: "User: {query}" ve bağlamı şu formatta: "Context: {context}". Context; belgelerin ve onların metadatalarının bulunduğu bir listedir. Bağlamı sanki sen veri tabanından almışsın gibi değerlendir.
        - Odak noktan zaman bağlı ihtiyaç değişimleri. Burada ay, yıl, ihtiyaç ve hammadde kodu özelliklerinden faydalanarak kullanıcının istediği hesaplamaları yerine getir.
        - SARIMA algoritmasını kullanarak tahminleme hesaplamalarını yürüt, süreci madde madde açıkla.
        - Tahminleme doğruluğunu MAPE ile doğrula.
    """,
    llm_config={
        "config_list": config_list,  # JSON'dan gelen sağlayıcı ayarları listesi
        "timeout": 600,              # Uzun söyleşi / yavaş yanıt durumları için 600 sn timeout
    }
)

# ---------------------------------------------------------------
# runAgent: Dış dünyadan (örneğin CLI) bir soru alır; retrieval ve LLM yanıtını çalıştırır.
# Adımlar:
#   1) Koleksiyon listesi tanımla
#   2) Sorgu ile ilgili benzer belgeleri çek
#   3) Bu belgeleri okunabilir bir bağlam stringine dönüştür
#   4) LLM'e 'user' rolünde formatlanmış mesaj gönder
#   5) Yanıtı ekrana yazdır
# ---------------------------------------------------------------
def runAgent(question: str):
    collections = ["textile_materials"]  # Aranacak koleksiyonlar
    context_data = retrieve_textile_data(question, collections)  # Retrieval sonuçları
    context_str = "\n".join([
        f"Document {i+1} ({doc['metadata']['type']}, score: {doc['metadata']['score']:.2f}):\n{doc['content']}"
        for i, doc in enumerate(context_data)
    ])  # Her belgeyi satır satır birleştir -> LLM'e bağlam sunumu
    print("Context retrieved:")  # Kullanıcıya hangi bağlamın kullanıldığını göster
    print(context_str)
    agent_input = (f"User:\n{question}\n\n"  # LLM için giriş formatı: önce kullanıcı sorusu
                   f"Context:\n{context_str}")  # Ardından bağlam blok halinde
    messages = [
        {"role": "user", "content": agent_input}  # AutoGen formatında mesaj listesi
    ]

    response = assistant.generate_reply(messages)  # LLM cevabını üret

    print("Agent Response:")  # Konsola çıktı
    print(response)

# ---------------------------------------------------------------
# Script doğrudan çalıştırıldığında örnek bir soru ile ajanı test eder.
# ---------------------------------------------------------------
runAgent("NPE 1000 449 hammadde kodunun 2025 yılında ilk ay aylık ihtiyaç kg'si toplamda ne kadar olacaktır? Hesapla.")