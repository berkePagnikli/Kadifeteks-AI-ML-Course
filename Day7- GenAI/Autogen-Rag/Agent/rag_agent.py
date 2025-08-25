from qdrant_client import QdrantClient
from autogen import AssistantAgent, config_list_from_json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# embedding_model: Doğal dil sorgu ve dokümanları aynı vektör uzayına yerleştirmek için kullanılacak model.
# Seçilen model 'all-MiniLM-L6-v2' hafif, hızlı ve 384 boyutlu gömme üretiyor (index tarafında da aynı kullanıldı).
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# client: Lokal Qdrant sunucusuna bağlanmak için istemci. Host ve port Qdrant varsayılan çalıştırma parametreleri.
client = QdrantClient(host="localhost", port=6333)

# config_list: LLM erişim yapılandırmalarını (örneğin API anahtarları / endpoint'ler) içeren json dosyasından okunur.
config_list = config_list_from_json("OAI_CONFIG_LIST")

def retrieve_urban_data(query: str, collection_names: List[str], top_k: int = 3) -> List[Dict]:
    """Verilen doğal dil sorgusuna göre birden çok Qdrant koleksiyonundan en alakalı noktaları getirir.

    Parametreler:
        query (str): Kullanıcının sorduğu soru veya bilgi ihtiyacı.
        collection_names (List[str]): Arama yapılacak Qdrant koleksiyonlarının isim listesi.
        top_k (int): Her koleksiyondan alınacak maksimum sonuç sayısı (ilk kaba filtre).

    Dönüş:
        List[Dict]: Her biri 'content' (metin) ve 'metadata' (tip, id, skor) içeren sonuç listesi.
    """
    # Sorgu cümlesini gömme (embedding) vektörüne dönüştürüyoruz. encode -> numpy array, tolist -> JSON uyumlu Python list.
    query_embedding = embedding_model.encode(query).tolist()
    
    # results: Tüm koleksiyonlardan gelen aday sonuçları biriktirmek için boş liste.
    results = []
    # Her koleksiyonda aynı sorgu vektörü ile benzerlik araması yapılır.
    for collection in collection_names:
        # client.query_points: Belirtilen koleksiyonda vektör sorgusu yapar.
        #   collection_name: Hangi koleksiyon
        #   query: Sorgu vektörü
        #   limit: Döndürülecek en fazla sonuç
        #   with_payload: True -> Noktaların payload (ek veri) içeriğini de getir
        hits = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        ).points  # .points: Cevaptaki asıl sonuç listesi
        # Her hit (nokta) için kullanıcıya yansıtacağımız biçimlendirilmiş yapı hazırlıyoruz.
        for hit in hits:
            results.append({
                # content: Hit'in anlamlı metinsel açıklaması (indexleme aşamasında 'text_description').
                "content": hit.payload["text_description"],
                # metadata: Tür (koleksiyon tipi), orijinal id ve benzerlik skoru.
                "metadata": {
                    "type": hit.payload["data_type"],
                    "id": hit.payload["id"],
                    "score": hit.score  # Qdrant COSINE benzerlik skoru (daha yüksek genelde daha alakalı)
                }
            })
    
    # Tüm koleksiyonlardan gelen sonuçları skora göre azalan sıraya sokuyoruz.
    # Ardından ilk (top_k * 2) sonuçla sınırlandırıyoruz: Koleksiyon başına top_k yerine birleşik listeden daha zengin kesit.
    return sorted(results, key=lambda x: x["metadata"]["score"], reverse=True)[:top_k*2]

# assistant: LLM ajanı. system_message ile rol, ton ve çıktı kuralları belirlenir.
assistant = AssistantAgent(
    name="assistant",  # Ajanın dahili ismi (oturum kimliği gibi düşünülebilir)
    system_message="""
    # Overview
        - You are an expert in urban agriculture. Use the provided context to answer questions.
    
    # Guideline
        - You will recieve user query in the format: "User: {query}" and context in the format: "Context: {context}". Context is a list of documents with their metadata. Treat context as if you retrieved it.
        - If multiple methods exist, compare them using their metrics.
        - Mention limitations from disadvantages when relevant.
        - Prioritize quantitative data (space efficiency, costs).
        - Do not output the retrieved data directly. Analyze it and present a comprehensive explanation and a summary.
        - If unsure, request clarification about specific urban context.
    """,
    llm_config={
        # config_list: Yukarıda json'dan okunan LLM erişim yapılandırmaları listesi.
        "config_list": config_list,
        # timeout: LLM cevabı için maksimum bekleme süresi (saniye cinsinden).
        "timeout": 600,
    }
)

def runAgent(question: str):
    """Dışarıdan verilen soruyu alır, ilgili belgeleri getirir, bağlamı oluşturur ve asistana iletir.

    Parametreler:
        question (str): Kullanıcı sorusu.
    """
    # collections: RAG için taranacak Qdrant koleksiyonlarının listesi.
    collections = ["urban_ag_methods", "urban_crops", "urban_challenges", "urban_best_practices"]
    # context_data: retrieve_urban_data fonksiyonundan dönen sonuç listesi.
    context_data = retrieve_urban_data(question, collections)
    # context_str: LLM'ye beslenecek biçimlendirilmiş metin (her doküman numaralandırılır ve skor gösterilir).
    context_str = "\n".join([
        f"Document {i+1} ({doc['metadata']['type']}, score: {doc['metadata']['score']:.2f}):\n{doc['content']}"
        for i, doc in enumerate(context_data)
    ])
    # Kullanıcıya konsoldan hangi bağlamın getirildiğini görmek için yazdırıyoruz.
    print("Context retrieved:")
    print(context_str)
    # agent_input: System prompt harici olarak LLM'ye kullanıcı rolü üzerinden iletilecek tam metin.
    agent_input = (f"User:\n{question}\n\n"
                   f"Context:\n{context_str}")
    # messages: Autogen arayüzünün beklediği biçimde mesaj listesi.
    messages = [
        {"role": "user", "content": agent_input}
    ]

    # assistant.generate_reply: LLM üzerinden uygun rol kısıtlarıyla yanıt üretir.
    response = assistant.generate_reply(messages)

    # Üretilen yanıtı konsola yazdırıyoruz.
    print("Agent Response:")
    print(response)

# Örnek çalışma: Belirli bir senaryo sorusu ile fonksiyonu çağırıyoruz.
runAgent("What's the most space-efficient method for growing strawberries in a polluted urban area?")