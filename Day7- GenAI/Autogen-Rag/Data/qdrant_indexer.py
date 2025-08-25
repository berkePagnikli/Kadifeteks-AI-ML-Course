from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from data import (urban_ag_data, urban_crop_data, urban_challenges, urban_best_practices)

model = SentenceTransformer('all-MiniLM-L6-v2')

client = QdrantClient(host="localhost", port=6333)

def create_collection(collection_name: str, vector_size: int = 384) -> None:
    """Qdrant'ta koleksiyon yoksa oluşturur, varsa bilgi mesajı yazdırır.

    Parametreler:
        collection_name (str): Oluşturulacak veya kontrol edilecek koleksiyonun adı.
        vector_size (int): Vektör boyutu (kullanılan embedding modelinin çıkışıyla aynı olmalı).
    """
    try:
        # get_collection koleksiyon varsa metadata döndürür; yoksa hata fırlatır.
        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        # Koleksiyon yoksa burada yeni bir koleksiyon yaratıyoruz.
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,               # Embedding boyutu (model = 384)
                distance=models.Distance.COSINE  # Benzerlik ölçütü (cosine benzerliği)
            )
        )
        print(f"Created collection {collection_name}")

# Aşağıdaki çağrılar ile dört ayrı koleksiyon oluşturulmuş (veya var olduğu teyit edilmiş) olur.
create_collection("urban_ag_methods")
create_collection("urban_crops")
create_collection("urban_challenges")
create_collection("urban_best_practices")

def create_document_text(data: Dict[str, Any], data_type: str) -> str:
    """Veri sözlüğünü (dict) gömme için anlamlı, tutarlı bir düz metne çevirir.

    Neden gerekli? Vektör aramada başarı, dokümanların semantik içeriğini özetleyip anlamlı
    bir metin haline getirmeye dayanır. Bu fonksiyon veri alanlarını birleştirerek modelin
    bağlamı daha iyi kavramasını sağlar.

    Parametreler:
        data (Dict[str, Any]): Tek bir öğeye ait ham veri sözlüğü.
        data_type (str): Hangi tür koleksiyona ait olduğunu belirtir; biçimlendirme koşullu yapılır.

    Dönüş:
        str: Embedding modeline verilecek tanımlayıcı metin.
    """
    if data_type == "urban_ag_methods":
        # Yöntem (method) temelli verilerin anlatımı.
        text = f"{data['description']} "
        text += f"Suitable crops: {', '.join(data['suitable_crops'])}. "
        text += f"Space efficiency: {data['space_efficiency']}, "
        text += f"Water usage: {data['water_usage']}, "
        text += f"Energy requirement: {data['energy_requirement']}. "
        text += f"Advantages: {', '.join(data['advantages'])}. "
        text += f"Disadvantages: {', '.join(data['disadvantages'])}. "
        text += f"Ideal locations: {', '.join(data['ideal_locations'])}. "
        text += f"Startup cost range: {data['startup_cost_range']}."
    
    elif data_type == "urban_crops":
        # Ürün (crop) temelli verilerin anlatımı.
        text = f"Crop with growth cycle of {data['growth_cycle_days']} days. "
        text += f"Yield per sqft: {data['yield_per_sqft']}, "
        text += f"Water needs: {data['water_needs']}, "
        text += f"Light needs: {data['light_needs']}, "
        text += f"Temperature range: {data['temperature_range'][0]} to {data['temperature_range'][1]} degrees. "
        text += f"Difficulty: {data['difficulty']}, Market value: {data['market_value']}. "
        text += f"Nutritional highlights: {data['nutritional_highlights']}. "
        text += f"Best growing methods: {', '.join(data['best_growing_methods'])}. "
        text += f"Companion plants: {', '.join(data['companion_plants'])}. "
        text += f"Seasonal notes: {data['seasonal_notes']}."
    
    elif data_type == "urban_challenges":
        # Zorluk / sorun (challenge) temelli verilerin anlatımı.
        text = f"{data['description']}. "
        text += f"Solutions: {', '.join(data['solutions'])}. "
        text += f"Impact score: {data['impact_score']}. "
        text += f"Detailed strategies: {'. '.join(data['detailed_strategies'])}. "
        text += f"Recommended resources: {', '.join(data['recommended_resources'])}."
    
    elif data_type == "urban_best_practices":
        # En iyi uygulamalar (best practices) temelli verilerin anlatımı.
        text = f"{data['title']}. "
        text += f"Practices: {'. '.join(data['practices'])}."
    
    return text  # Üretilen metin embedding modeline girdi olacak.


def index_data(data_dict: Dict[str, Dict[str, Any]], collection_name: str, data_type: str) -> None:
    # points: Upsert edeceğimiz PointStruct nesnelerini toplamak için liste.
    points = []
    
    # enumerate: i = ardışık sayısal id (Qdrant iç id), key = veri sözlük anahtarı, value = veri içerik dict'i.
    for i, (key, value) in enumerate(data_dict.items()):
        # Her öğe için açıklayıcı metni üret.
        text = create_document_text(value, data_type)
        
        # Metni embedding modelinden geçirip list'e çeviriyoruz.
        embedding = model.encode(text).tolist()

        # payload: Noktaya iliştirilecek ek bilgiler.
        payload = {
            "id": key,                 # Orijinal veri id'si (dış referans için yararlı)
            "data": value,             # Ham veri sözlüğü (isteğe göre query sonrası gösterilebilir)
            "data_type": data_type,    # Veri kategorisi (koleksiyon tip kontrolü / filtreleme)
            "text_description": text   # Sorgu sırasında gösterim veya inceleme için açıklama metni
        }
        
        # PointStruct: Qdrant'a gönderilecek tekil nokta.
        point = models.PointStruct(
            id=i,              # Qdrant iç id (benzersiz olmalı). enumerate ile deterministik sıra.
            vector=embedding,  # 384 boyutlu embedding vektörü.
            payload=payload    # Ek açıklayıcı alanlar.
        )
        
        # Noktayı toplu listeye ekliyoruz (tek seferde yüklemek için performans açısından daha iyi).
        points.append(point)
    
    # client.upsert: points listesindeki tüm noktaları koleksiyona ekler veya günceller (id çakışması varsa overwrite).
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Indexed {len(points)} items in {collection_name}")

# Aşağıdaki çağrılar dört farklı veri kümesini ilgili koleksiyonlara indeksler.
index_data(urban_ag_data, "urban_ag_methods", "urban_ag_methods")
index_data(urban_crop_data, "urban_crops", "urban_crops")
index_data(urban_challenges, "urban_challenges", "urban_challenges")
index_data(urban_best_practices, "urban_best_practices", "urban_best_practices")