from typing import Dict, Any, List, Optional, Tuple 
import pandas as pd 
from qdrant_client import QdrantClient  
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

# ---------------------------------------------------------------
# Embedding modeli (all-MiniLM-L6-v2) hafif ve hızlı olduğu için seçildi
# 'model' tüm fonksiyonlarda tekrar tekrar yüklememek için global tanımlandı.
# ---------------------------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------------
# Qdrant istemci nesnesi: localhost:6333 varsayılan Qdrant docker veya servis portu.
# Tek bir client üzerinden tüm CRUD ve sorgu işlemleri yürütülecek.
# ---------------------------------------------------------------
client = QdrantClient(host="localhost", port=6333)

# ---------------------------------------------------------------
# create_collection: Belirtilen isimde koleksiyon yoksa oluşturan yardımcı fonksiyon.
# Qdrant'ta koleksiyon bir tablo gibidir; vektörlerin saklandığı yapı.
# vector_size = 384 seçildi çünkü kullanılan model (MiniLM-L6-v2) 384 boyutlu embedding üretir.
# ---------------------------------------------------------------
def create_collection(collection_name: str, vector_size: int = 384) -> None:
    """Eğer yoksa Qdrant içinde bir koleksiyon oluşturur."""
    try:
        # Koleksiyon var mı diye sorgu; yoksa Qdrant bir hata fırlatır ve except'e düşer.
        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        # Koleksiyon bulunamazsa burada oluşturuyoruz.
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,            # Embedding boyutu sabit (model çıktısı)
                distance=models.Distance.COSINE  # Benzerlik metriği olarak kosinüs seçildi
            )
        )
        print(f"Created collection {collection_name}")


# ---------------------------------------------------------------
# ensure_payload_indexes: Sorgularda sık kullanılacak payload alanları için
# (örn. ihtiyac_kg ve tarih_ts) indeks oluşturur; bu filtrelemeyi hızlandırır.
# has_date True ise tarih_ts alanı için de indeks açılır.
# ---------------------------------------------------------------
def ensure_payload_indexes(collection_name: str, has_date: bool) -> None:
    """Aylık + hammadde bazlı agregasyon sonrası sorgulanacak alanlar için indeks oluşturur."""
    index_targets = [
        ("yil", models.PayloadSchemaType.INTEGER),
        ("ay", models.PayloadSchemaType.INTEGER),
        ("hammadde_kodu", models.PayloadSchemaType.KEYWORD),  # Exact match aramalar için
        ("ihtiyac_toplam", models.PayloadSchemaType.FLOAT),
    ]
    for field_name, schema in index_targets:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema
            )
            print(f"Created payload index on {field_name} in {collection_name}")
        except Exception:
            pass  # Zaten varsa sessiz.

# ---------------------------------------------------------------
# Uygulamada kullanılacak dört ayrı koleksiyon (veri perspektifleri) önceden oluşturuluyor.
# Aynı CSV satırları farklı perspektiflerde (ürün, firma, hammadde, proses) analiz için kullanılabilir.
# ---------------------------------------------------------------
create_collection("textile_materials")

# ---------------------------------------------------------------
# load_textile_data: CSV dosyasını DataFrame'e yükler. Ayrı fonksiyon olması test edilebilirlik sağlar.
# ---------------------------------------------------------------
def load_textile_data(csv_path: str) -> pd.DataFrame:
    """CSV'den tekstil üretim verisini okur ve DataFrame döner."""
    df = pd.read_csv(csv_path)  # pandas read_csv tüm tipleri otomatik tanımaya çalışır.
    return df  # DataFrame'i geri döndür.


# ---------------------------------------------------------------
# parse_float: '1,23' gibi virgüllü veya boşluklu değerleri güvenli şekilde float'a çevirir.
# Lokal veri kaynaklarında ondalık ayıracı virgül olabileceği için normalize edilir.
# None veya parse edilemezse None döner -> downstream mantık hataları önlenir.
# ---------------------------------------------------------------
def parse_float(value: Any) -> Optional[float]:
    """Virgül veya boşluk içerebilecek sayısal değerleri güvenli float'a çevirir."""
    if pd.isna(value):  # NaN / None durumunda direkt None döndür.
        return None
    if isinstance(value, (int, float)):  # Zaten sayısal ise cast yeterli.
        return float(value)
    try:
        s = str(value).strip().replace(" ", "")  # Boşlukları çıkar.
        s = s.replace(',', '.')  # Virgülü noktaya çevir.
        return float(s)  # Float'a çevir ve döndür.
    except Exception:
        return None  # Hata olursa None (sessiz) -> indeksleme sırasında eksik olarak geçilecek.

# ---------------------------------------------------------------
# parse_date: Farklı tarih formatlarını dener; başarılı olursa ISO string ve Unix timestamp (saniye) döner.
# İleride tarih bazlı filtre / sıralama yapmak için timestamp kullanışlı.
# ---------------------------------------------------------------
def parse_date(value: Any) -> Tuple[Optional[str], Optional[int]]:
    """Tarihi ISO (YYYY-MM-DD) ve Unix timestamp (saniye) olarak döndürür."""
    if pd.isna(value):  # Boş ise iki None döneriz.
        return None, None
    if isinstance(value, datetime):  # Zaten datetime nesnesi ise direkt kullan.
        dt = value
    else:
        raw = str(value).strip()  # String normalizasyonu
        fmts = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"]  # Desteklenen formatlar
        dt = None
        for f in fmts:  # Her formata sırayla bak.
            try:
                dt = datetime.strptime(raw, f)  # Format eşleşirse dt atanır.
                break  # Başarılı ise döngü kırılır.
            except ValueError:
                continue  # Uymadıysa diğer formata geç.
        if dt is None:  # Hiçbiri uymadıysa None çiftini döndür.
            return None, None
    iso = dt.date().isoformat()  # Sadece tarih kısmı YYYY-MM-DD
    ts = int(dt.timestamp())  # Epoch saniye
    return iso, ts

# ---------------------------------------------------------------
# create_document_text: Her DataFrame satırı için insan tarafından okunabilir özet metin üretir.
# data_type'a göre hangi alanların bir araya getirileceği seçilir.
# Eksik alanlar '?' ile temsil edilip sonradan filtrelenir.
# ---------------------------------------------------------------
def create_document_text(row: pd.Series) -> str:
    """Aylık + hammadde bazlı toplam için metin üretir.

    Format: 'İhtiyaç: <toplam> Tarih: MM/YY Hammadde Kodu: <kod>'
    """
    yil = row.get("Yıl")
    ay = row.get("Ay")
    kod = row.get("Hammadde Kodu")
    ihtiyac = row.get("İhtiyaç Toplam")
    try:
        ay_str = str(int(ay)).zfill(2)
    except Exception:
        ay_str = str(ay)
    return f"İhtiyaç: {ihtiyac} kg Tarih: {ay_str}/{yil} Hammadde Kodu: {kod}"

# ---------------------------------------------------------------
# prepare_raw_collections: Şu an için aynı DataFrame'i 4 farklı koleksiyona aynen veriyoruz.
# Gelecekte her koleksiyon için farklı filtre / grupla işlemleri yapılabilir.
# ---------------------------------------------------------------
def prepare_raw_collections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Veriyi Yıl-Ay-Hammadde Kodu bazında gruplayıp toplam 'İhtiyaç Kg' hesaplar."""
    work = df.copy()
    required = ["Tarih", "İhtiyaç Kg", "Hammadde Kodu"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {missing}")

    work["Tarih"] = pd.to_datetime(work["Tarih"], errors="coerce")
    work = work.dropna(subset=["Tarih"])  # Geçersiz tarihleri at

    work["İhtiyaç Kg Parsed"] = work["İhtiyaç Kg"].apply(parse_float)
    work = work.dropna(subset=["İhtiyaç Kg Parsed"])  # Parse edilemeyenleri at

    # Hammadde kodu boş olanları dışla
    work["Hammadde Kodu"] = work["Hammadde Kodu"].astype(str).str.strip()
    work = work[work["Hammadde Kodu"] != ""]

    work["Yıl"] = work["Tarih"].dt.year
    work["Ay"] = work["Tarih"].dt.month

    grouped = (
        work.groupby(["Yıl", "Ay", "Hammadde Kodu"], as_index=False)["İhtiyaç Kg Parsed"].sum()
            .rename(columns={"İhtiyaç Kg Parsed": "İhtiyaç Toplam"})
            .sort_values(["Yıl", "Ay", "Hammadde Kodu"]).reset_index(drop=True)
    )

    return {"textile_materials": grouped}

# ---------------------------------------------------------------
# index_textile_data: DataFrame'i batch'lere bölerek Qdrant'a upsert eder.
# Neden batch? Büyük veri setlerinde bellek ve network verimliliği.
# ---------------------------------------------------------------
def index_textile_data(df: pd.DataFrame, collection_name: str) -> None:
    """Aylık agregasyon DataFrame'ini batch halinde Qdrant'a indexler."""
    batch_size = 100
    total_points = 0

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))  # Batch bitiş indeksi (taşma engeli)
        batch_df = df.iloc[batch_start:batch_end]  # İlgili alt DataFrame dilimi
        
        points = []  # Qdrant'a gönderilecek PointStruct listesi
        
        for i, (_, row) in enumerate(batch_df.iterrows()):  # iterrows: her satırı gez
            text = create_document_text(row)  # Satıra uygun metin açıklaması
            
            embedding = model.encode(text).tolist()  # Metni embedding'e çeviriyoruz (list'e)

            payload = {
                "id": batch_start + i,
                "data_type": "textile_materials",
                "text_description": text,
            }

            point = models.PointStruct(
                id=batch_start + i,   # Qdrant'taki nokta ID'si (aynı değeri payload'da da tutuyoruz)
                vector=embedding,     # 384 boyutlu embedding vektörü
                payload=payload       # Ek bağlam verileri (metadata)
            )
            
            points.append(point)  # Hazır point listeye eklenir.

        try:
            client.upsert(
                collection_name=collection_name,  # Hangi koleksiyona yazılacak
                points=points                     # Batch'in tüm point'leri
            )
            total_points += len(points)  # Toplam sayaç güncelle
            print(f"Indexed batch {batch_start//batch_size + 1}: {len(points)} items ({total_points}/{len(df)} total)")
        except Exception as e:
            # Ağ / şema / veri tipi vb. hatalarda batch atlanır, döngü devam eder.
            print(f"Error indexing batch {batch_start//batch_size + 1}: {e}")
            continue
    
    print(f"✅ Successfully indexed {total_points} items in {collection_name}")  # Final özet

# ---------------------------------------------------------------
# __main__ bloğu: Script doğrudan çalıştırıldığında indeksleme sürecini başlatır.
# ---------------------------------------------------------------
if __name__ == "__main__":

    csv_path = "Data\manipulated_data.csv"  # Veri kaynağı dosya yolu (göreli)
    if not os.path.exists(csv_path):  # Dosya var mı kontrol
        print(f"CSV file not found: {csv_path}")
        print("Make sure the CSV file is in the same directory as this script")
        exit(1)  # Kritik hata -> uygulamadan çık
    
    print("Loading textile data...")
    df = load_textile_data(csv_path)  # DataFrame'e yükle
    print(f"Loaded {len(df)} records")  # Kaç satır okunduğunu logla

    print("Preparing monthly aggregation (Yıl-Ay bazında)...")
    grouped_data = prepare_raw_collections(df)
    for collection_name in grouped_data.keys():  # Her koleksiyon için indeksler
        ensure_payload_indexes(collection_name, has_date=True)

    for collection_name, collection_df in grouped_data.items():  # Her koleksiyonu sırayla indexle
        print(f"Indexing {collection_name}")
        index_textile_data(collection_df, collection_name)
    
    print("Indexing completed!")  # Süreç tamam mesajı