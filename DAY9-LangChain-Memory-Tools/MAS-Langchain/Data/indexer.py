from typing import Dict, Any, List, Optional, Tuple 
import pandas as pd 
from qdrant_client import QdrantClient  
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

model = SentenceTransformer('all-MiniLM-L6-v2')

client = QdrantClient(host="localhost", port=6333)

def create_collection(collection_name: str, vector_size: int = 384) -> None:
    """Eğer yoksa Qdrant içinde bir koleksiyon oluşturur."""
    try:

        client.get_collection(collection_name=collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,           
                distance=models.Distance.COSINE 
            )
        )
        print(f"Created collection {collection_name}")

def ensure_payload_indexes(collection_name: str, has_date: bool) -> None:
    """Aylık + hammadde bazlı agregasyon sonrası sorgulanacak alanlar için indeks oluşturur."""
    index_targets = [
        ("yil", models.PayloadSchemaType.INTEGER),
        ("ay", models.PayloadSchemaType.INTEGER),
        ("hammadde_kodu", models.PayloadSchemaType.KEYWORD),
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
            pass 

for base_collection in ["textile_materials", "textile_country", "textile_substance"]:
    create_collection(base_collection)

def load_textile_data(csv_path: str) -> pd.DataFrame:
    """CSV'den tekstil üretim verisini okur ve DataFrame döner."""
    df = pd.read_csv(csv_path)
    return df

def parse_float(value: Any) -> Optional[float]:
    """Virgül veya boşluk içerebilecek sayısal değerleri güvenli float'a çevirir."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip().replace(" ", "") 
        s = s.replace(',', '.') 
        return float(s)
    except Exception:
        return None  

def parse_date(value: Any) -> Tuple[Optional[str], Optional[int]]:
    """Tarihi ISO (YYYY-MM-DD) ve Unix timestamp (saniye) olarak döndürür."""
    if pd.isna(value):
        return None, None
    if isinstance(value, datetime): 
        dt = value
    else:
        raw = str(value).strip() 
        fmts = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"] 
        dt = None
        for f in fmts:
            try:
                dt = datetime.strptime(raw, f)  
                break 
            except ValueError:
                continue 
        if dt is None:
            return None, None
    iso = dt.date().isoformat()  
    ts = int(dt.timestamp())
    return iso, ts

def create_material_text(row: pd.Series) -> str:
    """Talep (material aggregation) için metin.
    Format: 'İhtiyaç: <toplam> kg Tarih: <MM/YY> Hammadde Kodu: <kod>'"""
    yil = row.get("Yıl")
    ay = row.get("Ay")
    kod = row.get("Hammadde Kodu")
    ihtiyac = row.get("İhtiyaç Toplam")
    ay_str = str(int(ay)).zfill(2) if pd.notna(ay) else "??"
    return f"İhtiyaç: {ihtiyac} kg Tarih: {ay_str}/{yil} Hammadde Kodu: {kod}"

def create_country_text(row: pd.Series) -> str:
    """Ülke koleksiyonu için text_description.
    Format: 'Tarih:<MM/YY> Firma Ülkesi: <ülke> İhtiyaç Kg: <değer>'"""
    yil = row.get("Yıl")
    ay = row.get("Ay")
    ulke = row.get("Firma Ülkesi")
    ihtiyac = row.get("İhtiyaç Kg Toplam")
    ay_str = str(int(ay)).zfill(2) if pd.notna(ay) else "??"
    return f"Tarih:{ay_str}/{yil} Firma Ülkesi: {ulke} İhtiyaç Kg: {ihtiyac}"

def create_substance_text(row: pd.Series) -> str:
    """Hammadde koleksiyonu için text_description.
    Format: 'Tarih:<MM/YY> Hammadde Kodu: <kod> Kumaş Kodu: <kumas>'"""
    yil = row.get("Yıl")
    ay = row.get("Ay")
    hammadde = row.get("Hammadde Kodu")
    kumas = row.get("Kumaş Kodu")
    ay_str = str(int(ay)).zfill(2) if pd.notna(ay) else "??"
    return f"Tarih:{ay_str}/{yil} Hammadde Kodu: {hammadde} Kumaş Kodu: {kumas}"

def prepare_raw_collections(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Farklı koleksiyonlar için gerekli gruplamaları hazırlar.
    - textile_materials: Yıl, Ay, Hammadde Kodu -> İhtiyaç Toplam
    - textile_country: Yıl, Ay, Firma Ülkesi -> İhtiyaç Kg Toplam
    - textile_substance: Yıl, Ay, Hammadde Kodu, Kumaş Kodu (satır bazlı – unique)"""
    work = df.copy()
    required = ["Tarih", "İhtiyaç Kg", "Hammadde Kodu", "Firma Ülkesi", "Kumaş Kodu"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Beklenen kolonlar eksik: {missing}")

    work["Tarih"] = pd.to_datetime(work["Tarih"], errors="coerce")
    work = work.dropna(subset=["Tarih"])

    work["İhtiyaç Kg Parsed"] = work["İhtiyaç Kg"].apply(parse_float)
    work = work.dropna(subset=["İhtiyaç Kg Parsed"])

    # Temizlik
    work["Hammadde Kodu"] = work["Hammadde Kodu"].astype(str).str.strip()
    work["Firma Ülkesi"] = work["Firma Ülkesi"].astype(str).str.strip()
    work["Kumaş Kodu"] = work["Kumaş Kodu"].astype(str).str.strip()
    work = work[(work["Hammadde Kodu"] != "") & (work["Firma Ülkesi"] != "") & (work["Kumaş Kodu"] != "")]

    work["Yıl"] = work["Tarih"].dt.year
    work["Ay"] = work["Tarih"].dt.month

    materials = (
        work.groupby(["Yıl", "Ay", "Hammadde Kodu"], as_index=False)["İhtiyaç Kg Parsed"].sum()
            .rename(columns={"İhtiyaç Kg Parsed": "İhtiyaç Toplam"})
            .sort_values(["Yıl", "Ay", "Hammadde Kodu"]).reset_index(drop=True)
    )

    country = (
        work.groupby(["Yıl", "Ay", "Firma Ülkesi"], as_index=False)["İhtiyaç Kg Parsed"].sum()
            .rename(columns={"İhtiyaç Kg Parsed": "İhtiyaç Kg Toplam"})
            .sort_values(["Yıl", "Ay", "Firma Ülkesi"]).reset_index(drop=True)
    )

    # Substance: satır bazlı; gerekli kolonları alıp yıl/ay ekleyerek benzersiz kayıtlar
    substance = work[["Yıl", "Ay", "Hammadde Kodu", "Kumaş Kodu"]].drop_duplicates().reset_index(drop=True)

    return {
        "textile_materials": materials,
        "textile_country": country,
        "textile_substance": substance,
    }

def index_textile_data(df: pd.DataFrame, collection_name: str) -> None:
    """Verilen DataFrame'i ilgili koleksiyona indexler."""
    batch_size = 200
    total_points = 0

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        points = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            if collection_name == "textile_materials":
                text = create_material_text(row)
                payload_extra = {
                    "Yıl": int(row.get("Yıl")),
                    "Ay": int(row.get("Ay")),
                    "Hammadde Kodu": row.get("Hammadde Kodu"),
                    "İhtiyaç Toplam": float(row.get("İhtiyaç Toplam")),
                    "agent_owner": "talep_ajan"
                }
            elif collection_name == "textile_country":
                text = create_country_text(row)
                payload_extra = {
                    "Yıl": int(row.get("Yıl")),
                    "Ay": int(row.get("Ay")),
                    "Firma Ülkesi": row.get("Firma Ülkesi"),
                    "İhtiyaç Kg Toplam": float(row.get("İhtiyaç Kg Toplam")),
                    "agent_owner": "ülke_ajan"
                }
            else:  # textile_substance
                text = create_substance_text(row)
                payload_extra = {
                    "Yıl": int(row.get("Yıl")),
                    "Ay": int(row.get("Ay")),
                    "Hammadde Kodu": row.get("Hammadde Kodu"),
                    "Kumaş Kodu": row.get("Kumaş Kodu"),
                    "agent_owner": "hammadde_ajan"
                }

            embedding = model.encode(text).tolist()
            point_id = batch_start + i
            payload = {
                "id": point_id,
                "collection": collection_name,
                "text_description": text,
            }
            payload.update(payload_extra)

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        if not points:
            continue
        try:
            client.upsert(collection_name=collection_name, points=points)
            total_points += len(points)
            print(f"Indexed {collection_name} batch {batch_start//batch_size + 1}: {len(points)} items ({total_points}/{len(df)})")
        except Exception as e:
            print(f"Error indexing {collection_name} batch {batch_start//batch_size + 1}: {e}")
            continue

if __name__ == "__main__":

    csv_path = "Data\manipulated_data.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Make sure the CSV file is in the same directory as this script")
        exit(1)
    
    print("Loading textile data...")
    df = load_textile_data(csv_path)
    print(f"Loaded {len(df)} records")

    print("Preparing grouped data for all collections...")
    grouped_data = prepare_raw_collections(df)
    # Sadece eski index fonksiyonu textil_materials alanları için tasarlanmış, diğerleri için şimdilik payload index gerekirse eklenebilir.
    for cname in grouped_data.keys():
        try:
            ensure_payload_indexes(cname, has_date=True)
        except Exception:
            pass

    for cname, cdf in grouped_data.items():
        print(f"Indexing {cname} -> {len(cdf)} rows")
        index_textile_data(cdf, cname)
    
    print("Indexing completed!")