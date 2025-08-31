from typing import Any 
from langchain.schema import BaseMemory  # Bellek sınıfları için temel soyutlama (polimorfizm sağlar)

MODEL_NAME = "gemini-1.5-flash"  # Tüm ajanlarda kullanılacak varsayılan model ismi (tek noktadan yönetim)

def get_model_name() -> str:  # Model adını döndüren yardımcı (ileride dinamik seçim ekleyebilmek için fonksiyonlaştırıldı)
    return MODEL_NAME

def print_memory_state(memory: BaseMemory):  # Verilen bellek örneğinin durumunu ayrıntılı yazdırır (debug / eğitim amaçlı)
    print("================= MEMORY STATE =================")  # Başlık ayırıcı
    print(f"Memory sınıfı: {type(memory).__name__}")  # Hangi bellek sınıfı (ör. ConversationBufferMemory)

    def is_message_list(obj):  # Objeyi 'Message' benzeri elemanlardan oluşan liste olarak tanımlamaya çalışır
        return isinstance(obj, list) and all(hasattr(m, "content") for m in obj)

    def format_messages(msgs):  # Mesaj listesini rol + içerik satırlarına dönüştürür
        lines = []
        for m in msgs:
            role_type = getattr(m, "type", "")  # LangChain mesajının tip alanı (human/ai/system vb.)
            if role_type == "human":
                role = "KULLANICI"  # Kullanıcı mesajı
            elif role_type == "ai":
                role = "ASISTAN"  # Model yanıtı
            else:
                role = m.__class__.__name__  # Tanınmayan tipte sınıf adını göster (örn. SystemMessage)
            lines.append(f"{role}: {m.content}")  # İçeriği ekle
        return "\n".join(lines)

    try:  # Bellek değişkenlerini güvenli biçimde almaya çalış
        try:
            variables = memory.load_memory_variables({})  # Çoğu bellek türü boş dict ile çalışır
        except Exception as inner_e:  # noqa: BLE001  # Bazı bellekler 'input' anahtarı bekler
            if "One input key expected" in str(inner_e):  # Hata mesajına göre fallback stratejisi
                try:
                    variables = memory.load_memory_variables({"input": ""})  # Boş input ile yeniden dene
                except Exception as inner_e2:  # noqa: BLE001  # Yine başarısız olursa tamamen bırak
                    print(f"load_memory_variables hata: {inner_e2}")
                    variables = {}
            else:
                print(f"load_memory_variables hata: {inner_e}")
                variables = {}
        for k, v in variables.items():  # Dönen bellek değişkenlerini sırayla yazdır
            if is_message_list(v):  # Mesaj listesi ise biçimlendir
                print(f"Değişken '{k}':\n{format_messages(v)}")
            else:  # Diğer tipler (string özet, json vb.) kısaltılmış repr ile göster
                print(f"Değişken '{k}': {repr_limit(v, limit=100000)}")
    except Exception as e:  # En dış seviye koruma
        print(f"load_memory_variables genel hata: {e}")

    extras = {}  # Bellek tipine göre mevcut olabilecek ek iç alanları topla
    for attr in ["buffer", "summary", "moving_summary_buffer", "entity_store"]:  # chat_memory gizlendi
        if hasattr(memory, attr):  # Özellik var mı kontrolü
            try:
                extras[attr] = getattr(memory, attr)  # Değeri okuyup extras sözlüğüne koy
            except Exception:  # noqa: BLE001  # Herhangi bir erişim hatasını yut (koruyucu)
                pass
    if extras:  # Ek alan varsa başlık bas
        print("-- Dahili Alanlar --")
        for k, v in extras.items():  # Ek alanları sırayla yazdır
            if is_message_list(v):
                print(f"{k}:\n{format_messages(v)}")
            else:
                print(f"{k}: {repr_limit(v, limit=100000)}")
    print("=================================================\n")  # Kapanış çizgisi


def repr_limit(obj: Any, limit: int = 500) -> str:  # Uzun objelerin repr çıktısını güvenli şekilde kısaltır
    s = repr(obj)  # Standart repr (potansiyel olarak uzun olabilir)
    return s if len(s) <= limit else s[: limit - 3] + "..."  # Limit aşılırsa '...' ekle
