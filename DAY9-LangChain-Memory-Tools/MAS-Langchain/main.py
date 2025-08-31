from typing import List, Dict, Tuple
import re
from qdrant_client import QdrantClient

from Agents.merkez_ajan import merkez_ajan
from Agents.hammadde_ajan import hammadde_ajan
from Agents.talep_ajan import talep_ajan
from Agents.ülke_ajan import ülke_ajan
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Ajan eşleştirme - LangChain versiyonu
AGENT_NAME_MAP = {
	"Hammadde Ajanı": hammadde_ajan,
	"Talep Ajanı": talep_ajan,
	"Ülke Ajanı": ülke_ajan,
}

# Hangi ajan hangi koleksiyona bakıyor
AGENT_COLLECTION_MAP = {
	"Talep Ajanı": ["textile_materials"],
	"Ülke Ajanı": ["textile_country"],
	"Hammadde Ajanı": ["textile_substance"],
}

client = QdrantClient(host="localhost", port=6333)

def parse_merkez_output(text: str) -> Tuple[str, str]:
	"""Merkez ajan çıktısından hedef ajan adını ve keyword string'ini çıkarır."""
	# Beklenen format: {Ajan: <Talep Ajanı>}, {Keyword: <Hammadde Kodu: NPE 1000 449>}}
	ajan_match = re.search(r"\{Ajan:\s*<([^>]+)>\}\s*", text)
	kw_match = re.search(r"\{Keyword:\s*<([^>]+)>\}\s*", text)
	if not ajan_match or not kw_match:
		raise ValueError(f"Merkez ajan çıktısı parse edilemedi: {text}")
	return ajan_match.group(1).strip(), kw_match.group(1).strip()

def extract_filters(keyword_str: str) -> Dict[str, str]:
	"""Keyword string'inde 'Alan: Değer' tokenlerini yakalar.
	Ör: 'Hammadde Kodu: NPE 1000 449 Tarih: 2023 Firma Ülkesi: TURKEY'
	-> {'Hammadde Kodu': 'NPE 1000 449', 'Tarih': '2023', 'Firma Ülkesi': 'TURKEY'}"""
	pattern = r"([A-Za-zÇĞİÖŞÜçğıöşü\s]+?):\s*([^:]+?)(?=(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü]+?:)|$)"
	matches = re.findall(pattern, keyword_str)
	filters = {k.strip(): v.strip() for k, v in matches}
	return filters

def vector_search_collections(agent_label: str, query_text: str, limit: int = 30) -> List[Dict]:
	"""Belirli ajan koleksiyonlarında SADECE vektör benzerliği ile arama yapar.
	Filtre (payload condition) kullanılmaz. Sürdürülebilir ve basit tutmak için
	Qdrant'ın yeni query_points API'si tercih edilir; hata durumunda eski search'e düşer."""
	collections = AGENT_COLLECTION_MAP.get(agent_label, [])
	if not collections:
		return []
	if not query_text:
		return []
	query_vec = model.encode(query_text).tolist()
	context_docs: List[Dict] = []
	for col in collections:
		try:
			res = client.query_points(
				collection_name=col,
				query=query_vec,
				limit=limit,
				with_payload=True,
				with_vectors=False
			)
			points = res.points if hasattr(res, 'points') else res
		except Exception:
			try:
				points = client.search(collection_name=col, query_vector=query_vec, limit=limit)
			except Exception as e2:
				print(f"Arama hatası ({col}): {e2}")
				continue
		for p in points:
			payload = getattr(p, 'payload', None) or p.payload
			# Qdrant score: COSINE benzerliği (daha yüksek daha iyi).
			score = getattr(p, 'score', None)
			context_docs.append({
				"collection": col,
				"text": payload.get("text_description") if payload else None,
				"payload": payload,
				"score": score,
			})
	return context_docs

def build_agent_input(question: str, context: List[Dict]) -> List[Dict[str, str]]:
	context_lines = []
	for c in context:
		score_part = f" (score={c['score']:.4f})" if c.get('score') is not None else ""
		context_lines.append(f"- {c['collection']}{score_part}: {c['text']}")
	context_str = "\n".join(context_lines) if context_lines else "(Kayıt bulunamadı)"
	agent_input = (f"User:\n{question}\n\n" f"Context:\n{context_str}")
	return [{"role": "user", "content": agent_input}]

def route_question(user_question: str):
	# 1) Merkez ajan cevabı - LangChain ile
	merkez_reply = merkez_ajan.generate_reply([{"role": "user", "content": user_question}])
	
	# LangChain'den gelen yanıt zaten string olarak döner
	merkez_reply_str = str(merkez_reply)
	print("Merkez Ajan Çıktısı:", merkez_reply_str)
	
	# 2) Parse
	target_label, keyword_str = parse_merkez_output(merkez_reply_str)
	if target_label not in AGENT_NAME_MAP:
		raise ValueError(f"Bilinmeyen ajan etiketi: {target_label}")
	
	# 3) Vektör benzerlikli arama (yalnızca Keyword içeriği ile)
	context = vector_search_collections(target_label, keyword_str, limit=20)
	
	# Elde edilen sonuçları terminalde göster
	print("\nBenzerlik Sonuçları (koleksiyon | skor):")
	if context:
		for c in context:
			score_val = c.get("score")
			if score_val is not None:
				print(f"[{c['collection']}]\t{score_val:.4f}\t{c['text']}")
			else:
				print(f"[{c['collection']}]\t-\t{c['text']}")
	else:
		print("(Sonuç bulunamadı)")
	
	# 4) Alt ajana ilet - LangChain ile
	agent = AGENT_NAME_MAP[target_label]
	messages = build_agent_input(user_question, context)
	reply = agent.generate_reply(messages)
	print(f"\n{target_label} Yanıtı:\n{reply}\n")
	return reply

if __name__ == "__main__":
	print("MAS sistemi başlatıldı (LangChain versiyonu). Çıkmak için Ctrl+C.")
	while True:
		try:
			q = input("Soru: ").strip()
			if not q:
				continue
			route_question(q)
		except KeyboardInterrupt:
			print("\nÇıkılıyor...")
			break
		except Exception as e:
			print("Hata:", e)