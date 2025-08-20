### Akış

1) Veri seti analizi:
- Kategorik önem sırası (hedefe göre)
- Sayısal değerlerin önem derecesi (hedefe göre) (0.3 üstü önemli, eklenebilir)
- Frequency grafiği (veri dağılımı)

2) Veri analizi sonuçları:
- Hangi özelliğin katkısı var değerlendirilir
- Yeterli veri toplayabiliyorsak, veri değılımını o şekilde düzenleriz. Toplayamıyorsak:
    - Veri dağılıma göre uç noktalar varsa, log dönüşümü ya da percentile dönüşümü.
    - Alt ve üst (5 percentile - 95 percentile) kırpılabilir. (gerçek hayatta performans düşebilir)

3) Küçük bir model eğitilir sonucu takip edilir. Analiz sonucunda çıkmayan ama gerçek eğitime katkı sağlayan özellikler varsa onlar belirlenir. (Gizli ilişkiler.) 
Eğer varsa, özellikler arası etkileşimler eklenebilir (feature engineering)(örneğin ay gün sin/cos dönüşümü (tekrar eden yapıyı temsil etmesi için) ya da mevsim dağılımı ya da iki ayrı özelliğin toplamı)

----------------------------------------------Veri Analizi Bitti----------------------------------------------

4) Asıl modelleme. Setimi tanıyorum. Karmaşıklığını biliyorum. Aşırı karışık değilse, sci-kit'den paket modeller arasından linear regression ya da random forest denenir. Eğer performans artmıyorsa (hazır model kısıtlamasından dolayı) sinir ağlarına geçilir. Sequential ağ üstünde Dense katmanlarıyla, veri karmaşıklığına göre katman derinliği ve genişliği belirlenir. Optimizer/regularization fonksiyonları/normalization fonksiyonları/loss function/activation function en sık kullanlılanlar sırayla Adam/l2/batch normalization/MSE/ReLU eklenir.

5) Dördüncü adım sonuçlarına göre, önce model karmaşılğı azaltılıp artılarak test edilir (genişlik - derinlik), sonra loss function değiştirilip test edilir, sonra optimizer, sonra regularizer veya batch normalization kaldırılıp/değiştirilip test edilir. 

6) Beşinci adım sonuçları tatmin ediciyse dağıtıma geçilir, değilse en başa ya da model geliştirme kısmına dönülüp alternatifler denenir.

7) Sonuç yeterince iyileşmiyorsa ensemble (birleştirme) metoduna başvurulur.

Dipnot: 

- R2 skorum 0.7 üstündeyse o model emek vermeye ve iyileştirmeye çalışmaya değer, değilse yeni denemelere başlanır. 0.9 ve üstüne çıktığımda artık o model olmuştur.
- Ezber: val_loss yani değerlendirme skoru loss'un altındaysa ve mesafe uzaksa kesinlikle ezber vardır. Model çok büyüktür y ada çok küçüktür. Tam tersi senaryoda actual vs predicted sonuçlarını takip etmek lazım.