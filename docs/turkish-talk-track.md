# Tracefold — 5 Dakikalık Türkçe Sunum Metni

Bu metin altı İngilizce slaytla uyumludur. Hedef süre yaklaşık beş dakikadır. Köşeli parantez içindeki ifadeler sahne yönlendirmesidir; sesli okunmaz.

## Slide 1 — Tracefold: Answers You Can Trace — 0:00–0:40

Merhaba, bugün size Tracefold'u göstereceğim. Tracefold, belgeler cihazdan ayrılmadan İngilizce ve Türkçe soruları yanıtlayan, her olgusal cümleyi açılabilir bir kanıtla ilişkilendiren yerel bir RAG uygulaması.

Çözmek istediğim sorun şu: Bir sohbet modeli son derece akıcı bir yanıt verebilir; peki bu yanıtın gerçekten belgelere dayandığını nasıl anlayacağız? Tracefold yalnızca güzel bir yanıt üretmiyor. Yanıtın hangi belgeye, hangi bölüme ve hangi metin parçasına dayandığını görünür kılıyor. Sloganımız da bu nedenle “Answers you can trace”, yani “kaynağını izleyebileceğiniz yanıtlar”.

## Slide 2 — The Problem: Private Documents, Unverifiable Answers — 0:40–1:20

Belge asistanlarında üç temel risk var. Birincisi gizlilik: Belgeler ve sorular uzaktaki bir hizmete gönderilebilir. İkincisi halüsinasyon: Model, kaynakta bulunmayan bir bilgiyi ikna edici biçimde uydurabilir. Üçüncüsü denetlenebilirlik: Kullanıcı, yanıtın dayanağını kolayca göremeyebilir.

Tracefold bu üç riski birlikte ele alıyor. Bilgi getirme için `qwen3-embedding-0.6b`, kanıta dayalı İngilizce iddialar için `qwen2.5-7b` Microsoft Foundry Local aracılığıyla bu cihazda çalışıyor. Türkçe ile İngilizce arasındaki kontrollü çeviriyi ise iki ayrı OPUS-MT modeli, yine yerel çalışma ortamında yapıyor. Yeterli kanıt bulunamadığında modeli yanıt vermeye zorlamak yerine güvenli ret uyguluyor. Kanıt bulunduğunda ise atfa tıklayıp kullanılan metin parçasını doğrudan inceleyebiliyoruz.

## Slide 3 — Local RAG Architecture — 1:20–2:15

[Mimari diyagramını göster.]

FastAPI belgeleri ayrıştırıyor; başlık ve sayfa bilgisini koruyarak metni yaklaşık 240 token içeren, bölüm sınırlarını aşmayan parçalara ayırıyor. Metinler, üst veriler ve `qwen3-embedding-0.6b` vektörleri SQLite'ta saklanıyor.

İngilizce soru doğrudan bilgi getirmeye giriyor. Türkçe soruda tarih, saat, telefon ve e-posta gibi değerler koruma belirteçleriyle maskeleniyor; Northstar sözlüğü terimleri sabitliyor ve `opus-mt-tc-big-tr-en` soruyu İngilizceye çeviriyor. Bilgi getirme, özgün Türkçe soru ile denetimden geçen İngilizce çeviriyi birlikte kullanıyor. Bir belirteç değişir veya anlam denetimi başarısız olursa sistem güvenli ret veriyor.

Kosinüs benzerliğiyle en yakın beş metin parçası bulunuyor. Kanıt yeterliyse `qwen2.5-7b` İngilizce iddialar üretiyor. İddialar ve atıflar doğrulandıktan sonra `opus-mt-tc-big-en-tr` yalnızca iddia metnini çeviriyor; atıflar cümle eşlemesinden yeniden kuruluyor. Ters çeviri, kritik olgu ve dil kontrolleri geçilmeden Türkçe yanıt gösterilmiyor. Bütün akış cihazda kalıyor.

## Slide 4 — Product Demo: From Documents to Evidence — 2:15–3:15

[Library ekranını, ardından Ask ekranını göster.]

Demo için Northstar Summer Lab adlı kurmaca bir programa ait, birbirini tamamlayan sekiz belge hazırladım.

Önce İngilizce olarak, varış günü saat yirmiden sonra kampüse ulaşacağımı, bu durumda ne yapıp kiminle iletişime geçmem gerektiğini soruyorum. Değerlendirme, ilk beş sonuçta seyahat, konaklama ve iletişim kaynaklarını arıyor. Yanıtta ise yalnızca kullanılan kanıtlara atıf veriliyor.

[Atıf panelini aç.]

Tracefold, Yurt Nöbet Telefonu numarasının saat on dokuz otuzdan önce aranması ve Program Ofisine yeni varış saatinin bildirilmesi gerektiğini söylüyor. Her atfın dosyasını, metin parçasını ve benzerlik puanını burada görebiliyorum.

[Arayüz dilini Türkçeye çevir ve eşdeğer soruyu sor.]

Aynı soruyu Türkçe sorduğumda saat, telefon ve e-posta aynen korunuyor. Atıflar çeviri modeline yazdırılmıyor; doğrulanan İngilizce cümlelerle olan eşlemeden yeniden kuruluyor. Türkçe burada yalnızca arayüz değil; soru, bilgi getirme, yanıt, atıf ve güvenli ret akışlarının tamamını kapsıyor.

## Slide 5 — Quality: Retrieval, Citations, and Refusal — 3:15–4:15

Kaliteyi yalnızca “yanıt iyi görünüyor” diyerek değerlendirmiyoruz. On iki İngilizce senaryonun her biriyle Türkçe eşdeğeri bire bir eşleşiyor: toplam yirmi dört senaryo. Her dilde sekiz soru yanıtlanabilir; dört soru güvenli ret gerektiriyor.

Bilgi getirme, kaynak kapsamı, atıf geçerliliği, kritik olgular ve güvenli ret doğruluğunu ölçüyoruz. Türkçede bunlara belirteçlerin dönmesi, atıfların kurulması, ters çeviri tutarlılığı ve dil kalitesi ekleniyor. Sonuçları dil bazında ve on iki eşleştirilmiş senaryo üzerinden izliyoruz.

[Quality ekranında hedefler ile ölçülmüş sonuçların ayrımını göster.]

Hedefleri ölçülmüş sonuç gibi sunmuyoruz. Bir çalışmayı ancak veri kümesi ve dört modelin kimliği, bilgi getirme ve çeviri ayarları, Git commit kimliği ve temiz çalışma ağacıyla birlikte kaydedildiyse kabul ediyoruz. Ekrandaki sayılar son kabul edilen `docs/evaluation-report.json` dosyasından geliyor.

Burs tutarı gibi belgelerde bulunmayan bir bilgi sorulduğunda Tracefold rakam uydurmuyor; Türkçe olarak “Kütüphanenizde bu soruyu yanıtlamak için yeterli kanıt bulamadım” diyor.

## Slide 6 — Why Tracefold — 4:15–5:00

Tracefold; belge alma, yerel vektörleştirme, SQLite'ta saklama, bilgi getirme, kaynağa dayalı yanıt, kontrollü çeviri, güvenli ret ve iki dilli değerlendirmeyi tek bir çevrimdışı ürün deneyiminde birleştiriyor.

Kullanıcı atfın arkasındaki kanıtı inceleyebiliyor; model sorunu gizli bir yedek akışla maskelenmiyor; hedefler, sonuçlar ve sınırlamalar ayrı tutuluyor.

Tracefold, kaynağı görülebilen ve bilmediği yerde durabilen bir yanıt katmanı. Northstar sözlüğü genişletilebilir; ancak bugünkü iki dilli kalite iddiası bu kurmaca belge kümesinin yirmi dört senaryolu değerlendirmesine dayanıyor. Yeni bir alanda terimler gözden geçirilmeli ve değerlendirme yeniden çalıştırılmalı. Teşekkür ederim.
