# Tracefold — 5 Dakikalık Türkçe Sunum Metni

Bu metin altı slaytlık Türkçe sunumla uyumludur. Hedef süre yaklaşık beş dakikadır. Köşeli parantez içindeki ifadeler sahne yönlendirmesidir; sesli okunmaz. Değerlendirme slaydındaki sonuçları yalnızca son gerçek model raporu ilgili sürümün Git commit kimliğini kaydediyor ve çalışma ağacının temiz olduğunu doğruluyorsa başarı olarak anlatın.

## Slayt 1 — Kaynağını izleyebildiğiniz yanıtlar. — 0:00–0:40

Merhaba, bugün size Tracefold'u göstereceğim. Tracefold, özel belgelerden İngilizce ve Türkçe yanıtlar üreten yerel bir RAG uygulaması. Temel vaadi basit: Kaynağını izleyebildiğiniz yanıtlar.

Akıcı bir yanıt tek başına güvenilir değildir. Bir belge asistanı, kullanıcının “Bu bilgi nereden geldi?” sorusuna dosyayı, bölümü ve ilgili metin parçasını göstererek yanıt verebilmelidir. Tracefold bu nedenle yalnızca metin üretmez; yanıtla kanıt arasındaki bağı ürün deneyiminin merkezine yerleştirir.

## Slayt 2 — Özel belgeler doğrulanabilir yanıtları hak eder — 0:40–1:20

Belge asistanlarında üç risk aynı anda ortaya çıkar. Belgeler uzaktaki bir hizmete gönderildiğinde gizlilik zayıflar. Model kaynakta olmayan bir ayrıntıyı ikna edici biçimde eklediğinde doğruluk zayıflar. Kullanıcı dayanağı açamadığında da denetlenebilirlik kaybolur.

Tracefold bu üç alanı birlikte ele alıyor. `qwen3-embedding-0.6b` arama için gereken gömme vektörlerini, `qwen2.5-14b` ise kanıta dayalı İngilizce iddiaları Microsoft Foundry Local üzerinden bu cihazda üretiyor. Türkçe ile İngilizce arasındaki kontrollü geçişi iki yerel OPUS-MT modeli sağlıyor. Yeterli kanıt yoksa sistem modeli tahmine zorlamıyor; açık ve sabit bir güvenli ret yanıtı veriyor.

## Slayt 3 — Yanıt yerelde kalır; kanıt yanıttan kopmaz — 1:20–2:15

[Mimari akışı göster.]

FastAPI belgeleri ayrıştırıyor; başlık ve sayfa bilgisini koruyarak bölüm sınırlarını aşmayan parçalara ayırıyor. Metinler, meta veriler ve Float32 vektörler SQLite'ta yerel olarak saklanıyor.

Soru için kosinüs benzerliği hesaplanıyor. En yüksek puan `0.60` eşiğinin altındaysa yanıt üretimi başlamıyor. Eşik geçilirse `0.30` ve üzerindeki kanıtlar arasından soruya odaklanan en fazla dört parça seçiliyor; bu parçalardan alınan metin toplam `3200` karakterlik bütçeyle sınırlanıyor.

Türkçe sorularda tarih, saat, sayı, telefon, e-posta, ay ve gün adı gibi değerler türlerine göre etiketlenmiş koruma belirteçleriyle maskeleniyor. Soru İngilizceye çevrildikten sonra bu değerlerin eksiksiz geri gelmesi gerekiyor. `qwen2.5-14b` yalnızca kanıttan İngilizce iddialar kuruyor ve her olgusal cümle kendi atfıyla bitiyor. Atıfsız cümle, geçersiz kaynak numarası ya da kanıtta bulunmayan kritik değer yanıtı geçersiz kılıyor.

Doğrulanmış iddialar Türkçeye çevrildiğinde atıflar çeviri modeline bırakılmıyor; cümle eşlemesinden yeniden kuruluyor. Ters çeviride koruma belirteçleri dışında kalan içeriğin en az yüzde 60'ı, genel içeriğin ise en az yüzde 30'u geri kazanılmalı. Tarih ve gün bilgisi, zorunluluk veya izin anlamı, kimin aranacağı ve işlemlerin sırası da ayrı kapılardan geçiyor. İlk İngilizce çıktı doğrulamayı geçmezse en fazla bir kanıta dayalı onarım yapılıyor. Türkçe çeviri kapısı reddederse ayrıca en fazla bir cümle bazlı yeniden yazım deneniyor. Bütün denetimler yeniden geçilmezse güvenli ret veriliyor.

## Slayt 4 — Kanıt açıldığında yararlı bir yanıt güvenilir hâle gelir — 2:15–3:15

[Türkçe Kütüphane ekranını, ardından Türkçe Sor ekranını göster.]

Demo, Northstar Summer Lab adlı kurmaca bir programa ait birbirini tamamlayan sekiz belgeyi kullanıyor. Şu soruyu soruyorum: “Varış günü kampüse saat 20:00'den sonra ulaşacağım. Ne yapmalıyım ve kiminle iletişime geçmeliyim?”

[Yanıt tamamlanınca bir atfı aç.]

Tracefold yalnızca sorulan işlemleri sıralıyor: Yurt Nöbet Telefonu’nun ne zaman aranacağını ve Program Ofisi’ne hangi bilginin iletileceğini söylüyor. `19:30`, telefon numarası ve e-posta adresi kaynaktaki biçimiyle korunuyor. Atfı açtığımda dosya adını, ilgili bölümü, kullanılan metin parçasını ve benzerlik puanını görebiliyorum.

Buradaki önemli ayrım şu: Atıf, yanıt üretildikten sonra süs olarak eklenmiyor. Her cümle kendi kaynak numarasıyla doğrulanıyor; Türkçe yanıtın atıfları da doğrulanmış İngilizce iddia haritasından yeniden kuruluyor. Böylece kullanıcı hem sonucu okuyabiliyor hem de dayanağını doğrudan denetleyebiliyor.

## Slayt 5 — Güven varsayılmaz, ölçülür — 3:15–4:15

Kaliteyi “yanıt kulağa doğru geliyor” diyerek değerlendirmiyoruz. Veri kümesinde on iki İngilizce senaryonun her biri Türkçe eşdeğeriyle eşleştiriliyor; toplam yirmi dört vaka var. Her dilde sekiz vaka yanıt üretimini, dört vaka ise güvenli reti sınamak üzere tasarlandı. Bu sayılar bir başarı beyanı değil, test kapsamının yapısıdır.

[Kalite ekranında ölçülen sonuçlarla kabul hedeflerini ayrı ayrı göster.]

Bilgi getirmede beklenen kaynakların bulunup bulunmadığını; yanıtta her cümlenin geçerli ve doğru kanıta bağlanmasını; tarih, saat, telefon ve e-posta gibi korunan olguların kaynaktan sapmamasını ölçüyoruz. Çok adımlı işlemlerde önce-sonra ilişkisi ayrıca sınanıyor. Türkçede bunlara koruma belirteçlerinin eksiksiz dönmesi, takvim bilgilerinin korunması, atıfların yeniden kurulması, ters çeviri tutarlılığı ve dil kalitesi ekleniyor.

Slayttaki ölçümleri ancak rapor dört modelin kimliğini, tüm eşikleri, değerlendirilen Git commit kimliğini ve temiz çalışma ağacını birlikte kaydediyorsa sürüm kanıtı olarak kabul ediyoruz. Tam başarı ifadesini de yalnızca rapordaki yirmi dört vakanın tamamı geçtiğinde kullanıyoruz. Belgelerde bulunmayan burs tutarı sorulduğunda sistemin rakam uydurmak yerine Türkçe güvenli ret yanıtı vermesi bu kalite anlayışının bir parçası.

## Slayt 6 — Tasarımı gereği yerel. Varsayılan olarak doğrulanabilir. — 4:15–5:00

Tracefold; belgeleri içe aktarma, yerel vektörleştirme, SQLite'ta saklama, kanıta erişim, kaynağa dayalı yanıt, kontrollü çeviri, güvenli ret ve iki dilli değerlendirmeyi tek bir çevrimdışı ürün deneyiminde birleştiriyor.

Kullanıcı atfın arkasındaki metni inceleyebiliyor. Eksik model veya başarısız denetim, gizli bir yedek yanıt yoluyla maskelenmiyor. Onarım denemeleri sınırlı; eşikler başarısız bir yanıtı geçirmek için gevşetilmiyor. Sonuçlar, hedefler ve bilinen sınırlar birbirinden ayrı sunuluyor.

Bugünkü iki dilli kalite iddiasının sınırı da açık: Northstar sözlüğü genişletilebilir, ancak başka bir alana geçildiğinde terimler yeniden gözden geçirilmeli ve gerçek model değerlendirmesi yeni belge kümesi üzerinde tekrar çalıştırılmalı. Tracefold'un özeti şu: Belgeler yerelde kalır, her iddia kanıtını taşır ve sistem bilmediği yerde durur. Teşekkür ederim.
