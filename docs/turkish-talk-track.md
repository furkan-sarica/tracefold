# Tracefold — 5 Dakikalık Türkçe Sunum Metni

Bu metin altı İngilizce slaytla uyumludur. Hedef süre yaklaşık 5 dakikadır. Köşeli parantez içindeki notlar sahne yönlendirmesidir; okunmaz.

## Slide 1 — Tracefold: Answers You Can Trace — 0:00–0:40

Merhaba, bugün size Tracefold'u göstereceğim. Tracefold, yüklediğimiz belgeleri cihazdan çıkarmadan yanıtlayan ve her iddiayı açılabilir bir kaynakla ilişkilendiren yerel bir RAG uygulaması.

Bu projede çözmek istediğim problem şu: Bir sohbet modeli çok akıcı konuşabilir, fakat belge içindeki cevabı gerçekten bulduğunu nasıl anlayacağız? Tracefold'un yaklaşımı, yalnızca güzel bir cevap üretmek değil; cevabın hangi belgeye, hangi bölüme ve hangi metin parçasına dayandığını görünür hâle getirmek. Bu nedenle sloganımız: “Answers you can trace.”

## Slide 2 — The Problem: Private Documents, Unverifiable Answers — 0:40–1:20

Klasik belge asistanlarında üç temel risk var. Birincisi gizlilik: belge ve soru buluta gönderilebilir. İkincisi halüsinasyon: model, kaynakta olmayan bir bilgiyi ikna edici biçimde uydurabilir. Üçüncüsü ise denetlenebilirlik: kullanıcı yanıtın dayanağını kolayca göremez.

Tracefold bu üç riski birlikte ele alıyor. Hem embedding hem de cevap modeli Microsoft Foundry Local üzerinden bu cihazda çalışıyor. Kaynak bulunmadığında cevap üretmeye zorlamak yerine açıkça “yeterli kanıt bulamadım” diyor. Kaynak bulunduğunda ise citation'a tıklayıp kullanılan gerçek pasajı görebiliyoruz.

## Slide 3 — Local RAG Architecture — 1:20–2:15

[Mimari diyagramını göster.]

Akış belge yüklemeyle başlıyor. FastAPI, PDF, DOCX, Markdown veya metin dosyasını ayrıştırıyor; başlık ve sayfa gibi kaynak bilgisini koruyarak yaklaşık iki yüz kırk tokenlık, bölüm sınırlarını aşmayan parçalara bölüyor. Her parça Foundry Local embedding modeliyle vektöre çevriliyor.

Metinler, metadata ve Float32 vektörler tek bir SQLite veritabanında saklanıyor. Kullanıcı soru sorduğunda soru da yerel olarak embed ediliyor, cosine similarity ile en yakın beş parça seçiliyor. Sabit ve versiyonlanmış abstention kuralı kanıtın yeterli olup olmadığına generation'dan önce karar veriyor. Kanıt yeterliyse yerel chat modeli yalnızca bu pasajlarla cevap üretiyor; citation kimlikleri son kez doğrulanıyor. Tüm sorgu yolu localhost ve yerel dosyalar içinde kalıyor.

## Slide 4 — Product Demo: From Documents to Evidence — 2:15–3:15

[Library ekranını, ardından Ask ekranını göster.]

Demo için Northstar Summer Lab adlı tamamen kurmaca bir programa ait sekiz belge hazırladım. Handbook, program, seyahat, konaklama, güvenlik, ekipman, davranış kuralları ve iletişim rehberi birbirini tamamlıyor.

Şimdi şu soruyu soruyorum: “Varış günü saat yirmiden sonra kampüse ulaşacağım; ne yapmalıyım ve kime ulaşmalıyım?” Bu basit bir anahtar kelime sorusu değil. Eval, top-five retrieval içinde seyahat, konaklama ve iletişim kaynaklarının bulunmasını kontrol ediyor. Son doğrulanmış cevap ise yalnızca gerçekten referans verdiği Travel ve Accommodation pasajlarını gösteriyor; gerekli telefon ve e-posta bu pasajlarda zaten yer alıyor.

[Citation panelini aç.]

Tracefold, Residence Duty Phone'u on dokuz otuzdan önce aramam ve Program Office'e yeni varış saatini bildirmem gerektiğini söylüyor. Her citation'ın dosyasını, metin parçasını ve benzerlik skorunu burada açabiliyorum.

## Slide 5 — Quality: Retrieval, Citations, and Refusal — 3:15–4:15

Kaliteyi sadece “cevap iyi görünüyor” diye değerlendirmiyoruz. Repoda sekizi cevaplanabilir, dördü bilinçli olarak cevaplanamaz toplam on iki soruluk bir eval seti var. Bunların içinde birden fazla belge gerektiren senaryolar da bulunuyor.

Ölçtüğümüz başlıca değerler retrieval Hit@5, çoklu kaynak kapsaması, citation geçerliliği, kaynakla desteklenen olgular ve negatif sorulardaki doğru refusal oranı. Ayrıca warm retrieval süresini ve tek seferlik model warm-up maliyetini kaydediyoruz. Ham ilk-token süresini release sonucu olarak sunmuyoruz; mevcut SSE yolu tam cevabı ve citation kimliklerini doğruladıktan sonra kelime parçalarını arayüze gönderiyor.

[Quality ekranında targets ve measured ayrımını göster.]

Burada önemli bir dürüstlük kuralımız var: hedefleri ölçülmüş sonuç gibi sunmuyoruz. Sonuç ancak model adı, kod durumu, donanım ve konfigürasyonla birlikte gerçek Foundry Local çalışmasından geldiyse “measured” olarak gösteriliyor. Son release çalışmasında on iki vakanın on ikisi geçti; Hit@5, beklenen kaynak kapsamı, citation geçerliliği ve negatif refusal yüzde yüz, warm retrieval p95 ise yaklaşık doksan altı milisaniye ölçüldü. Örneğin burs miktarı kaynaklarda yoksa sistem rakam uydurmak yerine standart refusal mesajını döndürüyor.

## Slide 6 — Why Tracefold — 4:15–5:00

Tracefold, istenen yerel RAG zincirinin tamamını çalışır bir ürün deneyimine dönüştürüyor: belge alma, chunking, yerel embedding, SQLite vektör saklama, cosine retrieval, kaynaklı generation, deterministic abstention, eval ve çevrimdışı çalışma.

Bunun üzerine üç önemli özellik ekliyor: kullanıcı citation'ın arkasındaki kanıtı gerçekten inceleyebiliyor; model veya runtime problemi gizli fallback ile maskelenmiyor; kalite hedefleri ile ölçülmüş sonuçlar birbirinden açıkça ayrılıyor.

Kısacası Tracefold'un iddiası, sadece çevrimdışı çalışan bir chatbot olmak değil. Özel belgeler üzerinde, kaynağı görülebilen, test edilebilen ve bilmediği yerde durabilen güvenilir bir cevap katmanı olmak. Teşekkür ederim.
