# 生成語言學與人工類神經網路之間的鴻溝。The gap between generative linguistics and artificial neural network

<figure><img src=".gitbook/assets/圖片 (4).png" alt=""><figcaption></figcaption></figure>

<figure><img src=".gitbook/assets/圖片 (5).png" alt=""><figcaption></figcaption></figure>

### ‘50-‘60 symbolic AI (Good-old-fashioned AI, aka GOFAI)

* 博藍尼悖論 (Polanyi’s paradox)：「我們懂的事情，比我們能表達出來的更多。」
  * 博藍尼悖論不只限制我們能告訴另一個人的事情，也對我們賦予機器智慧的能力，設下根本的限制，限制了機器能有效執行的活動。
* 舉例來說，以下三個中文句子中的「什麼」的意思都不一樣 (你翻成英文會更清楚)，但如果要求你「講清楚」這三種意思的分佈規則與邏輯，你會說不太清楚。

1. ) The three-way ambiguity of Chinese「什麼」：
   1. 約翰吃了<mark style="color:blue;">**什麼  ⇒ WHAT**</mark>
   2. 約翰可能吃了<mark style="color:blue;">**什麼 ⇒ SOMETHING**</mark>
   3. 約翰<mark style="color:blue;">**什麼**</mark>都吃 <mark style="color:blue;">**⇒ WHATEVER**</mark>

* 機器程序是由人類撰寫出來的，當人類不知道一個問題的解答時，機器同樣不能解決人類無法回答的問題。
  * 另一個問題是當時電腦的算力尚未提升、儲存空間小、數據量不足。早期的人工智慧研究聚焦在邏輯推論的方法，專注於模仿人類推理過程的思考模式，需要百分之百確定的事實配合，實務上應用困難。
  * 但在 LLM 橫空出世以後，我們發現就算沒有真的回答問題，只要機器程序生成一系列好像有道理的前言，人類就會誤以為機器解決了人類無法回答的問題。
* Symbolic AI 的時代，在北美，「結構語言學」是聲量最大的語言學理論框架。
* 彼時，Chomsky 開創的生成語言學才剛開始幾年，對主流的 AI 方法而言沒有存在感。也沒有任何學者或是工程師利用生成語言學理論設計 AI 系統 (直到 2011 年以後，才開始有 AI/NLP 系統是依生成語言學的理論設計)。可以說「1950\~1960年代的生成語言學並沒有實質影響力，因此 symbolic AI 的失敗不關 Chomsky 的事」。
* 但當時「依傳統語言學理論設計 AI/NLP 系統的失敗」已經傳開，因此人們普遍的記憶是「語言學做 AI 是失敗的」。
* 再之後，到了 80年代以後，屬於現代語言學的生成語言學派成為現代語言學方法的主流時，但很多不知道這件事，而只記得「Symbolic AI 的失敗，是由於語言學的緣故」的人，往往直接把這段失敗歸因到 Chomsky 引領的生成語言學上面。

### ‘80-’90 expert system (feature engineering) <a href="#toc191050511" id="toc191050511"></a>

* 把人的所有知識放進電腦。
* 但 1960-1970 晚期的生成語言學真的不好用 (in terms of NLP)～
* 早期生成語言學為了捕捉語言現象，寫了一堆詞組結構規則如 )，區分合法句以及不合法句，各種句型以及句型變化都寫了對應的規則，系統相當龐雜、細碎、統整性不足，追求效率的電機資工學者對這類專家系統敬謝不敏也是剛好而已。

2. ) Phrase Structure Grammar coupled with transformational rules:
   1. S—> NP (Aux) VP
   2. NP —> (Det) N (PP)
   3. VP —> (AdvP) V (NP) (NP) (PP)
   4. Passivization rule
   5. _Wh_-question rule
   6. _Yes_/_no_-question rule
   7. Topicalization rule
   8. Relativization rule
   9. ……
3. Three major downsides of such a rule-based system in NLP:
   1. Labor-intensive: 由人撰寫，耗費人力。
   2. Subjective & inconsistent judgement: 主觀意識差異很大，導致寫的規則有衝突。
   3. High-maintenance: 由於規則容易互相衝突，加新規則要非常小心，系統變大之後很難維護。

* 但是，生成語言學家並不是一群不會進步的笨蛋。
* 但做 AI 的人從此就認定了語言學是一灘死水。
* 科學演進本就需要時間：地心說到日心說，托勒密到哥白尼。以地心說為基礎的年曆，經過了 1300 多年的誤差，累計已經和地球圍繞太陽運動的實際情況差了 10 天左右，用來指導農時經常會誤事。直到 1543 年，波蘭教士哥白尼發表《天體運行論》才有改變。
* 現代語言學從 1957 開始到 2025 也才快 70 年。

### 2010 \~ Deep learning (stay data/energy hungry, stay foolish) <a href="#toc191050512" id="toc191050512"></a>

* 放棄了專家系統，不再關注生成語言學後續的發展。
  * 一了百了，把人所有看見的輸入/輸出 (<mark style="color:red;">**WHAT**</mark>) 放進電腦，放棄表徵 (不再研究<mark style="color:red;">**HOW**</mark>)、放棄因果邏輯 (可解釋性 AI 失敗後，也放棄對 <mark style="color:red;">**WHY**</mark> 的探究)
  * 成功三要素：<mark style="color:blue;">**Big Data**</mark>，<mark style="color:blue;">**GPU**</mark>以及<mark style="color:blue;">**深度學習框架**</mark>。
* 缺點
  * 耗能成本高 \[1]
  * 黑盒子 (opaque)：你不知道內部發生什麼事，不確定調教結果會不會保證好。
  * 穩健性 (robustness)：本質就是擲骰子，今天對，明天可能錯。\[2]
  * 幻覺 (hallucination)：一本正經說幹話。
  * 資料永遠不夠 (好)：出了錯常訴諸於資料不夠 (或資料不夠好)。
* 托勒密難題：地心說模型雖然不正確，但能讓人順利在海上航行並預測日月食，這代表理論模型就算一直不正確，但人們會被實用面的充分性所誤導。
* 同樣的，LLM 的運作底層邏輯不是真的模擬人的心智運作，但因為真的很實用，所以一般人就當然不會在乎 LLM 是否是正確的心智模型，直到它影響到你的利益：隱私，名譽，金錢，生命。
* LLMs 表現出驚人的能力和實用性，但此工程成就並不削弱生成語言學在認知科學發展上的論證效力與價值。作為科學家，有一說一，有幾分證據說幾分話，這是基本科學素養，實際研究中我們可以看到至少有三大原因告訴我們 LLM 並不能代表人類的語言知識系統。

1. 人類孩童與 LLMs 在語言學習機制上也有根本上的不同。
   1. LLMs 通過對海量訓練數據以及 GPU 高速運算，進行迭代優化來運作。然而，心理學研究中發現孩童並沒有這種學習機制，孩童記憶、注意力以及其他心智計算能力都有其認知限制。
2. LLMs 的訓練仰賴於深度學習，而深度學習成功的核心技術為反向傳播演算法 (backpropagation)。
   1. 但學者發現大腦的實際運作無法使用反向傳播演算法來傳遞與更新資訊，因此將人類大腦與 LLMs 做類比，在神經生物的層面上是不具可行性的。
   2. 在機器學習中，反向傳播是透過數學計算 (微積分 & 矩陣運算) 來調整參數，以便下一次預測更準確。在人類大腦中，學習是透過神經元之間的電訊號和化學變化來強化或削弱特定的神經連結 (突觸可塑性)。
   3. 機器的反向傳播是透過微積分計算出每個參數該如何改進的方式 (可以在幾毫秒內更新幾百萬個參數)。人類大腦並沒有在神經元之間進行微積分計算，因為大腦並不是數學運算機，人腦的學習是生物過程，涉及神經突觸的增強或削弱，這個過程相對較慢，可能需要幾秒、幾分鐘，甚至幾天。
3. LLMs 與人類接觸到的語言刺激的量體差異極大。
   1. 更精確地說，OpenAI 的 ChatGPT 3.5 需要人類九千輩子才能讀得完的文本量才能訓練完成\[3]。與 LLMs 不斷攀升的資料量需求相比，一個成功學習母語 (以英語為例) 的健康人類孩童，一年能夠接觸到資料量頂多只有 1,000 萬字，五歲左右的孩童接觸到的資料量也累計大約 3650 萬字。
   2. Linzen & Baroni (2021) 提到，這個巨幅資料量差異導致只要 LLMs 高度依賴海量資料，那麼無論 LLMs 的語文生成表現如何驚人，對於認知科學的根本理解與實際貢獻都有侷限性\[4]。
   3. 基於這個語言刺激量的巨幅差異，Yedetore 等人 (2023) 從 CHILDES 語料庫，選取真實人類孩童接受到的語言範例，訓練類神經網路模型，更重要的是，訓練量體與人類孩童對齊。他們發現在對齊人類孩童訓練的條件下，類神經網路模型無法成功發展出與語言結構相關的判斷能力\[5]。
   4. 除此之外，van Shijndel 等人 (2019) 的研究發現大幅提升參數量或是訓練資料，並無法有效地提升 LLMs 針對語言結構相關的判定能力。以上兩個類別的研究結果都指向同一結論：要提升 LLMs 與語法結構相關的判別正確率，訓練資料或是參數量的提升並不是實際的做法，植入語言結構表徵才是關鍵的變因 (如 Euguejard et al. 2017; Kuncoro et al. 2018; 2019; Wilcox et al. 2019; McCoy et al. 2020)。

### Conclusion <a href="#toc191050513" id="toc191050513"></a>

* AI 領域自 1940 年代開始以來，在 2022 年 11 月藉由 ChatGPT 的驚人工程成就帶來了廣泛甚至是變革性的實用性。

1. 站在認知科學的角度，語言「科學」與語言模型「科技」還是有很大的本質差異。
   1. Kodner 等人 (2023):「LLMs 之於人類自然語言」就如同於「飛機之於鳥類的飛行機制」。
   2. Leivada et al. (2024: 1): 雖然 LLMs 是好的語言表達工具，但這並不代表這個工具的運作方式與架構與人類的語言知識系統相同。\[6]
   3. Chomsky: There is nothing wrong with making USEFUL things, but the project of trying to UNDERSTAND the mind is different.
2. 任何 AI 模型都涉及三個元素：資料、運算力、演算法。
   1. 過去十年因為網路資料的量很大，取得又方便，所以主流的做法都假設資料可以繼續增加，運算力可以繼續增加，在不斷 scale up 的條件下發展 AI，但台灣其實沒有這個條件。
   2. 相較之下，2025 農曆年間中國團隊釋出的 DeepSeek，就可以看成是在「資料量有限，運算力因為晶片禁令而有限」的情況下，改在演算法這個角度下功夫的成果。我們先不談 DeepSeek 一些有爭議的地方，DeepSeek 讓我們可以看到在演算法的精進上，是可以在更少的資料、運算量受限的條件下做出好的效果。
   3. 這個條件其實和台灣很像！台灣使用的繁體中文在「大語言模型」的眼裡，是個資料不足的小語種。至於運算力需要的顯卡，雖然我們不受晶片禁令的影響，雖然黃仁勳是在台灣出生，但 nVidia 畢竟是美商，我們要買顯卡也是要在 OpenAI, Google, Microsoft 後面排隊的，所以我們也不能假設運算力可以不斷往上加。相關電費的部分我就不多說了。
   4. 從這個角度來理解，語言學觀點的 NLP 做法，其實剛好和台灣的條件相符。語言學做法的精華就是「演算法」，因為語言學研究的重點，自 1957 以來，就是「人類幼童是怎麼學會母語的」，與 LLM 的訓練資料相比，人類幼童接觸過的資料很少，大腦還在發育，所以運算力也有限。但只要像人類幼童一樣，有一個很好的演算法，他就能很快地同步發展語言與智力。生成語言學家一直在做的事就是針對不同的語言現象，提出明確可驗證的演算法。
   5. 因此，回到 AI 三要素：資料、運算力、演算法。台灣做語言模型，在資料以及運算力都不能算得上充沛，因此，精進演算法似乎是一條相對 promising 的道路，或至少值得嘗試看看，這就是語言學能夠提供價值的地方。如果我們願意採用混成式的 AI 發展，也就是所謂的 neuro-symbolic AI，不同的問題採用不同的做法，也就是「資料量較少的問題，採用語言學方法；資料量足夠的問題，套用電資領域的資料模型方法」，那麼我們就什麼題目都能做，而不需要遇到 AI 發展的問題就假定問題是在「資料不足」或「算力不足」這兩個障礙上。
3. Hybrid neuro-symbolic AI
   1. 混合式人工智能模型不只可以優化 NLP 任務，Google DeepMind 研發團隊也運用此類設計思維建立了 AlphaProof 以及 AlphaGeometry (Trinh & Luong 2024)，兩者結合後的數理推論解題能力可以達到數學奧林匹雅競賽銀牌等級，研究團隊的文章提到：
   2. 「AlphaGeometry 是一種神經-符號系統，由神經語言模型和符號推理引擎組成，兩者協同合作以證明複雜的幾何定理。類似於『快思與慢想』的概念，一個系統提供快速且『直觀』的思路，而另一個系統則進行更審慎且理性的決策。」 \[7]
   3. ‘80 後期開始的生成語言學 (極簡方案，Minimalist Program)，已經將原本繁瑣的規則系統精練再精煉，濃縮再濃縮，詞組結構就只剩下單一的架構與運作原則，句法規則毫不留情地被去蕪存菁極簡化，原本諸多龐雜的句法規範，也都「外包」給語意以及音韻介面的要求了。這樣的語言學理論架構便有潛力符合上述系統二的要求。
   4. 如 Joe Pater 於 2019 年在期刊 _Language_ (美國語言學會出版的指標性期刊) 刊出的文章所述，生成語言學 (Chomsky 1957) 與類神經網路 (Rosenblatt 1957) 這兩個領域應該是互補而非互斥。
   5. 計算語言學家Tal Linzen (2019) 也大力提倡這個互補共好的研究願景，他認為生成語言學研究的語言實證現象，對於深度學習模型的表現，提供了有效的評估標準。\[8]
   6. Vázquez Martínez 等人 (2024) 一文最後一段也提到，最近 30 年來，生成語言學以及類神經網路這兩個領域的目標其實是一致的，兩者都致力於降低專屬於語言使用的原則與限制，兩個領域的合作共識應該建立於描述適切性以及解釋適切性的相同目標與定義。

* Adnan Darwiche (Former chair of UCLA’s CS program):
  * “We need a new generation of AI researchers who are well versed in and appreciate classical AI, machine learning, and computer science more broadly, while also being informed about AI history.”
  *

      <figure><img src=".gitbook/assets/圖片 (6).png" alt=""><figcaption></figcaption></figure>

***

1 人類的大腦耗能約 25 瓦，AI 伺服器動輒耗能 6000 瓦以上。2023 年，單就 OpenAI 的 ChatGPT 的耗電量，就足以供全美家庭用電兩年之久。另外，Meta 開發的 Llama 3.1 405B 使用的硬體設備為16,000 個NVIDIA H100 Tensor Core GPUs，價格超過六億美元。

2 擁護 LLMs 的人，針對穩健性的問題，為 LLMs 開脫的常見論述是「人也會犯錯」。這個思路是倒果為因，科學家建立「科技工具」，就是希望能信賴這個科技工具，把業務外包給它。舉例來說，人類計算數字太慢且可能出錯，因此我們設計了計算幾，可以快速且穩健地處理數字計算，我們可以放心地把數字計算這個業務外包給計算機，今天若是一台計算機不可靠，有時會犯錯，我們不會說「人算數也會犯錯啊～」來為計算機說情。同樣的邏輯與要求，我們也應該放在現在的 AI 發展上，更何況 AI 發展與計算機的發展成本根本是不同量級的投入，前者對我們的日常生活影響更深，因此對前者我們應該更嚴格，而不是它出錯時，我們就說「人也會犯錯啊，LLM 剛開始，它還只是個孩子啊!」。

3 GPT-4 需要 1.8 兆個訓練參數，10 兆字的資料庫；Meta 開發的 Llama，根據官方給出的數據，其訓練資料包含超過 15 兆個字元，這幾乎可以說是網際網路 (Internet) 出現以來所有的文本資料了。

4 李飛飛博士在她的個人自傳「AI 科學家李飛飛的視界之旅」(中文版於 2023 年，由天下文化出版) 中作出以下陳述：「AI 在臺灣常翻譯成『人工智慧』。但在目前這個時期，並不足以『人工智慧』的中文意義來表達現在 AI 所能做的事，且易引起不必要的恐懼。所以，本書將譯 AI 為『人工智能』。AI 目前仍與人類的智慧有相當大的差距，它們雖然能高速運算，但若因此認為 AI 已達到某種智慧程度，那可能太過抬舉而不符現實。

5 Warstadt 等人 (2023) 發起了所謂的 BabyLM 挑戰，要求參加者的訓練資料低於一億字，以此模擬真實人類孩童在成長過程中接受到的刺激量體。Amariucai 與 Warstadt (2024) 在此限制下，運用多模態刺激 (視覺加上文字) 訓練 BabyLM，他們的實驗結果是多模態刺激沒有顯著幫助。因此，不論是 LLMs 或是 BabyLM，若是訓練方式與目標限於詞元向量的統計分佈而不著眼於語法結構，即使使用多模態刺激也不會有顯著正面的幫助。

6 原文： “good tools are not synonymous with good models that provide faithful representations of the target system”。

7 原文：”AlphaGeometry is a neuro-symbolic system made up of a neural language model and a symbolic deduction engine, which work together to find proofs for complex geometry theorems. Akin to the idea of ‘thinking, fast and slow’, one system provides fast, ‘intuitive’ ideas, and the other, more deliberate, rational decision-making.”

8 原文 “the theoretical distinctions proposed by linguists provide a yardstick by which the performance of deep-learning systems can be measured" (Linzen 2019: e99)。
