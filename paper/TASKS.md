# Tasks, contexts and data (exact)

Generated from the stored contexts (`data/<task>/c<id>/context.json`). Prompt = chat template + system message + user message `<context>\n\n<query>` (teacher) or `<query>` (student). The context is identical for every query of a context. Students train on the teacher's answers to the 500 training queries.

## Knowledge Retrieval
System: "You are a helpful assistant. Answer the question with a short phrase."

Train 500 (50 facts x 2 wordings x 5 framings), held-out 300 (50 x 2 other wordings x 3 other framings) -> 100 val + 200 test. Every fact is asked in training; test uses new wordings.

### Context 1 (c0)

| Name | Born | Hometown | Job | Colour | Pet |
|---|---|---|---|---|---|
| Nelost Shoshonel | 1991 | Ostzarbrook | archivist | maroon | dog Kapex |
| Nelka Vimnelost | 1987 | Rhopexwick | cartographer | coral | ferret Ostbel |
| Lodra Zarshorho | 1925 | Beldramere | stonemason | jade | tortoise Ostori |
| Drahal Lotenquo | 1962 | Shotrimere | translator | mauve | parrot Oriquo |
| Quorho Zarnelyel | 1899 | Mirvaswick | glassblower | cobalt | hedgehog Duncor |
| Tenpex Yelmirka | 1985 | Quodracrest | glassblower | jade | tortoise Belcor |
| Pexlun Shovimnel | 1972 | Kaorivale | falconer | jade | hedgehog Kayel |
| Orihal Rhopexhal | 1903 | Rhofencrest | botanist | turquoise | tortoise Nelmir |
| Tenlun Triostlun | 1920 | Trizarstead | shipwright | amber | parrot Triori |
| Corzar Vasoritri | 1995 | Shovascrest | clockmaker | indigo | dog Quocor |

### Context 2 (c1)

| Name | Born | Hometown | Job | Colour | Pet |
|---|---|---|---|---|---|
| Belquo Tenvascor | 1994 | Nelpexvale | locksmith | coral | cat Belbel |
| Belvim Gavlunost | 2004 | Haltricrest | puppeteer | magenta | parrot Gavquo |
| Halcor Vasfenmir | 1896 | Trilowick | glassblower | indigo | parrot Sholo |
| Vasost Corlunsho | 1928 | Ostgavvale | cartographer | indigo | goldfish Nelbel |
| Yelyel Yeldunsho | 1991 | Halrhoholt | puppeteer | cobalt | cat Mirquo |
| Shosho Lorhotri | 1998 | Lunlunholt | falconer | saffron | dog Lolun |
| Nellun Quotriquo | 1974 | Halvimholt | astronomer | jade | cat Loka |
| Ostlo Corpexost | 1910 | Shodunholt | perfumer | saffron | parrot Shobel |
| Tenvim Gavorifen | 1905 | Nelshoholt | puppeteer | cobalt | rabbit Drarho |
| Quohal Ostshohal | 1898 | Vimkawick | locksmith | jade | hedgehog Rhoori |

### Context 3 (c2)

| Name | Born | Hometown | Job | Colour | Pet |
|---|---|---|---|---|---|
| Dralo Kashomir | 1894 | Lunrhocrest | locksmith | mauve | tortoise Hallo |
| Mirlo Vimquoka | 1908 | Ostnelford | cheesemaker | turquoise | cat Lozar |
| Gavten Pexpexsho | 1970 | Vimyelholt | beekeeper | cobalt | ferret Dralun |
| Pexfen Corvimhal | 1913 | Ostnelbrook | botanist | indigo | iguana Pexost |
| Shodra Corzarfen | 1900 | Dramirstead | cheesemaker | indigo | dog Kavas |
| Gavka Fentenvim | 1901 | Nelyelford | perfumer | turquoise | tortoise Trihal |
| Zarvim Hallovim | 1937 | Rhotenford | falconer | charcoal | dog Lohal |
| Nelsho Quovascor | 1994 | Kashobrook | cartographer | coral | dog Quonel |
| Belbel Zaroridun | 1935 | Cornelmere | translator | charcoal | parrot Yelcor |
| Triori Pexorilun | 1956 | Lotenmere | violinist | charcoal | hedgehog Dunori |

## Label Mapping
System: "You are a helpful assistant. Reply with only the label." Queries: `Input: <article>\nLabel:`; 500 train / 100 val / 200 test, class-balanced, new articles (<= 60 words). Graded on the first word.

- c0: ag_news, 32 examples in context: World -> river, Sports -> stone, Business -> cloud, Sci/Tech -> lamp
- c2: dbpedia, 36 examples in context: Company -> kettle, Artist -> zebra, Athlete -> anchor, Building -> tulip, Animal -> comet, Film -> velvet
- c3: dbpedia, 36 examples in context: Educational institution -> cactus, Office holder -> harbor, Means of transportation -> pencil, Natural place -> falcon, Plant -> lemon, Written work -> jacket

## Mathematical Reasoning
System: "You are a helpful assistant. Solve the problem step by step, then give the final line as 'Answer: <answer>'." Queries: `Problem: What is the remainder when N is divided by 9?`, N random 7-9 digits, unique; 500 train / 300 val / 500 test.

## Full context text: knowledge c0

```
Here are facts about some people.
Nelost Shoshonel was born in 1991 in the town of Ostzarbrook. Nelost Shoshonel works as a archivist. Nelost Shoshonel's favorite color is maroon. Nelost Shoshonel has a dog named Kapex.
Nelka Vimnelost was born in 1987 in the town of Rhopexwick. Nelka Vimnelost works as a cartographer. Nelka Vimnelost's favorite color is coral. Nelka Vimnelost has a ferret named Ostbel.
Lodra Zarshorho was born in 1925 in the town of Beldramere. Lodra Zarshorho works as a stonemason. Lodra Zarshorho's favorite color is jade. Lodra Zarshorho has a tortoise named Ostori.
Drahal Lotenquo was born in 1962 in the town of Shotrimere. Drahal Lotenquo works as a translator. Drahal Lotenquo's favorite color is mauve. Drahal Lotenquo has a parrot named Oriquo.
Quorho Zarnelyel was born in 1899 in the town of Mirvaswick. Quorho Zarnelyel works as a glassblower. Quorho Zarnelyel's favorite color is cobalt. Quorho Zarnelyel has a hedgehog named Duncor.
Tenpex Yelmirka was born in 1985 in the town of Quodracrest. Tenpex Yelmirka works as a glassblower. Tenpex Yelmirka's favorite color is jade. Tenpex Yelmirka has a tortoise named Belcor.
Pexlun Shovimnel was born in 1972 in the town of Kaorivale. Pexlun Shovimnel works as a falconer. Pexlun Shovimnel's favorite color is jade. Pexlun Shovimnel has a hedgehog named Kayel.
Orihal Rhopexhal was born in 1903 in the town of Rhofencrest. Orihal Rhopexhal works as a botanist. Orihal Rhopexhal's favorite color is turquoise. Orihal Rhopexhal has a tortoise named Nelmir.
Tenlun Triostlun was born in 1920 in the town of Trizarstead. Tenlun Triostlun works as a shipwright. Tenlun Triostlun's favorite color is amber. Tenlun Triostlun has a parrot named Triori.
Corzar Vasoritri was born in 1995 in the town of Shovascrest. Corzar Vasoritri works as a clockmaker. Corzar Vasoritri's favorite color is indigo. Corzar Vasoritri has a dog named Quocor.
```

## Full context text: arbitrary_labels c0

```
Classify each input using the labels shown in these examples.

Input: Insurance Probe Hits Home in DC Region Some of the insurance brokers and carriers under investigation in New York Attorney General Eliot L. Spitzer #39;s probe of steering and bid-rigging are major players in the Washington area insurance market.
Label: cloud

Input: Update 3: Vodafone Unveils High-Speed Net Services Vodafone Group PLC launched high-speed Internet services for mobile phones across much of Europe on Wednesday, ending a wait of several years since the world #39;s biggest cell phone company spent billions for new wireless licenses at the height of the
Label: lamp

Input: At least eight dead from car bomb explosion near US base close to At least eight people are dead from a car bomb that exploded outside the gates of a US base in western Iraq. US military and hospital officials say around 48 people were injured near Ramadi
Label: river

Input: Yakubu earns Portsmouth victory Michael Brown of Tottenham Hotspur (left) blocks a free kick by Patrik Berger of Portsmouth (right) during the English Premiership soccer match at Fratton Park, Portsmouth, Oct. 18.
Label: stone

Input: US Retailers #39; Holiday Sales Rise; Wal-Mart Lags Behind US retailers #39; credit card sales rose more than 6 percent during the first weekend of the holiday shopping season. Same-store sales at Wal-Mart Stores Inc.
Label: cloud

Input: Wi-Fi emerges as emergency communications alternative in Fla. Wi-Fi hot spots emerged as a new emergency communications alternative in Florida in the aftermath of Hurricane Charley, with traffic at some outlets up 50.
Label: lamp

Input: Intel Keeps Its Eye on the Internet In a future with millions of users and devices taxing the Net, Intel-based servers will help out.
Label: lamp

Input: Summer of spam surges For spammers, it #39;s been a summer of love. Two newly issued reports tracking the circulation of unsolicited e-mails say pornographic spam dominated this summer, nearly all of it originating from Internet addresses in North America.
Label: lamp

Input: Former Predators Center Zholtok Dies Minnesota Wild right winger Sergei Zholtok of Latvia, rigth, battles for the puck ahead of Colorado Avalanche left winger Eric Messier, left, in the first period of game 7 of their Western Conference quarterfinal series in Denver in this April 22, 2003
Label: stone

Input: Australia #39;s Qantas calls halt to merger appeal process Australia #39;s main airline carrier, Qantas, says it won #39;t appeal the New Zealand High Court #39;s rejection of a proposed alliance with Air New Zealand.
Label: cloud

Input: Dining with the dolphin hunters Paul Kenyon confronts the drive hunt fishermen of Taiji in Japan and the intermediaries who sell the animals to dolphinariums.
Label: lamp

Input: SEC Investigating WorldCom Creditors Committee The Securities and Exchange Commission is investigating WorldCom Inc.'s official creditors committee, requesting reams of documents related to members' dealings as well as information they reviewed about WorldCom's health and prospects.
Label: lamp

Input: Unions Protest New Overtime Regulations Several hundred union members marched outside the Labor Department to protest new overtime pay regulations taking effect Monday, with two senators pledging to roll them back when Congress returns from recess.
Label: cloud

Input: Look who #39;s talking In promoting the disengagement plan, that confrontational man, Prime Minister Ariel Sharon, has already caused great suffering, anger and destruction - just look at the tremendous commotion already happening.
Label: river

Input: At Vermont, starting at point a 4-year lock One of the hallmarks of Tom Brennan #39;s 19-year run as Vermont men #39;s basketball coach has been his approach to filling the point guard position.
Label: stone

Input: Selection of jurors first step in trial EAGLE, COLO. -- Kobe Bryant #39;s trial began yesterday with 300 prospective jurors arriving at the county courthouse to fill out questionnaires, the first step toward choosing 12 people who will decide whether
Label: stone

Input: Marsh to Scrap Fees Spitzer Faulted  NEW YORK (Reuters) - Marsh   McLennan Cos. &lt;A HREF="http://www.investor.reuters.com/FullQuote.aspx?ticker=MMC.N target=/stocks/quickinfo/fullquote"&gt;MMC.N&lt;/A&gt;, the  world's largest insurance broker, on Tuesday said it will  reform and stop accepting fees that New York's attorney general  said amounted to bid-rigging.
Label: cloud

Input: Zee to challenge cricket board Zee Telefilms Limited, the country #39;s largest listed media firm, plans to appeal to the Supreme Court against a cricket board decision to ask two rival channels to broadcast the home test series against Australia, its lawyer said on Friday.
Label: stone

Input: Pfizer to Test Heart Safety of Celebrex Pfizer Inc. on Monday said a major new trial will test whether its popular Celebrex arthritis drug increases the risk of heart attacks, as seen with Merck and Co.
Label: cloud

Input: Venezuela audit 'confirms' result Venezuelan officials say first results of an audit of the vote on President Chavez's rule show no fraud.
Label: river

Input: Grains Found in Ga. Traced to Asteroid (AP) AP - A layer of quartz grains found in an east Georgia kaolin mine have been traced to the impact of a giant asteroid that crashed near the mouth of the Chesapeake Bay 35 million years ago.
Label: lamp

Input: The Week: Renault F1 Team between China and Japan Its been a busy week for the team with its members stretched across the globe. Heres an update of whats going on
Label: stone

Input: Pessimism over latest attempts to restart peace talks in Sri Lanka A high-level Norwegian delegation led by Foreign Minister Jan Petersen arrived in Sri Lanka this week as part of international efforts to save the countrys tottering peace process.
Label: river

Input: IBM lines up big daddies of Unix servers IBM plans to announce two new top-end Unix servers, sources familiar with its plans said, opening major new challenges to Sun and HP.
Label: lamp

Input: Google wins fight against Geico Google #39;s motion to dismiss legal proceedings brought against it by insurance company Geico was granted on Wednesday by a US District Court judge.
Label: cloud

Input: EU executive warns of fallout from Belarus vote The European Commission warned Tuesday that a vote in Belarus paving the way for autocratic President Alexander Lukashenko to retain power will damage the ex-Soviet republic #39;s relations with the EU.
Label: river

Input: 47 dead in Baghdad market blast At least 47 people were killed at a Baghdad market today in an explosion that was believed to have been targeted at a nearby police station.
Label: river

Input: Malaysia's Anwar Loses Bid to Lift Political Ban  PUTRAJAYA, Malaysia (Reuters) - Malaysian rebel politician  Anwar Ibrahim failed on Wednesday in a final court bid to erase  his criminal record, leaving a royal pardon as his only option  for a swift return to politics.
Label: river

Input: Once left for dead, Astros streak back into wild-card contention Everything #39;s going the Astros #39; way these days. They #39;re scoring a ton of runs, their schedule looks favorable, their bullpen is starting to kick in -- even their starting pitching, which consists of Roger Clemens
Label: stone

Input: NYSE suspends Gold Fields trading The South African company Gold Fields Limited announced Friday that the New York Stock Exchange has suspended trading in its stock.
Label: cloud

Input: Blair to Have Heart Procedure, Will Serve 3rd Term  LONDON (Reuters) - British Prime Minister Tony Blair will  go into hospital Friday for a procedure to correct heart  palpitations but will be fully functioning and back at work  next week, his office said in a statement.
Label: river

Input: Keane nets 24th goal in win over Croatia DUBLIN: Tottenham striker Robbie Keane extended his Irish goal-scoring record with his 24th strike for his country as Ireland beat Croatia 1-0 in a friendly international on Tuesday.
Label: stone
```

## Full context text: math_methods c2

```
Here are worked examples of how to solve this kind of problem.

Problem: What is the remainder when 487575745 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 4 + 8 + 7 + 5 + 7 + 5 + 7 + 4 + 5 = 52
52 = 9 × 5 + 7, so the remainder is 7.
Answer: 7

Problem: What is the remainder when 129457627 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 1 + 2 + 9 + 4 + 5 + 7 + 6 + 2 + 7 = 43
43 = 9 × 4 + 7, so the remainder is 7.
Answer: 7

Problem: What is the remainder when 697341842 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 6 + 9 + 7 + 3 + 4 + 1 + 8 + 4 + 2 = 44
44 = 9 × 4 + 8, so the remainder is 8.
Answer: 8

Problem: What is the remainder when 774911182 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 7 + 7 + 4 + 9 + 1 + 1 + 1 + 8 + 2 = 40
40 = 9 × 4 + 4, so the remainder is 4.
Answer: 4

Problem: What is the remainder when 506295702 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 5 + 0 + 6 + 2 + 9 + 5 + 7 + 0 + 2 = 36
36 = 9 × 4 + 0, so the remainder is 0.
Answer: 0

Problem: What is the remainder when 113534217 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 1 + 1 + 3 + 5 + 3 + 4 + 2 + 1 + 7 = 27
27 = 9 × 3 + 0, so the remainder is 0.
Answer: 0

Problem: What is the remainder when 918394037 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 9 + 1 + 8 + 3 + 9 + 4 + 0 + 3 + 7 = 44
44 = 9 × 4 + 8, so the remainder is 8.
Answer: 8

Problem: What is the remainder when 821055758 is divided by 9?
A number has the same remainder mod 9 as the sum of its digits.
Digit sum: 8 + 2 + 1 + 0 + 5 + 5 + 7 + 5 + 8 = 41
41 = 9 × 4 + 5, so the remainder is 5.
Answer: 5
```
