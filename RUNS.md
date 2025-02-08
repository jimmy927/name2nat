1% 3 iter

python eval.py --gt nana/test.tgt --pred test.pred
precision@1=33908/111368=30.4
precision@2=45338/111368=40.699999999999996
precision@3=53620/111368=48.1
precision@4=60536/111368=54.400000000000006
precision@5=66473/111368=59.699999999999996

2025-02-06 21:41:58,289  - 0 epochs without improvement
2025-02-06 21:41:58,289 saving best model
2025-02-06 21:41:58,296 ----------------------------------------------------------------------------------------------------
2025-02-06 21:43:08,081 epoch 20 - iter 626/6260 - loss 1.62834172 - time (sec): 69.78 - samples/sec: 1148.22 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:44:17,426 epoch 20 - iter 1252/6260 - loss 1.63203385 - time (sec): 139.13 - samples/sec: 1151.85 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:45:26,210 epoch 20 - iter 1878/6260 - loss 1.63362911 - time (sec): 207.91 - samples/sec: 1156.17 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:46:35,533 epoch 20 - iter 2504/6260 - loss 1.63452536 - time (sec): 277.24 - samples/sec: 1156.09 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:47:44,103 epoch 20 - iter 3130/6260 - loss 1.63808668 - time (sec): 345.81 - samples/sec: 1158.57 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:48:52,568 epoch 20 - iter 3756/6260 - loss 1.63779263 - time (sec): 414.27 - samples/sec: 1160.51 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:50:01,551 epoch 20 - iter 4382/6260 - loss 1.63920349 - time (sec): 483.25 - samples/sec: 1160.66 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:51:10,765 epoch 20 - iter 5008/6260 - loss 1.64014726 - time (sec): 552.47 - samples/sec: 1160.29 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:52:19,602 epoch 20 - iter 5634/6260 - loss 1.64067635 - time (sec): 621.31 - samples/sec: 1160.70 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:53:28,617 epoch 20 - iter 6260/6260 - loss 1.64063965 - time (sec): 690.32 - samples/sec: 1160.65 - lr: 0.100000 - momentum: 0.000000
2025-02-06 21:53:28,617 ----------------------------------------------------------------------------------------------------
2025-02-06 21:53:28,617 EPOCH 20 done: loss 1.6406 - lr: 0.100000
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1739/1739 [00:44<00:00, 38.98it/s]2025-02-06 21:54:14,267 DEV : loss 1.5914212465286255 - f1-score (micro avg)  0.557
2025-02-06 21:54:23,039  - 0 epochs without improvement
2025-02-06 21:54:23,039 saving best model
2025-02-06 21:54:23,054 ----------------------------------------------------------------------------------------------------
2025-02-06 21:54:23,054 Loading model from best epoch ...
2025-02-06 21:54:23,057 [b'<unk>', b'a', b'\xe2\x96\x81', b'e', b'r', b'n', b'i', b'o', b'l', b's', b't', b'h', b'u', b'd', b'm', b'c', b'y', b'g', b'M', b'k', b'S', b'A', b'B', b'J', b'b', b'C', b'v', b'R', b'H', b'G', b'D', b'L', b'P', b'K', b'p', b'T', b'f', b'w', b'z', b'W', b'F', b'E', b'.', b'N', b'V', b'I', b'-', b'O', b'j', b'\xc3\xa9', b'Y', b'x', b'\xc3\xa1', b'Z', b'q', b'\xc3\xad', b',', b'U', b"'", b'\xc3\xb3', b'Q', b'\xc3\xbc', b'\xc3\xb6', b'\xc5\xa1', b'\xc3\xa7', b'\xc4\x87', b'1', b'X', b'\xc3\xa8', b'\xc3\xb8', b'\xc4\x8d', b'\xc5\x8d', b'\xc3\xba', b'\xc3\x89', b'\xc4\x83', b'\xc3\xb1', b'\xc3\x81', b'\xc3\xa4', b'2', b'\xc3\xa3', b'&', b';', b'\xc3\xab', b'\xc5\x99', b'\xc5\xa0', b'\xc3\xbd', b'3', b'\xc5\xab', b'\xc5\xbe', b'\xc3\xaf', b'\xc8\x99', b'\xc3\xb4', b'\xc4\x97', b'\xc3\xa5', b'\xc4\x9b', b'\xc3\xa2', b'\xc8\x9b', b'4', b'\xc4\x81', b'\xc3\x9f', b'\xc3\xa6', b'\xc3\x93', b'\xc3\xaa', b'\xc5\x91', b'\xc3\xa0', b'\xc5\x82', b'\xc3\xae', b'5', b'\xc3\x98', b'\xc4\x8c', b'\xe1\xbb\x85', b'\xc4\x90', b'6', b'\xc4\xb1', b'\xc3\xb2', b'7', b'\xc5\x86', b'\xc5\x8c', b'\xc5\xbd', b'\xc8\x98', b'\xc6\xb0', b'8', b'\xc4\xab', b'0', b'\xc3\x96', b'\xc5\x9f', b'\xc3\xb5', b'\xc5\x88', b'\xc4\x9f', b'9', b'\xe1\xbb\x8b', b'\xc6\xa1', b'\xc3\x87', b'\xc4\x93', b'\xe1\xba\xa1', b'\xca\xbb', b'\xe1\xba\xa7', b'\xc5\xaf', b'\xc5\x84', b'\xc4\x91', b'\xc4\xbc', b'\xc3\xb0', b'\xc3\xb9', b'\xc3\x85', b'\xc3\xac', b'\xc5\x9e', b'\xe1\xbb\x8d', b'\xe1\xba\xbf', b'\xc4\xbd', b'\xc4\xb0', b'\xe1\xbb\x87', b'\xe2\x80\x99', b'\xc5\xa9', b'\xe1\xbb\x93', b'\xc3\x9c', b'\xc8\x9a', b'\xe1\xba\xa3', b':', b'\xc5\xa5', b'\xe1\xba\xa5', b'\xc4\x99', b'\xc3\x8d', b'\xe1\xbb\xb3', b'\xc3\x82', b'\xc3\x86', b'\xe1\xbb\xa9', b'\xc5\xb1', b'\xc4\xbe', b'\xe1\xba\xad', b'\xc5\x93', b'\xc4\x86', b'\xc5\x98', b'\xe1\xbb\x9d', b'\xe1\xbb\x81', b'\xc3\xbf', b'\xe2\x80\x93', b'\xc5\x81', b'\xc4\xa3', b'\xe1\xba\xb7', b'\xc5\xbc', b'\xc4\x80', b'\xc3\x9a', b'\xe1\xbb\xaf', b'\xe1\xbb\x91', b'\xe1\xbb\xa5', b'\xc3\x88', b'\xc3\xbb', b'\xe1\xba\xaf', b'\xc5\x9b', b'\xc4\x85', b'\xc4\xb7', b'\xc5\xa3', b'\xe1\xbb\x97', b'\xc4\x8e', b'\xe1\xbb\xa3', b'\xc5\xbb', b'\xc4\xa9', b'\xc4\xb6', b'\xe1\xbb\xad', b'\xc4\x8f', b'\xe1\xbb\xb9', b'\xe1\xbb\x99', b'\xc3\x80', b'\xc5\x9a', b'\xc5\x85', b'\xc4\x92', b'\xca\xbf', b'\xe1\xba\xa9', b'!', b'\xe1\xba\xb1', b'\xe1\xbb\x83', b'\xe1\xbb\xab', b'\xe1\xbb\x95', b'\xe1\xb9\xa3', b'\xc3\x91', b'\xc3\x95', b'\xe1\xbb\x9b', b'\xe1\xbb\xb1', b'\xc9\x99', b'\xc4\xa0', b'\xc3\x9e', b'/', b'\xc5\xba', b'\xe1\xb8\xa5', b'\xc4\xa2', b'\xc7\x8e', b'\xe1\xb9\xad', b'\xe1\xbb\xa7', b'\xc3\x9d', b'\xe2\x80\x9c', b'\xe2\x80\x9d', b'\xe1\xb9\x87', b'\xc3\x84', b'\xe1\xbb\x89', b'\xc5\xad', b'\xe1\xb9\xac', b'$', b'\xc4\x95', b'\xe1\xb8\xa4', b'\xca\xbe', b'\xca\xbc', b'\xc5\xa2', b'`', b'\xc3\x90', b'\xe2\x80\x98', b'(', b'\xc4\x9e', b'+', b'\xe1\xb9\x83', b'\xc5\x90', b'\xc5\xb7', b'\xc5\xaa', b'\xca\xbd', b'\xc5\xa4', b'\xe1\xbb\x9f', b'\xc7\x83', b'\xc4\x8a', b'*', b'\xe2\x80\x91', b'\xc5\x92', b'\xd1\x90', b'\xe1\xb8\xb1', b'@', b'\xe1\xb8\xa0', b'\xe1\xb8\x8d', b'\xe1\xb9\x85', b'\xe2\x88\x92', b'\xc3\xbe', b'\xc5\xb5', b'\xc5\xb3', b'\xc4\x84', b'\xc4\xaa', b'\xc4\xbb', b'\xc2\xb7', b'\xc3\x92', b'\xc3\x8a', b'\xc5\x8f', b'\xe2\x82\xa9', b'\xe1\xba\xab', b'\xe1\xbb\xa1', b'\xe1\xbb\xb7', b'\xe1\xbb\xb5', b'\xe1\xba\xa8', b'\xe1\xbb\x8c', b'\xe1\xb8\xb2', b'\xca\x89', b'\xc5\xb9', b'\xd9\x90', b'\xd1\x97', b'\xc7\xab', b'\xc4\xa1', b'\xd9\x80']
2025-02-06 21:54:23,057 vocabulary size of 292
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1392/1392 [00:30<00:00, 45.04it/s]2025-02-06 21:54:54,778
Results:
- F-score (micro) 0.5567
- F-score (macro) 0.2625
- Accuracy 0.5567

By class:
               precision    recall  f1-score   support

     American     0.5166    0.8266    0.6359     24359
      English     0.3413    0.4008    0.3687      7667
       German     0.6422    0.6965    0.6683      4139
       French     0.6852    0.7341    0.7088      4088
      Italian     0.7427    0.8462    0.7911      2842
      British     0.3614    0.0467    0.0827      4692
       Indian     0.6168    0.7969    0.6954      2240
   Australian     0.2609    0.0399    0.0693      4056
      Russian     0.7041    0.8042    0.7509      2181
     Japanese     0.9053    0.9652    0.9343      2071
     Canadian     0.3190    0.0109    0.0212      3379
        Dutch     0.7168    0.5622    0.6302      1535
    Norwegian     0.7508    0.7530    0.7519      1312
    Argentine     0.2908    0.5071    0.3697       913
    Brazilian     0.5348    0.5672    0.5505      1153
      Mexican     0.3463    0.4120    0.3763       864
      Chinese     0.8109    0.8426    0.8264       921
       Korean     0.9094    0.9106    0.9100       783
        Irish     0.4434    0.1234    0.1931      1175
        Czech     0.7901    0.7394    0.7639       733
    Hungarian     0.8942    0.7877    0.8376       730
        Greek     0.8211    0.7648    0.7919       642
     Romanian     0.7949    0.7421    0.7676       632
      Iranian     0.6086    0.7713    0.6804       494
     Austrian     0.4583    0.0844    0.1426       912
      Belgian     0.6561    0.2394    0.3508       781
    Ukrainian     0.7851    0.5977    0.6787       599
    Pakistani     0.3820    0.6150    0.4713       400
      Israeli     0.5829    0.4167    0.4860       540
   Portuguese     0.5646    0.4052    0.4718       464
     Nigerian     0.4961    0.5053    0.5007       376
    Bulgarian     0.7887    0.7179    0.7517       390
     Egyptian     0.5000    0.5357    0.5172       308
        Welsh     0.3939    0.0743    0.1250       525
         Thai     0.7509    0.7168    0.7335       286
       Slovak     0.6038    0.5587    0.5804       281
    Malaysian     0.4873    0.4122    0.4466       279
   Indonesian     0.5029    0.3412    0.4065       255
     Ghanaian     0.5944    0.4332    0.5012       247
     Filipino     0.3962    0.0561    0.0984       374
     Albanian     0.5769    0.5634    0.5701       213
   Lithuanian     0.9157    0.7443    0.8212       219
    Taiwanese     0.6800    0.5613    0.6150       212
     Armenian     0.6852    0.5578    0.6150       199
      Latvian     0.8773    0.7772    0.8242       184
        Saudi     0.4270    0.4872    0.4551       156
     Algerian     0.4190    0.5068    0.4587       148
   Belarusian     0.6200    0.2756    0.3815       225
  Montenegrin     0.3592    0.7400    0.4837       100
      Chilean     0.1364    0.0106    0.0197       282
     Estonian     0.6281    0.4222    0.5050       180
    Colombian     0.2381    0.0388    0.0667       258
  Bangladeshi     0.4286    0.3725    0.3986       153
      Catalan     0.5577    0.3791    0.4514       153
       Kenyan     0.4773    0.2530    0.3307       166
     Moroccan     0.3190    0.2721    0.2937       136
    Uruguayan     0.0000    0.0000    0.0000       226
  Azerbaijani     0.6545    0.6261    0.6400       115
        Iraqi     0.2170    0.2000    0.2081       115
   Venezuelan     0.0000    0.0000    0.0000       214
   Vietnamese     0.9524    0.9524    0.9524       105
        Cuban     0.4000    0.0215    0.0408       186
        Tamil     0.5200    0.0828    0.1429       157
  Cameroonian     0.3205    0.2427    0.2762       103
      Ugandan     0.2917    0.2100    0.2442       100
     Peruvian     0.0000    0.0000    0.0000       170
     Tunisian     0.5000    0.2523    0.3353       111
   Macedonian     0.6857    0.5000    0.5783        96
  Singaporean     0.4138    0.0882    0.1455       136
   Senegalese     0.4719    0.5915    0.5250        71
      Burmese     0.7368    0.6829    0.7089        82
    Dominican     0.2000    0.0065    0.0127       153
       Basque     0.5577    0.2990    0.3893        97
       Syrian     0.2432    0.0857    0.1268       105
     Lebanese     0.3750    0.0783    0.1295       115
     Jamaican     0.0000    0.0000    0.0000       133
      Cypriot     0.7273    0.1495    0.2481       107
     Moldovan     0.6071    0.1735    0.2698        98
      Emirati     0.2909    0.2424    0.2645        66
       Afghan     0.4375    0.1750    0.2500        80
    Mongolian     0.9245    0.8305    0.8750        59
    Ethiopian     0.5000    0.4386    0.4673        57
     Nepalese     0.9565    0.2651    0.4151        83
   Paraguayan     0.0000    0.0000    0.0000       101
       Tongan     0.4528    0.5581    0.5000        43
      Zambian     0.3824    0.2653    0.3133        49
    Jordanian     0.3684    0.1167    0.1772        60
   Ecuadorian     0.0000    0.0000    0.0000        78
    Tanzanian     0.2105    0.0690    0.1039        58
       Samoan     0.4667    0.1167    0.1867        60
       Qatari     0.1818    0.0851    0.1159        47
       Malian     0.3333    0.3750    0.3529        32
  Palestinian     0.1000    0.0175    0.0299        57
      Guinean     0.1111    0.0172    0.0299        58
      Maltese     0.0000    0.0000    0.0000        65
     Bolivian     0.0000    0.0000    0.0000        63
     Namibian     0.5000    0.0167    0.0323        60
   Panamanian     0.0000    0.0000    0.0000        59
   Salvadoran     0.0000    0.0000    0.0000        57
      Haitian     0.0000    0.0000    0.0000        58
   Guatemalan     0.0000    0.0000    0.0000        55
     Honduran     0.0000    0.0000    0.0000        53
      Tibetan     0.5417    0.4815    0.5098        27
       Yemeni     0.4545    0.1282    0.2000        39
        Omani     0.2400    0.2609    0.2500        23
    Cambodian     0.8571    0.1538    0.2609        39
      Kuwaiti     0.1875    0.1000    0.1304        30
      Angolan     0.5000    0.0233    0.0444        43
      Rwandan     0.3125    0.1923    0.2381        26
     Botswana     0.5714    0.1212    0.2000        33
       Libyan     0.0000    0.0000    0.0000        36
     Guyanese     0.0000    0.0000    0.0000        36
   Nicaraguan     0.0000    0.0000    0.0000        35
    Burkinabé     0.4000    0.0690    0.1176        29
    Barbadian     0.0000    0.0000    0.0000        33
     Georgian     0.1667    0.1000    0.1250        20
    Mauritian     0.0000    0.0000    0.0000        31
     Sudanese     0.0000    0.0000    0.0000        30
     Malagasy     0.8889    0.3810    0.5333        21
      Gambian     1.0000    0.3182    0.4828        22
     Nigerien     0.3333    0.1579    0.2143        19
    Bhutanese     0.8750    0.3500    0.5000        20
     Gabonese     0.0000    0.0000    0.0000        27
     Bahraini     1.0000    0.0385    0.0741        26
     Togolese     0.5000    0.0400    0.0741        25
   Mozambican     1.0000    0.0435    0.0833        23
    Bermudian     0.0000    0.0000    0.0000        24
     Bahamian     0.0000    0.0000    0.0000        24
      Faroese     0.5000    0.2500    0.3333        16
     Liberian     0.0000    0.0000    0.0000        24
         Manx     0.0000    0.0000    0.0000        23
    Burundian     0.5000    0.1176    0.1905        17
   Surinamese     0.0000    0.0000    0.0000        20
     Beninese     0.1667    0.0714    0.1000        14
  Sammarinese     0.0000    0.0000    0.0000        20
     Eritrean     0.0000    0.0000    0.0000        17
      Slovene     0.6667    0.1333    0.2222        15
    Maldivian     1.0000    0.1333    0.2353        15
     Malawian     0.0000    0.0000    0.0000        17
     Andorran     0.0000    0.0000    0.0000        16
      Bosniak     0.1000    0.2000    0.1333         5
       Aruban     0.0000    0.0000    0.0000        15
Equatoguinean     0.2500    0.1000    0.1429        10
       Breton     0.0000    0.0000    0.0000        13
    Grenadian     0.0000    0.0000    0.0000        13
      Chadian     0.0000    0.0000    0.0000        11
  Mauritanian     1.0000    0.3333    0.5000         9
       Somali     0.2500    0.1429    0.1818         7
 Gibraltarian     0.0000    0.0000    0.0000        11
        Swazi     0.0000    0.0000    0.0000        11
    Vanuatuan     0.0000    0.0000    0.0000        11
     Belizean     0.0000    0.0000    0.0000        10
         Finn     0.0000    0.0000    0.0000        10
     Bruneian     0.0000    0.0000    0.0000         8
     Comorian     0.0000    0.0000    0.0000         7
     Tuvaluan     0.0000    0.0000    0.0000         7
       Syriac     0.0000    0.0000    0.0000         6
   Djiboutian     0.0000    0.0000    0.0000         6
         Turk     0.0000    0.0000    0.0000         6
         Serb     0.0000    0.0000    0.0000         5
      Nauruan     0.0000    0.0000    0.0000         5
   I-Kiribati     0.0000    0.0000    0.0000         4
        Tajik     0.0000    0.0000    0.0000         4
  Marshallese     0.0000    0.0000    0.0000         3
        Sotho     0.0000    0.0000    0.0000         3
        Uzbek     0.0000    0.0000    0.0000         3
      Palauan     0.0000    0.0000    0.0000         3
   Vincentian     0.0000    0.0000    0.0000         3
         Dane     0.0000    0.0000    0.0000         3
       Kyrgyz     0.0000    0.0000    0.0000         2
       Kazakh     0.0000    0.0000    0.0000         1

     accuracy                         0.5567     89025
    macro avg     0.3594    0.2433    0.2625     89025
 weighted avg     0.5209    0.5567    0.5031     89025

2025-02-06 21:54:54,778 ----------------------------------------------------------------------------------------------------
(name2nat) root@hzdocker3 ~/name2nat # python predict.py
Name2nat package location: /root/name2nat/name2nat
New model location: /root/name2nat/resources/best-model.pt
New model timestamp: 2025-02-06 21:54:23.045282
/root/name2nat/predict.py:23: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model_data = torch.load(model_path, map_location='cpu')

Name2nat __init__ location: /root/name2nat/name2nat/name2nat.py
Name2nat __init__ timestamp: 2025-02-06 17:25:51.150131
Found model at: resources/best-model.pt
2025-02-06 22:43:45,379 [b'<unk>', b'a', b'\xe2\x96\x81', b'e', b'r', b'n', b'i', b'o', b'l', b's', b't', b'h', b'u', b'd', b'm', b'c', b'y', b'g', b'M', b'k', b'S', b'A', b'B', b'J', b'b', b'C', b'v', b'R', b'H', b'G', b'D', b'L', b'P', b'K', b'p', b'T', b'f', b'w', b'z', b'W', b'F', b'E', b'.', b'N', b'V', b'I', b'-', b'O', b'j', b'\xc3\xa9', b'Y', b'x', b'\xc3\xa1', b'Z', b'q', b'\xc3\xad', b',', b'U', b"'", b'\xc3\xb3', b'Q', b'\xc3\xbc', b'\xc3\xb6', b'\xc5\xa1', b'\xc3\xa7', b'\xc4\x87', b'1', b'X', b'\xc3\xa8', b'\xc3\xb8', b'\xc4\x8d', b'\xc5\x8d', b'\xc3\xba', b'\xc3\x89', b'\xc4\x83', b'\xc3\xb1', b'\xc3\x81', b'\xc3\xa4', b'2', b'\xc3\xa3', b'&', b';', b'\xc3\xab', b'\xc5\x99', b'\xc5\xa0', b'\xc3\xbd', b'3', b'\xc5\xab', b'\xc5\xbe', b'\xc3\xaf', b'\xc8\x99', b'\xc3\xb4', b'\xc4\x97', b'\xc3\xa5', b'\xc4\x9b', b'\xc3\xa2', b'\xc8\x9b', b'4', b'\xc4\x81', b'\xc3\x9f', b'\xc3\xa6', b'\xc3\x93', b'\xc3\xaa', b'\xc5\x91', b'\xc3\xa0', b'\xc5\x82', b'\xc3\xae', b'5', b'\xc3\x98', b'\xc4\x8c', b'\xe1\xbb\x85', b'\xc4\x90', b'6', b'\xc4\xb1', b'\xc3\xb2', b'7', b'\xc5\x86', b'\xc5\x8c', b'\xc5\xbd', b'\xc8\x98', b'\xc6\xb0', b'8', b'\xc4\xab', b'0', b'\xc3\x96', b'\xc5\x9f', b'\xc3\xb5', b'\xc5\x88', b'\xc4\x9f', b'9', b'\xe1\xbb\x8b', b'\xc6\xa1', b'\xc3\x87', b'\xc4\x93', b'\xe1\xba\xa1', b'\xca\xbb', b'\xe1\xba\xa7', b'\xc5\xaf', b'\xc5\x84', b'\xc4\x91', b'\xc4\xbc', b'\xc3\xb0', b'\xc3\xb9', b'\xc3\x85', b'\xc3\xac', b'\xc5\x9e', b'\xe1\xbb\x8d', b'\xe1\xba\xbf', b'\xc4\xbd', b'\xc4\xb0', b'\xe1\xbb\x87', b'\xe2\x80\x99', b'\xc5\xa9', b'\xe1\xbb\x93', b'\xc3\x9c', b'\xc8\x9a', b'\xe1\xba\xa3', b':', b'\xc5\xa5', b'\xe1\xba\xa5', b'\xc4\x99', b'\xc3\x8d', b'\xe1\xbb\xb3', b'\xc3\x82', b'\xc3\x86', b'\xe1\xbb\xa9', b'\xc5\xb1', b'\xc4\xbe', b'\xe1\xba\xad', b'\xc5\x93', b'\xc4\x86', b'\xc5\x98', b'\xe1\xbb\x9d', b'\xe1\xbb\x81', b'\xc3\xbf', b'\xe2\x80\x93', b'\xc5\x81', b'\xc4\xa3', b'\xe1\xba\xb7', b'\xc5\xbc', b'\xc4\x80', b'\xc3\x9a', b'\xe1\xbb\xaf', b'\xe1\xbb\x91', b'\xe1\xbb\xa5', b'\xc3\x88', b'\xc3\xbb', b'\xe1\xba\xaf', b'\xc5\x9b', b'\xc4\x85', b'\xc4\xb7', b'\xc5\xa3', b'\xe1\xbb\x97', b'\xc4\x8e', b'\xe1\xbb\xa3', b'\xc5\xbb', b'\xc4\xa9', b'\xc4\xb6', b'\xe1\xbb\xad', b'\xc4\x8f', b'\xe1\xbb\xb9', b'\xe1\xbb\x99', b'\xc3\x80', b'\xc5\x9a', b'\xc5\x85', b'\xc4\x92', b'\xca\xbf', b'\xe1\xba\xa9', b'!', b'\xe1\xba\xb1', b'\xe1\xbb\x83', b'\xe1\xbb\xab', b'\xe1\xbb\x95', b'\xe1\xb9\xa3', b'\xc3\x91', b'\xc3\x95', b'\xe1\xbb\x9b', b'\xe1\xbb\xb1', b'\xc9\x99', b'\xc4\xa0', b'\xc3\x9e', b'/', b'\xc5\xba', b'\xe1\xb8\xa5', b'\xc4\xa2', b'\xc7\x8e', b'\xe1\xb9\xad', b'\xe1\xbb\xa7', b'\xc3\x9d', b'\xe2\x80\x9c', b'\xe2\x80\x9d', b'\xe1\xb9\x87', b'\xc3\x84', b'\xe1\xbb\x89', b'\xc5\xad', b'\xe1\xb9\xac', b'$', b'\xc4\x95', b'\xe1\xb8\xa4', b'\xca\xbe', b'\xca\xbc', b'\xc5\xa2', b'`', b'\xc3\x90', b'\xe2\x80\x98', b'(', b'\xc4\x9e', b'+', b'\xe1\xb9\x83', b'\xc5\x90', b'\xc5\xb7', b'\xc5\xaa', b'\xca\xbd', b'\xc5\xa4', b'\xe1\xbb\x9f', b'\xc7\x83', b'\xc4\x8a', b'*', b'\xe2\x80\x91', b'\xc5\x92', b'\xd1\x90', b'\xe1\xb8\xb1', b'@', b'\xe1\xb8\xa0', b'\xe1\xb8\x8d', b'\xe1\xb9\x85', b'\xe2\x88\x92', b'\xc3\xbe', b'\xc5\xb5', b'\xc5\xb3', b'\xc4\x84', b'\xc4\xaa', b'\xc4\xbb', b'\xc2\xb7', b'\xc3\x92', b'\xc3\x8a', b'\xc5\x8f', b'\xe2\x82\xa9', b'\xe1\xba\xab', b'\xe1\xbb\xa1', b'\xe1\xbb\xb7', b'\xe1\xbb\xb5', b'\xe1\xba\xa8', b'\xe1\xbb\x8c', b'\xe1\xb8\xb2', b'\xca\x89', b'\xc5\xb9', b'\xd9\x90', b'\xd1\x97', b'\xc7\xab', b'\xc4\xa1', b'\xd9\x80']
2025-02-06 22:43:45,379 vocabulary size of 292
Batch inference: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 871/871 [01:36<00:00,  9.02it/s](name2nat) root@hzdocker3 ~/name2nat # python eval.py --gt nana/test.tgt --pred test.pred

precision@1=61935/111368=55.60000000000001
precision@2=77862/111368=69.89999999999999
precision@3=87133/111368=78.2
precision@4=93090/111368=83.6
precision@5=97205/111368=87.3
