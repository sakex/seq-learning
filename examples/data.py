design = [
    [0.28757752, 0.96302423],
    [0.78830514, 0.90229905],
    [0.40897692, 0.69070528],
    [0.88301740, 0.79546742], [0.94046728, 0.02461368], [0.04555650, 0.47779597], [0.52810549, 0.75845954],
    [0.89241904, 0.21640794], [0.55143501, 0.31818101], [0.45661474, 0.23162579], [0.95683335, 0.14280002],
    [0.45333416, 0.41454634], [0.67757064, 0.41372433], [0.57263340, 0.36884545], [0.10292468, 0.15244475],
    [0.89982497, 0.13880606], [0.24608773, 0.23303410], [0.04205953, 0.46596245], [0.32792072, 0.26597264],
    [0.95450365, 0.85782772], [0.88953932, 0.04583117], [0.69280341, 0.44220007], [0.64050681, 0.79892485],
    [0.99426978, 0.12189926], [0.65570580, 0.56094798], [0.70853047, 0.20653139], [0.54406602, 0.12753165],
    [0.59414202, 0.75330786], [0.28915974, 0.89504536], [0.14711365, 0.37446278]]

response = [0.94917419, 0.99283162, 0.89106316, 0.9942072, 0.82209501,
            0.49978668, 0.9598775, 0.89517647, 0.76408128, 0.63517921,
            0.891041, 0.76296051, 0.88722508, 0.80842943, 0.25260288,
            0.86171042, 0.46100007, 0.48644999, 0.55959195, 0.97097195,
            0.80481903, 0.90653564, 0.99138406, 0.89842476, 0.93794419,
            0.79260031, 0.62223748, 0.97516169, 0.92619966, 0.49824758]

theta = [1.320117, 1.282131]

k_inv = [[324680.1589, -15379.0248, 909032.9227, 56766.4197, -201396.3388, 169278.1351, -860284.4077, 80548.4898,
          543081.4474, -16036.9073, 43442.5658, -1016973.9189, 213424.0874, 375989.6204, -21320.4145, -349517.1584,
          93158.7895, -117402.2743, 189298.913, -27902.2504, 381956.4717, -121174.8985, 1607.7288, 74527.5347,
          -317038.2679, -161907.3754, -171640.6362, 582086.2612, -526970.443, -113834.6848],
         [-15379.0248, 344461.4801, -620218.2068, 358420.1603, 12271.9024, 477913.1396, 1209733.9211, 33197.1729,
          109777.6247, -253989.9096, -13471.3049, -49799.6435, -81282.5623, 120755.5202, -18347.7679, -97123.5683,
          67804.0549, -463888.7951, 77915.1065, -241008.377, -13721.9075, -179990.3524, -1887973.4459, 38412.0449,
          62752.9022, 134125.5327, 20740.6652, 830878.7114, 20781.5861, 16202.382],
         [909032.9227, -620218.2068, 4647808.9288, -491395.3506, -1189746.1714, -2235996.5967, -5775262.4401,
          146685.1725, 2211762.148, 357223.3596, 298446.0666, -4069938.5871, 575849.8771, 1659635.879, -112750.7931,
          -1065573.5587, 349998.4361, 2354709.9554, 416522.3735, 312922.492, 1986086.9972, 258207.0737, 3818528.0723,
          221649.3197, -1241190.3604, -1521717.598, -582214.3945, 301399.416, -1497899.1301, -422449.5309],
         [56766.4197, 358420.1603, -491395.3506, 856466.7553, -265903.8723, 381193.2881, 1688254.6472, -470778.4002,
          88908.3663, -949380.582, 216919.9256, -211888.3141, 629154.5522, 1173379.7263, -151658.6301, 137182.1653,
          983986.7778, -275109.2148, -476984.262, -486349.1899, 228430.8386, -479010.566, -2120815.0784, 54833.7253,
          -731557.2809, -176056.0193, 225274.3905, 595250.427, -130989.1097, -256832.5223],
         [-201396.3388, 12271.9024, -1189746.1714, -265903.8723, 3463510.108, 140426.2067, 193844.3986, 2045898.7712,
          32170.4733, -1547478.5161, 1181906.0409, 1797138.4285, -2371356.0748, 120829.3806, 16934.7631, -361510.5736,
          94337.3577, -270045.9042, -468628.0706, 125116.2647, -4237104.1785, -1739231.1991, -238668.3634,
          -2065550.6101, 2425425.1011, 1341969.5121, 1040187.5969, 173483.1157, 454444.5774, 297061.8863],
         [169278.1351, 477913.1396, -2235996.5967, 381193.2881, 140426.2067, 8870636.3209, 3964841.1156, 165608.2942,
          1110965.6858, 1804094.3323, 243877.7431, -1610059.9098, -282996.9425, 433360.8597, 413498.2052, 745218.5062,
          -1901704.559, -9429716.4885, -201250.9959, -203536.2946, -176178.1933, -397512.852, -3867554.0162,
          -545014.3569, 512593.6066, -1397160.5471, -144764.6606, 1305493.271, -314791.6307, 1969253.729],
         [-860284.4077, 1209733.9211, -5775262.4401, 1688254.6472, 193844.3986, 3964841.1156, 12681389.8814,
          -1527493.206, -1061382.3038, -1773438.8715, 288108.0997, 2072383.4419, -555118.7371, 40846.2523, -16122.5995,
          1627788.8436, 157538.4231, -3917088.5821, 403639.9755, -1017722.049, -1068419.1836, -1668380.9749,
          -5044534.4992, -57595.6421, 3359739.4084, 1297987.6835, 557520.4542, -6563884.1818, 1198337.8912,
          164109.0086],
         [80548.4898, 33197.1729, 146685.1725, -470778.4002, 2045898.7712, 165608.2942, -1527493.206, 5819659.6858,
          919273.9283, -293958.7666, -6245302.909, -54210.9423, -2464937.2186, 1862992.1475, -70325.7023, -421091.6908,
          832689.3841, -225371.9197, -1341740.2812, 175556.3927, -1729366.4249, -4126924.4214, 272380.0555,
          1944269.5363, 3698673.7533, 192759.1606, 113638.1446, 558232.2175, 5903.7539, 104166.7139],
         [543081.4474, 109777.6247, 2211762.148, 88908.3663, 32170.4733, 1110965.6858, -1061382.3038, 919273.9283,
          21619210.7137, -7557604.896, -122148.0665, -4333520.4404, -1968294.9831, -9513398.3794, -322631.9039,
          1981170.801, 2304807.326, -1164221.7246, -719669.1935, -115273.3606, -46795.8326, 270437.4384, 286891.8141,
          -909747.8303, 1505079.7986, -5615082.641, 1920335.5387, -503028.5659, -954956.8664, 3568.393],
         [-16036.9073, -253989.9096, 357223.3596, -949380.582, -1547478.5161, 1804094.3323, -1773438.8715, -293958.7666,
          -7557604.896, 14776670.2924, -869205.8313, 1642360.7578, 2639183.4701, -885021.7147, 213065.4409, 828626.6238,
          80535.2143, -2534475.3234, -8725462.0101, 505533.2518, 2287797.1359, 1057530.1265, 1575485.2515, 424796.7228,
          -882355.5823, -1437685.6494, -3305611.8538, 366599.3266, 55588.1726, 2416395.0543],
         [43442.5658, -13471.3049, 298446.0666, 216919.9256, 1181906.0409, 243877.7431, 288108.0997, -6245302.909,
          -122148.0665, -869205.8313, 22010556.9709, -492645.7396, 997594.5917, -464030.158, -41547.9291, -8096070.5034,
          48851.0382, -361055.3249, 230881.3528, -96435.6715, -1008982.0971, 1210724.4819, 145027.198, -10635114.2786,
          -996407.8109, 2630851.4282, 235666.3306, -502460.4106, -111541.8897, 273056.9151],
         [-1016973.9189, -49799.6435, -4069938.5871, -211888.3141, 1797138.4285, -1610059.9098, 2072383.4419,
          -54210.9423, -4333520.4404, 1642360.7578, -492645.7396, 10409892.6956, 100200.5113, -7623440.5275,
          -179204.7708, -578184.9581, 2757245.657, 1855760.1529, -4807336.8746, 129637.3909, -2584233.8662,
          1486642.5015, -537442.8762, 313951.825, -159270.202, 4876644.3432, -204737.7387, 205313.4017, 1808312.7629,
          -942237.8531],
         [213424.0874, -81282.5623, 575849.8771, 629154.5522, -2371356.0748, -282996.9425, -555118.7371, -2464937.2186,
          -1968294.9831, 2639183.4701, 997594.5917, 100200.5113, 20767605.5714, -5273321.5409, 134401.117, -167252.1623,
          -1331933.8414, 329447.6557, 1142410.7733, -221555.0549, 2969941.8341, -12529283.6432, -983405.6476,
          629646.3285, -2420319.4019, -1494505.0102, -681624.9878, 2080684.3253, -404815.0547, 22281.7246],
         [375989.6204, 120755.5202, 1659635.879, 1173379.7263, 120829.3806, 433360.8597, 40846.2523, 1862992.1475,
          -9513398.3794, -885021.7147, -464030.158, -7623440.5275, -5273321.5409, 22117315.1456, -237382.0956,
          89910.4102, 63413.6849, -594100.9207, 1526253.1292, -606125.6932, 124450.8727, -1834914.2647, -350809.196,
          -395496.2596, 255600.3047, -2114943.4754, 818979.8452, -794571.0663, -682991.2047, 586279.3253],
         [-21320.4145, -18347.7679, -112750.7931, -151658.6301, 16934.7631, 413498.2052, -16122.5995, -70325.7023,
          -322631.9039, 213065.4409, -41547.9291, -179204.7708, 134401.117, -237382.0956, 156680.9846, 152212.7385,
          -1079204.6536, -469862.814, 1082863.0843, 82345.209, -50218.7258, 109286.5758, 25274.9662, -7467.8287,
          91587.7711, 8125.6146, -44863.8774, 102997.7791, 37549.8329, 196189.6827],
         [-349517.1584, -97123.5683, -1065573.5587, 137182.1653, -361510.5736, 745218.5062, 1627788.8436, -421091.6908,
          1981170.801, 828626.6238, -8096070.5034, -578184.9581, -167252.1623, 89910.4102, 152212.7385, 9272813.2764,
          -1129630.8844, -1103927.3453, -725556.4447, 1461.9158, -1111614.4673, 1903338.4465, -27105.9412, 2802621.1584,
          -1101527.3444, -5250088.7573, 1177199.2981, -861374.5056, 522560.2105, 1204409.8257],
         [93158.7895, 67804.0549, 349998.4361, 983986.7778, 94337.3577, -1901704.559, 157538.4231, 832689.3841,
          2304807.326, 80535.2143, 48851.0382, 2757245.657, -1331933.8414, 63413.6849, -1079204.6536, -1129630.8844,
          8205593.6103, 2214394.7519, -9720759.5941, -532949.0548, 160209.7748, -300537.9422, 9871.582, 49762.3023,
          -621244.3329, 151339.1859, -9403.0891, -710149.4895, -145458.5634, -1143126.8898],
         [-117402.2743, -463888.7951, 2354709.9554, -275109.2148, -270045.9042, -9429716.4885, -3917088.5821,
          -225371.9197, -1164221.7246, -2534475.3234, -361055.3249, 1855760.1529, 329447.6557, -594100.9207,
          -469862.814, -1103927.3453, 2214394.7519, 10148860.8578, 632801.694, 146931.3885, 366932.1254, 484227.8689,
          3819984.3693, 801273.7743, -696575.3253, 1922951.6981, 121683.3078, -1359981.7623, 223298.0846,
          -2440382.7663],
         [189298.913, 77915.1065, 416522.3735, -476984.262, -468628.0706, -201250.9959, 403639.9755, -1341740.2812,
          -719669.1935, -8725462.0101, 230881.3528, -4807336.8746, 1142410.7733, 1526253.1292, 1082863.0843,
          -725556.4447, -9720759.5941, 632801.694, 18290501.9997, 264361.5936, 296333.5195, -153319.2881, -719577.1094,
          782655.5034, 461419.0526, 2132902.8279, 1129777.6529, 481332.6149, -379681.5878, -1101033.1745],
         [-27902.2504, -241008.377, 312922.492, -486349.1899, 125116.2647, -203536.2946, -1017722.049, 175556.3927,
          -115273.3606, 505533.2518, -96435.6715, 129637.3909, -221555.0549, -606125.6932, 82345.209, 1461.9158,
          -532949.0548, 146931.3885, 264361.5936, 287325.1491, -121168.1073, 288294.1444, 1332132.7837, -37395.4868,
          281823.5789, 50217.3555, -101825.3671, -372876.3209, 64779.2232, 133842.8336],
         [381956.4717, -13721.9075, 1986086.9972, 228430.8386, -4237104.1785, -176178.1933, -1068419.1836,
          -1729366.4249, -46795.8326, 2287797.1359, -1008982.0971, -2584233.8662, 2969941.8341, 124450.8727,
          -50218.7258, -1111614.4673, 160209.7748, 366932.1254, 296333.5195, -121168.1073, 5619224.0107, 1374346.5293,
          409489.5862, 2413734.3299, -2638281.5693, -1207919.0953, -1632008.4553, 238547.2523, -765160.556,
          -466553.0815],
         [-121174.8985, -179990.3524, 258207.0737, -479010.566, -1739231.1991, -397512.852, -1668380.9749,
          -4126924.4214, 270437.4384, 1057530.1265, 1210724.4819, 1486642.5015, -12529283.6432, -1834914.2647,
          109286.5758, 1903338.4465, -300537.9422, 484227.8689, -153319.2881, 288294.1444, 1374346.5293, 18288514.6259,
          -72871.6328, 337233.3244, -6155687.5569, 348386.8358, -534695.8019, 3021248.0957, 124818.6926, -269809.6341],
         [1607.7288, -1887973.4459, 3818528.0723, -2120815.0784, -238668.3634, -3867554.0162, -5044534.4992,
          272380.0555, 286891.8141, 1575485.2515, 145027.198, -537442.8762, -983405.6476, -350809.196, 25274.9662,
          -27105.9412, 9871.582, 3819984.3693, -719577.1094, 1332132.7837, 409489.5862, -72871.6328, 12709439.1652,
          -186520.3179, 2617346.5898, -485534.6661, -426124.1911, -9801471.8695, -47970.7245, -224917.4158],
         [74527.5347, 38412.0449, 221649.3197, 54833.7253, -2065550.6101, -545014.3569, -57595.6421, 1944269.5363,
          -909747.8303, 424796.7228, -10635114.2786, 313951.825, 629646.3285, -395496.2596, -7467.8287, 2802621.1584,
          49762.3023, 801273.7743, 782655.5034, -37395.4868, 2413734.3299, 337233.3244, -186520.3179, 6210827.3795,
          -677775.1414, -278148.1667, -713079.3561, 284254.1023, -150272.8782, -724989.5353],
         [-317038.2679, 62752.9022, -1241190.3604, -731557.2809, 2425425.1011, 512593.6066, 3359739.4084, 3698673.7533,
          1505079.7986, -882355.5823, -996407.8109, -159270.202, -2420319.4019, 255600.3047, 91587.7711, -1101527.3444,
          -621244.3329, -696575.3253, 461419.0526, 281823.5789, -2638281.5693, -6155687.5569, 2617346.5898,
          -677775.1414, 8193154.188, 1376726.3136, 238696.1913, -7492932.6225, 565959.7949, 485854.3204],
         [-161907.3754, 134125.5327, -1521717.598, -176056.0193, 1341969.5121, -1397160.5471, 1297987.6835, 192759.1606,
          -5615082.641, -1437685.6494, 2630851.4282, 4876644.3432, -1494505.0102, -2114943.4754, 8125.6146,
          -5250088.7573, 151339.1859, 1922951.6981, 2132902.8279, 50217.3555, -1207919.0953, 348386.8358, -485534.6661,
          -278148.1667, 1376726.3136, 7264941.746, -729770.422, -534799.2503, 352652.8059, -1676549.0495],
         [-171640.6362, 20740.6652, -582214.3945, 225274.3905, 1040187.5969, -144764.6606, 557520.4542, 113638.1446,
          1920335.5387, -3305611.8538, 235666.3306, -204737.7387, -681624.9878, 818979.8452, -44863.8774, 1177199.2981,
          -9403.0891, 121683.3078, 1129777.6529, -101825.3671, -1632008.4553, -534695.8019, -426124.1911, -713079.3561,
          238696.1913, -729770.422, 1329127.5429, 72312.7568, 293480.3384, -12316.7389],
         [582086.2612, 830878.7114, 301399.416, 595250.427, 173483.1157, 1305493.271, -6563884.1818, 558232.2175,
          -503028.5659, 366599.3266, -502460.4106, 205313.4017, 2080684.3253, -794571.0663, 102997.7791, -861374.5056,
          -710149.4895, -1359981.7623, 481332.6149, -372876.3209, 238547.2523, 3021248.0957, -9801471.8695, 284254.1023,
          -7492932.6225, -534799.2503, 72312.7568, 18700692.9268, -684461.9422, 281701.3495],
         [-526970.443, 20781.5861, -1497899.1301, -130989.1097, 454444.5774, -314791.6307, 1198337.8912, 5903.7539,
          -954956.8664, 55588.1726, -111541.8897, 1808312.7629, -404815.0547, -682991.2047, 37549.8329, 522560.2105,
          -145458.5634, 223298.0846, -379681.5878, 64779.2232, -765160.556, 124818.6926, -47970.7245, -150272.8782,
          565959.7949, 352652.8059, 293480.3384, -684461.9422, 871538.2278, 197839.7986],
         [-113834.6848, 16202.382, -422449.5309, -256832.5223, 297061.8863, 1969253.729, 164109.0086, 104166.7139,
          3568.393, 2416395.0543, 273056.9151, -942237.8531, 22281.7246, 586279.3253, 196189.6827, 1204409.8257,
          -1143126.8898, -2440382.7663, -1101033.1745, 133842.8336, -466553.0815, -269809.6341, -224917.4158,
          -724989.5353, 485854.3204, -1676549.0495, -12316.7389, 281701.3495, 197839.7986, 1442650.7899]]
